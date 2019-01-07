# Copyright 2018 Daniel Hernandez Diaz, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import numpy as np
import tensorflow as tf

from neurolib.encoder import MultivariateNormalFullCovariance  # @UnresolvedImport
from neurolib.encoder.basic import InnerNode
from _collections import defaultdict

# pylint: disable=bad-indentation, no-member


def blk_tridiag_chol(A_Txdxd, B_Tm1xdxd):
    """
    Compute the Cholesky decomposition of a symmetric, positive definite
    block-tridiagonal matrix.

    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix 
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper) 1st block 
        off-diagonal matrix

    Outputs: 
    R - python list with two elements
        * R[0] - [T x n x n] tensor of block diagonal elements of Cholesky decomposition
        * R[1] - [T-1 x n x n] tensor of (lower) 1st block off-diagonal elements of Cholesky
    """
    def compute_chol(LC, AB_2xdxd):
        L_dxd = LC[0]
        A_dxd, B_dxd = AB_2xdxd[0], AB_2xdxd[1]
        C_dxd = tf.matmul(B_dxd, tf.matrix_inverse(L_dxd), 
                      transpose_a=True, transpose_b=True)
        D = A_dxd - tf.matmul(C_dxd, C_dxd, transpose_b=True)
        L_dxd = tf.cholesky(D)
        return [L_dxd, C_dxd]
        
    L1_dxd = tf.cholesky(A_Txdxd[0])
    C1_dxd = tf.zeros_like(B_Tm1xdxd[0], dtype=tf.float64)
    
    result_2xTm1xdxd = tf.scan(fn=compute_chol, elems=[A_Txdxd[1:], B_Tm1xdxd],
                               initializer=[L1_dxd, C1_dxd])

    AChol_Txdxd = tf.concat([tf.expand_dims(L1_dxd, 0), result_2xTm1xdxd[0]], 
                            axis=0)    
    BChol_Tm1xdxd = result_2xTm1xdxd[1]
    
    return [AChol_Txdxd, BChol_Tm1xdxd]

def blk_chol_inv(A_Txdxd, B_Tm1xdxd, b_Txd, lower=True, transpose=False):
    """
    Solve the equation Cx = b for x, where C is assumed to be a block-bi-
    diagonal triangular matrix - only the first lower/upper off-diagonal block
    is nonvanishing.
    
    This function will be used to solve the equation Mx = b where M is a
    block-tridiagonal matrix due to the fact that M = C^T*C where C is block-
    bidiagonal triangular.
    
    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix 
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper or lower) 
        1st block off-diagonal matrix
     
    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the 
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve 
          the problem C^T x = b with a representation of C.) 
 
    Outputs: 
    x - solution of Cx = b
    """
    # Define a matrix-vector dot product because the tensorflow developers feel
    # this is beneath them.
    tf_dot = lambda M, v : tf.reduce_sum(tf.multiply(M, v), axis=1)
    if transpose:
        A_Txdxd = tf.transpose(A_Txdxd, [0,2,1])
        B_Tm1xdxd = tf.transpose(B_Tm1xdxd, [0,2,1])
    
    # Whether B is lower or upper doesn't matter. The function to be passed to
    # scan is the same.
    def step(x_d, ABb_2x_):
        A_dxd, B_dxd, b_d = ABb_2x_[0], ABb_2x_[1], ABb_2x_[2]
        return tf_dot(tf.matrix_inverse(A_dxd),
                         b_d - tf_dot(B_dxd, x_d))
    if lower:
        x0_d = tf_dot(tf.matrix_inverse(A_Txdxd[0]), b_Txd[0])
        result_Tm1xd = tf.scan(fn=step, elems=[A_Txdxd[1:], B_Tm1xdxd, b_Txd[1:]], 
                             initializer=x0_d)
        result_Txd = tf.concat([tf.expand_dims(x0_d, axis=0), result_Tm1xd], axis=0)
    else:
        xN_d = tf_dot(tf.matrix_inverse(A_Txdxd[-1]), b_Txd[-1])
        result_Tm1xd = tf.scan(fn=step, 
                             elems=[A_Txdxd[:-1][::-1], B_Tm1xdxd[::-1], b_Txd[:-1][::-1]],
                             initializer=xN_d )
        result_Txd = tf.concat([tf.expand_dims(xN_d, axis=0), result_Tm1xd],
                               axis=0)[::-1]

    return result_Txd 
  

class MergeNormals(InnerNode):
  """
  
  """
  def __init__(self,
               builder,
               state_size,
               node_list,
               num_extra_mergers=0,
               name=None,
               **dirs):
    """
    Initialize the MergeNormals Node
    """
    super(MergeNormals, self).__init__(builder)
    
    self.name = "MergeNormals_" + str(self.label) if name is None else name
    self.num_mergers = len(node_list)
    self.node_list = node_list
    self.state_size = state_size
    self.main_dim = state_size[0][0]
    self.num_extra_mergers = num_extra_mergers
    
    num_mergers = len(node_list) + num_extra_mergers
    self.num_expected_inputs = 2*num_mergers
    self.num_expected_outputs = 3
    self.num_declared_mergers = 0
    self.free_islots = list(range(self.num_expected_inputs))
    self.free_oslots = list(range(self.num_expected_outputs))
    
    self.cov_means = {}
        
    self.dist = None

    # Slot names
    self.oslot_to_name[1] = 'loc_' + str(self.label) + '_1'
    self.oslot_to_name[2] = 'scale_' + str(self.label) + '_2'

    self._update_directives(**dirs)
    
    self._add_links_from_mergers()
    
  def _add_links_from_mergers(self):
    """
    """
    islot = 0
    for node in self.node_list:
      for oslot, name in node.oslot_to_name.items():
        if any([i in name.split('_') for i in ['loc', 'scale']]):
          self.builder.addDirectedLink(node, self,
                                       islot=islot,
                                       oslot=oslot)
          islot += 1
          
      self.num_declared_mergers += 1
      
  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    """
    InnerNode.__call__(self, inputs=inputs, islot_to_itensor=islot_to_itensor)
      
  def update_when_linked_as_node2(self):
    """
    TODO: Add update to self.covs
    """
    pass
      
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    self.directives.update({'output_1_name' : 'loc',
                            'output_2_name' : 'scale'})
    self.directives.update(dirs)
  
  def _get_mean_covariance(self):
    """
    Get the total covariance from the input scales
    """
    md = self.main_dim
    for islot in range(self.num_expected_inputs):
      name_split = self.islot_to_name[islot].split('_')
      dist_id = '_'.join(name_split[1:-1])
      if dist_id not in self.cov_means:
        self.cov_means[dist_id] = [tf.zeros([1, md, md], dtype=tf.float64),
                                  tf.zeros([1, md, 1], dtype=tf.float64)]
      if 'scale' in name_split: 
        sc = self.get_input(islot)
        print("sc", sc)
        cov = tf.matmul(sc, sc, transpose_b=True)
        self.cov_means[dist_id][0] = cov
      if 'loc' in name_split:
        mn = self.get_input(islot)
#         print("mn", mn)
        mn = tf.expand_dims(mn, axis=2)
        self.cov_means[dist_id][1] = mn
    
    print("self.cov_means", self.cov_means)
    print("list(zip(*self.cov_means.values()))[0]", list(zip(*self.cov_means.values()))[0])
    total_cov = sum([cov for cov in list(zip(*self.cov_means.values()))[0]])
    print("total_cov", total_cov)
    post_mean = sum([tf.matmul(tf.matrix_inverse(cov_mean[0]), cov_mean[1]) 
                     for cov_mean in self.cov_means.values()])
    print("post_mean", post_mean)
    post_mean = tf.matmul(tf.matrix_inverse(total_cov), post_mean)
    print("post_mean", post_mean)
    post_mean = tf.squeeze(post_mean, axis=2)
        
    return post_mean, total_cov
      
  def _build(self):
    """
    Builds the merged Normal distribution
    """
    mean, cov = self._get_mean_covariance()
    self.dist = MultivariateNormalFullCovariance(loc=mean,
                                                 covariance_matrix=cov)
    samp = self.dist.sample(name='Out' + str(self.label) + '_0')
    
    o0_name = self.directives['output_0_name']
    self._oslot_to_otensor[0] = tf.identity(samp, name=o0_name)
    o1_name = self.directives['output_1_name']
    self._oslot_to_otensor[1] = tf.identity(mean, name=o1_name)
    o2_name = self.directives['output_2_name']
#     print("self._oslot_to_otensor[1]", self._oslot_to_otensor[1])
    self._oslot_to_otensor[2] = tf.identity(self.dist.scale.to_dense(), name=o2_name)
    print("self._oslot_to_otensor[2]", self._oslot_to_otensor[2])
    
    self._is_built = True


class MergeSeqsNormalLDSEv(InnerNode):
  """
  """
  def __init__(self,
               builder,
               state_size,
               node_list,
#                input_series,
#                ev_seq,
               name=None,
               **dirs):
    """
    """
    super(MergeSeqsNormalLDSEv, self).__init__(builder)
    
    self.state_size = state_size
    self.input_series, self.ev_seq = node_list
    self.prior = self.ev_seq.get_init_inodes()[0]

    self.state_sizes = self.ev_seq.state_sizes
    self.main_oshape = self.get_state_full_shapes()
    self.D = self.get_state_size_ranks()
    self._oslot_to_shape[0] = self.main_oshape
          
    self.name = "MergeSeqs_" + str(self.label) if name is None else name
    self.main_dim = self.state_sizes[0][0]
    
    self.num_expected_inputs = 5
    self.num_expected_outputs = 4

    self.free_islots = list(range(self.num_expected_inputs))
    self.free_oslots = list(range(self.num_expected_outputs))
    
    self.cov_means = {}
        
    self.dist = None

    # Slot names
    self.oslot_to_name[1] = 'loc' + str(self.label) + '_1'
    self.oslot_to_name[2] = 'chol-diag' + str(self.label) + '_2'
    self.oslot_to_name[3] = 'chol-offdiag' + str(self.label) + '_3'

    self._update_directives(**dirs)
    
    self._add_secondary_links()
    
    self._add_links_from_mergers()

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    self.directives = {'output_loc_name' : self.name + '_' + str(1) + '_loc',
                      'output_scale_name' : self.name + '_' + str(2) + '_scale'}
    self.directives.update(dirs)
    
  def _add_secondary_links(self):
    """
    """
    # Loc oslot
    self._oslot_to_shape[1] = self.main_oshape
    o1 = self.builder.addOutput(name=self.directives['output_loc_name'])
    self.builder.addDirectedLink(self, o1, oslot=1)
    
  def _add_links_from_mergers(self):
    """
    """
    islot = 0
    self.islot_collections = defaultdict(list)
#     for node in self.node_list:
    for oslot, name in self.input_series.oslot_to_name.items():
      print("merge; oslot, name", oslot, name)
      if any([i in name.split('_') for i in ['loc', 'precision']]):
        self.builder.addDirectedLink(self.input_series,
                                     self,
                                     islot=islot,
                                     oslot=oslot)
        self.islot_collections[self.input_series.name].append(islot)
        islot += 1
    for oslot, name in self.ev_seq.oslot_to_name.items():
      print("merge; oslot, name", oslot, name)
      if any([i in name.split('_') for i in ['A', 'invQ']]):
        self.builder.addDirectedLink(self.ev_seq,
                                     self,
                                     islot=islot,
                                     oslot=oslot)
        self.islot_collections[self.ev_seq.name].append(islot)
        islot += 1
    for oslot, name in self.prior.oslot_to_name.items():
      print("merge; oslot, name", oslot, name)
      if any([i in name.split('_') for i in ['scale']]):
        self.builder.addDirectedLink(self.prior,
                                     self,
                                     islot=islot,
                                     oslot=oslot)
        self.islot_collections[self.prior.name].append(islot)
#         islot += 1
    
  def _get_loc_inv_scale(self):
      """
      Compute the Cholesky decomposition for the full precision matrix K of the
      Recognition Model. The entries of this matrix are
      
      K_00 = A*Q^-1*A.T
      K_TT = Q^-1
      K_ii (diagonal terms): A*Q^-1*A.T + Q^-1 + Lambda  for i = 1,...,T-1
      
      K_{i,i-1} (off-diagonal terms): A*Q^-1 for i = 1,..., T
      
      Args:
      
      Returns:
          A pair containing:
          
          - TheChol_2xxNxTxdxd: 2 NxTxdxd tensors representing the Cholesky
              decomposition of the full precision matrix. For each batch, the
              first tensor contains the T elements in the diagonal while the
              second tensor contains the T-1 elements in the lower diagonal. The
              remaining entries in the Cholesky decomposition of this
              tri-diagonal precision matrix vanish.
          - checks: A list containing the tensors that form the full precision
              matrix
      """
      input_series_name = self.input_series.name
      for islot in self.islot_collections[input_series_name]:
        if 'precision' in self.islot_to_name[islot].split('_'):
          lmbda_NxTxdxd = self.get_input(islot)
        elif 'loc' in self.islot_to_name[islot].split('_'):
          mu_NxTxd = self.get_input(islot)
          mu_NxTxdx1 = tf.expand_dims(mu_NxTxd, axis=-1)
      lmbdaMu_NxTxd = tf.squeeze(tf.matmul(lmbda_NxTxdxd, mu_NxTxdx1), axis=-1)
      print('\n GOOD \n')
#       if InputY: _, Lambda_NxTxdxd, self.LambdaMu_NxTxd = self.get_Mu_Lambda(InputY)
#       else: Lambda_NxTxdxd = self.Lambda_NxTxdxd
      
#       if InputX is None and Ids is not None:
#         raise ValueError("Must provide an Input for these Ids")
#       X_NxTxd = self.X if InputX is None else InputX
#       if Ids is None: Ids = self.Ids
      
      Nsamps = tf.shape(mu_NxTxd)[0]
      NTbins = self.max_steps
#       NTbins = tf.shape(_NxTxd)[1]
      xDim = self.main_dim
      
      # Simone Biles level tensorflow gymnastics in the next 150 lines or so
      ev_seq_name = self.ev_seq.name
      for islot in self.islot_collections[ev_seq_name]:
        if 'A' in self.islot_to_name[islot].split('_'):
          A_NxTxdxd = self.get_input(islot)
        if 'invQ' in self.islot_to_name[islot].split('_'):
          invQ_NxTxdxd = self.get_input(islot)
      for islot in self.islot_collections[self.prior.name]:
        if 'scale' in self.islot_to_name[islot].split('_'):
          Q0scale_Nxdxd = self.get_input(islot)
          Q0_Nxdxd = tf.matmul(Q0scale_Nxdxd, Q0scale_Nxdxd, transpose_b=True)
          invQ0_Nxdxd = tf.matrix_inverse(Q0_Nxdxd)
          invQ0_Nx1xdxd = tf.expand_dims(invQ0_Nxdxd, axis=1)
      print("normal; A_NxTxdxd", A_NxTxdxd)
      print("normal; invQ_NxTxdxd", invQ_NxTxdxd)
      print("normal; invQ0_Nx1xdxd", invQ0_Nx1xdxd)
      
#       A_NxTxdxd = (self.lat_ev_model.A_NxTxdxd if InputX is None 
#                    else self.lat_ev_model._define_evolution_network_wi(InputX, Ids)[0])
      A_NTm1xdxd = tf.reshape(A_NxTxdxd[:,:-1,:,:], [Nsamps*(NTbins-1), xDim, xDim])
      invQ_NxTm2xdxd = invQ_NxTxdxd[:,:-2,:,:]
      invQ_NTm1xdxd = tf.reshape(invQ_NxTxdxd[:,:-1,:,:], [Nsamps*(NTbins-1), xDim, xDim])
      invQ0Q_NxTm1xdxd = tf.concat([invQ0_Nx1xdxd, invQ_NxTm2xdxd], axis=1)
      invQ0Q_NTm1xdxd = tf.reshape(invQ0Q_NxTm1xdxd, [Nsamps*(NTbins-1), xDim, xDim])
#       self.A_NTm1xdxd = A_NTm1xdxd = tf.reshape(A_NxTxdxd[:,:-1,:,:],
#                                                 [Nsamps*(NTbins-1), xDim, xDim])
      print("normal; invQ0Q_NTm1xdxd", invQ0Q_NTm1xdxd)
      
#       QInv_dxd = self.lat_ev_model.QInv_dxd
#       Q0Inv_dxd = self.lat_ev_model.Q0Inv_dxd
      
      # Constructs the block diagonal matrix:
      #     Qt^-1 = diag{Q0^-1, Q^-1, ..., Q^-1}
#       self.QInvs_NTm1xdxd = QInvs_NTm1xdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=0),
#                                                      [Nsamps*(NTbins-1), 1, 1])
#       QInvs_Tm2xdxd = tf.tile(tf.expand_dims(QInv_dxd, axis=0), [NTbins-2, 1, 1])
#       Q0Inv_1xdxd = tf.expand_dims(Q0Inv_dxd, axis=0)
#       Q0QInv_Tm1xdxd = tf.concat([Q0Inv_1xdxd, QInvs_Tm2xdxd], axis=0)
#       QInvsTot_NTm1xdxd = tf.tile(Q0QInv_Tm1xdxd, [Nsamps, 1, 1])

#       use_tt = self.params.use_transpose_trick
      use_tt = False
      
      
      # The off-diagonal blocks of K(z):
      #     K(z)_{i,i+1} = -A(z)^T*Q^-1,     for i in {1,..., T-2}
      AinvQ_NTm1xdxd = -tf.matmul(A_NTm1xdxd, invQ_NTm1xdxd,  #pylint: disable=invalid-unary-operand-type
                                   transpose_a=use_tt)  
      # The diagonal blocks of K(z) up to T-1:
      #     K(z)_ii = A(z)^T*Qq^{-1}*A(z) + Qt^{-1},     for i in {1,...,T-1 }
      AinvQA_NTm1xdxd = (invQ0Q_NTm1xdxd - 
                         tf.matmul(AinvQ_NTm1xdxd, A_NTm1xdxd, transpose_b=not use_tt))
                         
#       AinvQA_NTm1xdxd = (tf.matmul(A_NTm1xdxd, 
#                                     tf.matmul(invQ_NTm1xdxd, A_NTm1xdxd,
#                                               transpose_b=not use_tt),
#                                     transpose_a=use_tt) + invQ0Q_NTm1xdxd)
      AinvQA_NxTm1xdxd = tf.reshape(AinvQA_NTm1xdxd,
                                     [Nsamps, NTbins-1, xDim, xDim])                                     
      
      
      # Tile in the last block K_TT. 
      # This one does not depend on A. There is no latent evolution beyond T.
#       QInvs_Nx1xdxd = tf.tile(tf.reshape(QInv_dxd, shape=[1, 1, xDim, xDim]),
#                               [Nsamps, 1, 1, 1])
      invQ_Nx1xdxd = invQ_NxTm2xdxd = invQ_NxTxdxd[:,:1,:,:]
      AinvQAinvQ_NxTxdxd = tf.concat([AinvQA_NxTm1xdxd, invQ_Nx1xdxd], axis=1)
      
      # Add in the covariance coming from the observations
      AA_NxTxdxd = lmbda_NxTxdxd + AinvQAinvQ_NxTxdxd
      BB_NxTm1xdxd = tf.reshape(AinvQ_NTm1xdxd, [Nsamps, NTbins-1, xDim, xDim])        
      
      print("normal; AA_NxTxdxd", AA_NxTxdxd)
      print("normal; BB_NxTm1xdxd", BB_NxTm1xdxd)
      
      # Compute the Cholesky decomposition for the total precision matrix
      aux_fn1 = lambda _, seqs : blk_tridiag_chol(seqs[0], seqs[1])
      TheChol_2xxNxTxdxd = tf.scan(fn=aux_fn1, 
                                   elems=[AA_NxTxdxd, BB_NxTm1xdxd],
                                   initializer=[tf.zeros_like(AA_NxTxdxd[0]), 
                                                tf.zeros_like(BB_NxTm1xdxd[0])])
      
      
      checks = [A_NxTxdxd, AA_NxTxdxd, BB_NxTm1xdxd]
#       A_NTm1xdxd = self.A_NTm1xdxd
#       LambdaMu_NxTxd = self.LambdaMu_NxTxd
      
      def postX_from_chol(tc1, tc2, lm):
          """
          postX = (Lambda1 + S)^{-1}.(Lambda1_ij.*Mu_j + X^T_k.*S_kj;i.*X_j)
          """
          return blk_chol_inv(tc1, tc2, blk_chol_inv(tc1, tc2, lm), 
                              lower=False, transpose=True)
      aux_fn2 = lambda _, seqs : postX_from_chol(seqs[0], seqs[1], seqs[2])
      postX = tf.scan(fn=aux_fn2, 
                      elems=[TheChol_2xxNxTxdxd[0], TheChol_2xxNxTxdxd[1], lmbdaMu_NxTxd],
                      initializer=tf.zeros_like(lmbdaMu_NxTxd[0], dtype=tf.float64) )      

      print("normal; TheChol_2xxNxTxdxd", TheChol_2xxNxTxdxd)
      print("normal; postX", postX)
#       assert False
      return postX, TheChol_2xxNxTxdxd, checks
  
  def get_sample(self):
    """
    Sample from the posterior
    """
    assert all([i in self._oslot_to_otensor for i in [1, 2, 3]]), (
            "Outputs must be built before samples can be drawn.")
    
    mean = self.get_output(1)
    inv_scale_d = self.get_output(2)
    inv_scale_od = self.get_output(3)

    Nsamps = tf.shape(mean)[0]
    NTbins = self.max_steps
    xDim = self.main_dim
    
    prenoise_NxTxd = tf.random_normal([Nsamps, NTbins, xDim], dtype=tf.float64)
    
    aux_fn = lambda _, seqs : blk_chol_inv(seqs[0], seqs[1], seqs[2],
                                           lower=False, transpose=True)
    noise = tf.scan(fn=aux_fn, elems=[inv_scale_d, inv_scale_od, prenoise_NxTxd],
                    initializer=tf.zeros_like(prenoise_NxTxd[0], dtype=tf.float64) )
    sample = tf.add(mean, noise, name='sample')
#     sample = tf.add(self.postX_NxTxd, noise, name='sample')
                
    return sample #, noisy_postX_ng
  
  def _build(self):
    """
    """
    mean, inv_scale, _ = self._get_loc_inv_scale()
    
    self._oslot_to_otensor[1] = mean
    self._oslot_to_otensor[2] = inv_scale[0]
    self._oslot_to_otensor[3] = inv_scale[1]
    
    self._oslot_to_otensor[0] = self.get_sample()
        
    self._is_built = True
  
  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    """
    InnerNode.__call__(self, inputs=inputs, islot_to_itensor=islot_to_itensor)    
    
  def entropy(self):
    """
    Compute the Entropy. 
    
    Args:
        Input (tf.Tensor):
        Ids (tf.Tensor):
        
    Returns:
        - Entropy: The entropy of the Recognition Model
    """
#     if Input is None and Ids is not None:
#       raise ValueError("Must provide an Input for these Ids")
#     X_NxTxd = self.X if Input is None else Input
#     if Ids is None: Ids = self.Ids

    mean = self.get_output(1)
    inv_scale_d = self.get_output(2)

    NTbins = self.max_steps
    xDim = self.main_dim
    Nsamps = tf.shape(mean)[0]
    inv_scale_d = tf.reshape(inv_scale_d, [Nsamps*NTbins, xDim, xDim])

#     TheChol_2xxNxTxdxd = ( self.TheChol_2xxNxTxdxd if Input is None else
#                            self._compute_TheChol(Input, Ids)[0] ) 
#          
#     with tf.variable_scope('entropy'):
#     self.thechol0 = tf.reshape(TheChol_2xxNxTxdxd[0], 
#                                [Nsamps*NTbins, xDim, xDim])
    log_det = -2.0*tf.reduce_sum(tf.log(tf.matrix_determinant(inv_scale_d)))
            
    Nsamps = tf.cast(Nsamps, tf.float64)        
    NTbins = tf.cast(NTbins, tf.float64)        
    xDim = tf.cast(xDim, tf.float64)
    
    # Yuanjun has xDim here so I put it but I don't think this is right.
    entropy = tf.add(0.5*Nsamps*NTbins*(1 + np.log(2*np.pi)), 0.5*log_det,
                     name='Entropy')
    
    return entropy
  