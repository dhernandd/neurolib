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
from neurolib.encoder.inner import InnerNode
from neurolib.utils.directives import NodeDirectives
from neurolib.utils.shapes import infer_shape, match_tensor_shape

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
  

class MergeNode(InnerNode):
  """
  A MergeNode creates a new InnerNode from a list of InnerNodes. 
  
  The defining feature of a MergeNode is that its state sizes and oshapes can be
  derived from its inputs and hence do not need to be specified by the user.
  """
  def __init__(self,
               builder,
               node_list=None,
               node_dict_tuples=None,
               is_sequence=False,
               name_prefix=None,
               **dirs):
    """
    Initialize the MergeNode
    """
    # here call to super before defining state sizes
    super(MergeNode, self).__init__(builder,
                                    is_sequence,
                                    name_prefix=name_prefix,
                                    **dirs)
    
    # directives object
    self.directives = NodeDirectives(self.directives)
    self.oslot_names = self.directives.output_names

    # Get state_sizes
    if node_list is None:
      if node_dict_tuples is None:
        raise ValueError("`node_dict` argument is mandatory if `node_list` is None")
      self.node_dict_tuples = node_dict_tuples
      self.node_list_tuples = self.make_list_inode_otuples_from_nodedict()
      self.node_list = self.make_inode_list_from_dtuples()
    else:
      self.node_list = self._complete_node_list(node_list)
      print("self.node_list", self.node_list)
      self.node_list_tuples = self._make_list_islot_otuples_from_nodelist()
    self.state_sizes = self._get_state_sizes()
    
      # Declare num_expected_inputs, outputs, etc.
    self.num_expected_inputs = len(self.node_list) 
    
    # shapes
    self.oshapes = self._get_all_oshapes()
    self.state_ranks = self.get_state_size_ranks()
    
    # Free i/o slots
    self._islot_to_itensor = [{} for _ in range(self.num_expected_inputs)]
    self.free_islots = list(range(self.num_expected_inputs))
    self.free_oslots = list(range(self.num_expected_outputs))
            
  def _complete_node_list(self, node_list):
    """
    Add implicit nodes to the `node_list` before assigning to the attribute of
    self.
    """
    return node_list
  
  def _make_list_islot_otuples_from_nodelist(self):
    """
    Make a list indexed by the islot, whose values are lists of tuples,
    identifying specific oslots of the parent node.
    """
    raise NotImplementedError
    
  def _get_state_sizes(self):
    """
    """
    raise NotImplementedError
    
  def _update_directives(self, **dirs):
    """
    Update directives
    """
    this_node_dirs = {}
    this_node_dirs.update(dirs)
    super(MergeNode, self)._update_directives(**this_node_dirs)


class MergeNormals(MergeNode):
  """
  Merge two or more normal distributions.
  """
  num_expected_outputs = 3
  
  def __init__(self,
               builder,
               node_list=None,
               node_dict_tuples=None,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize the MergeNormals Node
    """
    # set name
    name_prefix = name_prefix or 'MergeNormals'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)

    super(MergeNormals, self).__init__(builder,
                                       node_list=node_list,
                                       node_dict_tuples=node_dict_tuples,
                                       name_prefix=name_prefix,
                                       is_sequence=False,
                                       **dirs)
    
    # shapes
    self.xdim = self.oshapes['main'][-1]
    
    # dist
    self.dist = None
    
    # add links from mergers
    self._add_links_from_mergers()
    
  def _make_list_islot_otuples_from_nodelist(self):
    """
    Make the merge list.
    """
    if self.node_list is None:
      raise ValueError("`node_list` attribute undefined")
    return [[('loc', 'scale')] for _ in self.node_list]
    
  def _get_state_sizes(self):
    """
    Define state_sizes from node list
    """
    an_iname = self.node_list[0]
    an_inode = self.builder.nodes[an_iname]
    
    islot = 0
    parent_oslot_pair = 0
    oslot = 0
    an_ishape = an_inode.oshapes[self.node_list_tuples[islot][parent_oslot_pair][oslot]]
    
    return [[an_ishape[-1]]]
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    
    Add the directives for specific of this class and propagate up the class
    hierarchy
    """
    this_node_dirs = {'outputname_1' : 'loc',
                      'outputname_2' : 'cov'}
    this_node_dirs.update(dirs)
    super(MergeNormals, self)._update_directives(**this_node_dirs)

  def _get_all_oshapes(self):
    """
    Declare the shapes for every node output
    """
    an_iname = self.node_list[0]
    an_inode = self.builder.nodes[an_iname]
    an_ishape = an_inode.oshapes['loc']
    
    return {'main' : an_ishape,
            'loc' : an_ishape,
            'cov' : an_ishape + [an_ishape[-1]]}
        
  def _add_links_from_mergers(self):
    """
    Create the edges from the parents to this MergeNode
    """
    for i, node_name in enumerate(self.node_list):
      self.builder.addDirectedLink(node_name, self, islot=i)
    
  def __call__(self, *inputs):
    """
    Evaluate the node on a list of inputs.
    """
    if not inputs:
      raise ValueError("Inputs are mandatory for the DeterministicNNNode")
    
    l = len(inputs)
    if l % 2:
      raise ValueError("Odd number of inputs")
      
    islot_to_itensor = [{'loc' : inputs[i], 'scale' : inputs[i+1]} for i 
                        in range(l/2)]
    return self.build_outputs(islot_to_itensor)
  
  def build_outputs(self, islot_to_itensor=None):
    """
    Get MergeNormals outputs
    """
    if islot_to_itensor is not None:
      _input = islot_to_itensor
    else:
      _input = self._islot_to_itensor
    print("_input", _input)
    
    precs = []
    locs = []
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      for islot in range(self.num_expected_inputs):
        for loc_sc_tuple in self.node_list_tuples[islot]:
          this_loc = _input[islot][loc_sc_tuple[0]]
          this_loc = tf.expand_dims(this_loc, axis=2)
          
          sh = infer_shape(this_loc)
          this_sc = _input[islot][loc_sc_tuple[1]]
          this_sc = match_tensor_shape(sh, this_sc, 3)
          
          this_cov = tf.matmul(this_sc, this_sc, transpose_b=True)
          
          locs.append(this_loc)
          precs.append(tf.matrix_inverse(this_cov))
      
      total_cov = tf.matrix_inverse(sum(precs))
      precmean = sum([tf.matmul(prec, loc) for prec, loc 
                     in zip(precs, locs)])
      post_mean = tf.matmul(total_cov, precmean)
      post_mean = tf.squeeze(post_mean, axis=2)
    
      self.dist = MultivariateNormalFullCovariance(loc=post_mean,
                                                   covariance_matrix=total_cov)
      samp = self.dist.sample()

    return samp, post_mean, total_cov
      
  def _build(self):
    """
    Build the MergeNormals Node
    """
    samp, loc, cov = self.build_outputs()
    
    self.fill_oslot_with_tensor(0, samp)
    self.fill_oslot_with_tensor(1, loc)
    self.fill_oslot_with_tensor(2, cov)

    self._is_built = True 

  
class MergeSeqsNormalwNormalEv(MergeNode):
  """
  Merge a normal sequence with an LDS evolution sequence
  """
  num_expected_outputs = 4
  def __init__(self,
               builder,
               node_list=None,
               node_dict_tuples=None,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize the MergeSeqsNormalwNormalEv 
    """
    # set name
    name_prefix = name_prefix or 'MergeNormalSeqLDS'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    super(MergeSeqsNormalwNormalEv, self).__init__(builder,
                                               node_list=node_list,
                                               node_dict_tuples=node_dict_tuples,
                                               is_sequence=True,
                                               name_prefix=name_prefix,
                                               **dirs)
    # shapes
    self.xdim = self.oshapes['main'][-1]

    # dist
    self.dist = None
    
    # add links from mergers
    self._add_links_from_mergers()

  def _complete_node_list(self, node_list):
    """
    Add implicit nodes to the `node_list` before assigning to the attribute of
    self.
    
    TODO: This wont work with an RNN defined DS. Fix!
    """
#     lds_name = node_list[1]
#     lds = self.builder.nodes[lds_name]
#     node_list.append(lds.get_init_inodes()[0])
    return node_list

  def _make_list_islot_otuples_from_nodelist(self):
    """
    Make the merge list.
    """
    if self.node_list is None:
      raise ValueError("`node_list` attribute undefined")
    
    # [prior, input sequence, LDS]
    return [[('loc', 'prec')], [('prec', 'A')], [('scale',)]]
    
  def _get_state_sizes(self):
    """
    Define state_sizes from node list
    """
    inseq_name = self.node_list[0]
    inseq = self.builder.nodes[inseq_name]
    
    islot = 0
    parent_oslot_pair = 0
    oslot = 0
    itensor_oslot = self.node_list_tuples[islot][parent_oslot_pair][oslot]
    ishape = inseq.oshapes[itensor_oslot]
    
    return [[ishape[-1]]]
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'usett' : False,
                      'outputname_1' : 'loc',
                      'outputname_2' : 'scaled',
                      'outputname_3' : 'scaleoffd'}
    this_node_dirs.update(dirs)
    super(MergeSeqsNormalwNormalEv, self)._update_directives(**this_node_dirs)
  
  def _get_all_oshapes(self):
    """
    Declare the shapes for every node output
    """
    iseq_name = self.node_list[0]
    iseq = self.builder.nodes[iseq_name]
    iseq_mainshape = iseq.oshapes['main']
    
    return {'main' : iseq_mainshape,
            'loc' : iseq_mainshape,
            'scaled' : iseq_mainshape + [iseq_mainshape[-1]],
            'scaleoffd' : iseq_mainshape + [iseq_mainshape[-1]]}

  def _add_links_from_mergers(self):
    """
    Create the edges from the parents to this MergeNode
    """
    print("self.node_list", self.node_list)
    for i, node_name in enumerate(self.node_list):
      self.builder.addDirectedLink(node_name, self, islot=i)
    print("self.builder.adj_matrix", self.builder.adj_matrix)
    
  def __call__(self, *inputs):
    """
    Evaluate the node on a list of inputs.
    """
    raise NotImplementedError
  
  def build_outputs(self, islot_to_itensor=None):
    """
    Get MergeNormals outputs
    """
    if islot_to_itensor is not None:
      _input = islot_to_itensor
    else:
      _input = self._islot_to_itensor
    print("_input", _input)

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      NTbins = self.max_steps
      xDim = self.xdim

      # Get inputs from the inner sequence
      islot = 0
      inseq_loc_oslot = self.node_list_tuples[islot][0][0]
      inseq_scale_oslot = self.node_list_tuples[islot][0][1]
      
      mu_NxTxd = _input[islot][inseq_loc_oslot]
      mu_NxTxdx1 = tf.expand_dims(mu_NxTxd, axis=-1)
      lmbda_NxTxdxd = _input[islot][inseq_scale_oslot]
      lmbdaMu_NxTxd = tf.squeeze(tf.matmul(lmbda_NxTxdxd, mu_NxTxdx1), axis=-1)
      
      print("mu_NxTxd", mu_NxTxd)
      main_sh = infer_shape(mu_NxTxd)
      Nsamps = tf.shape(mu_NxTxd)[0]
  
      # Get inputs from the evolution sequence
      islot = 1
      evseq_invQ_oslot = self.node_list_tuples[islot][0][0]
      evseq_A_oslot = self.node_list_tuples[islot][0][1]
      
      # works for LDS and VIND specific
      invQ = _input[islot][evseq_invQ_oslot]
      A = _input[islot][evseq_A_oslot]
      invQ_NxTxdxd = match_tensor_shape(main_sh, invQ, 4)
      A_NxTxdxd = match_tensor_shape(main_sh, A, 4)
      
      # Get input from the evolution sequence prior
      islot = 2
      prior_invQ_oslot = self.node_list_tuples[islot][0][0]
      
      Q0scale_dxd = _input[islot][prior_invQ_oslot]
      Q0scale_Nxdxd = match_tensor_shape(main_sh, Q0scale_dxd, 3)
      print("Q0scale_Nxdxd", Q0scale_Nxdxd)
      Q0_Nxdxd = tf.matmul(Q0scale_Nxdxd, Q0scale_Nxdxd, transpose_b=True)
      invQ0_Nxdxd = tf.matrix_inverse(Q0_Nxdxd)
      invQ0_Nx1xdxd = tf.expand_dims(invQ0_Nxdxd, axis=1)

      # prepare tensors
      A_NTm1xdxd = tf.reshape(A_NxTxdxd[:,:-1,:,:], [Nsamps*(NTbins-1), xDim, xDim])
      invQ_NxTm2xdxd = invQ_NxTxdxd[:,:-2,:,:]
      invQ_NTm1xdxd = tf.reshape(invQ_NxTxdxd[:,:-1,:,:], [Nsamps*(NTbins-1), xDim, xDim])
      invQ0Q_NxTm1xdxd = tf.concat([invQ0_Nx1xdxd, invQ_NxTm2xdxd], axis=1)
      invQ0Q_NTm1xdxd = tf.reshape(invQ0Q_NxTm1xdxd, [Nsamps*(NTbins-1), xDim, xDim])

      # compute the off-diagonal blocks of full precision K:
      #     K(z)_{i,i+1} = -A(z)^T*Q^-1,     for i in {1,..., T-2}
      usett = self.directives.usett # use the transpose trick?
      AinvQ_NTm1xdxd = -tf.matmul(A_NTm1xdxd, invQ_NTm1xdxd,  #pylint: disable=invalid-unary-operand-type
                                   transpose_a=usett)  
      # The diagonal blocks of K up to T-1:
      #     K(z)_ii = A(z)^T*Qq^{-1}*A(z) + Qt^{-1},     for i in {1,...,T-1 }
      AinvQA_NTm1xdxd = (invQ0Q_NTm1xdxd - 
                         tf.matmul(AinvQ_NTm1xdxd, A_NTm1xdxd, transpose_b=not usett))
                         
      AinvQA_NxTm1xdxd = tf.reshape(AinvQA_NTm1xdxd,
                                     [Nsamps, NTbins-1, xDim, xDim]) 

      # Tile in the last block K_TT. 
      # This one does not depend on A. There is no latent evolution beyond T.
      invQ_Nx1xdxd = invQ_NxTm2xdxd = invQ_NxTxdxd[:,:1,:,:]
      AinvQAinvQ_NxTxdxd = tf.concat([AinvQA_NxTm1xdxd, invQ_Nx1xdxd], axis=1)
      
      # add in the piece coming from the observations
      AA_NxTxdxd = lmbda_NxTxdxd + AinvQAinvQ_NxTxdxd
      BB_NxTm1xdxd = tf.reshape(AinvQ_NTm1xdxd, [Nsamps, NTbins-1, xDim, xDim])        
      
      # compute the Cholesky decomposition for the total precision matrix
      aux_fn1 = lambda _, seqs : blk_tridiag_chol(seqs[0], seqs[1])
      invscale_2xxNxTxdxd = tf.scan(fn=aux_fn1, 
                                    elems=[AA_NxTxdxd, BB_NxTm1xdxd],
                                    initializer=[tf.zeros_like(AA_NxTxdxd[0]), 
                                                tf.zeros_like(BB_NxTm1xdxd[0])])
      
      
      checks = [A_NxTxdxd, AA_NxTxdxd, BB_NxTm1xdxd]
      
      def postX_from_chol(tc1, tc2, lm):
          """
          postX = (Lambda1 + S)^{-1}.(Lambda1_ij.*Mu_j + X^T_k.*S_kj;i.*X_j)
          """
          return blk_chol_inv(tc1, tc2, blk_chol_inv(tc1, tc2, lm), 
                              lower=False, transpose=True)
      aux_fn2 = lambda _, seqs : postX_from_chol(seqs[0], seqs[1], seqs[2])
      postX = tf.scan(fn=aux_fn2, 
                      elems=[invscale_2xxNxTxdxd[0], invscale_2xxNxTxdxd[1], lmbdaMu_NxTxd],
                      initializer=tf.zeros_like(lmbdaMu_NxTxd[0], dtype=tf.float64) )      

      samp = self.get_sample(postX, invscale_2xxNxTxdxd) 
    
    return samp, postX, invscale_2xxNxTxdxd, checks                                    
    
  def get_sample(self, loc, trid_invscale):
    """
    Sample from the posterior
    """
    xDim = self.xdim
    NTbins = self.max_steps
    Nsamps = tf.shape(loc)[0]
    
    inv_scale_d = trid_invscale[0]
    inv_scale_od = trid_invscale[1]
    
    prenoise_NxTxd = tf.random_normal([Nsamps, NTbins, xDim], dtype=tf.float64)
    
    aux_fn = lambda _, seqs : blk_chol_inv(seqs[0], seqs[1], seqs[2],
                                           lower=False, transpose=True)
    noise = tf.scan(fn=aux_fn, elems=[inv_scale_d, inv_scale_od, prenoise_NxTxd],
                    initializer=tf.zeros_like(prenoise_NxTxd[0],
                                              dtype=tf.float64) )
    sample = tf.add(loc, noise, name='sample')
                
    return sample

  def _build(self):
    """
    Build the MergeNormalSeqLDS
    """
    samp, loc, invscale, _ = self.build_outputs()
    
    self.fill_oslot_with_tensor(0, samp)
    self.fill_oslot_with_tensor(1, loc)
    self.fill_oslot_with_tensor(2, invscale[0])
    self.fill_oslot_with_tensor(3, invscale[1])

    self._is_built = True 
  
  def entropy(self):
    """
    Compute the Entropy. 
    
    Args:
        Input (tf.Tensor):
        Ids (tf.Tensor):
        
    Returns:
        - Entropy: The entropy of the Recognition Model
    """
    if not self._is_built:
      raise ValueError("Cannot access entropy attribute for unbuilt node")
    invscale_d = self.get_output_tensor('scaled')

    NTbins = self.max_steps
    xDim = self.xdim
    Nsamps = tf.shape(invscale_d)[0]

    invscale_d = tf.reshape(invscale_d, [Nsamps*NTbins, xDim, xDim])

    # compute entropy
    Nsamps = tf.cast(Nsamps, tf.float64)        
    NTbins = tf.cast(NTbins, tf.float64)        
    log_det = -2.0*tf.reduce_sum(tf.log(tf.matrix_determinant(invscale_d)))
    entropy = tf.add(0.5*Nsamps*NTbins*(1 + np.log(2*np.pi)), 0.5*log_det,
                     name='Entropy') # xdim?
    self.builder.add_to_output_names(self.name+':Entropy', entropy)
    
    return entropy
  
  
  