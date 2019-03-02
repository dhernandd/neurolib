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

  
class MergeSeqsNormalwNormalEv(InnerNode):
  """
  Merge a normal sequence with an LDS/LLDS evolution sequence
  """
  num_expected_outputs = 4
  
  def __init__(self,
               builder,
               seq_inputs,
               ds_inputs,
               prior_inputs,
#                node_list=None,
#                node_dict_tuples=None,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize the MergeSeqsNormalwNormalEv 
    """
    # set name
    name_prefix = name_prefix or 'MergeNormalSeqLDS'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    # for merge nodes call to super before defining state sizes
    super(MergeSeqsNormalwNormalEv, self).__init__(builder,
                                                   is_sequence=True,
                                                   name_prefix=name_prefix,
                                                   **dirs)
    
    # directives object
    self.directives = NodeDirectives(self.directives)
    self.oslot_names = self.directives.output_names

    # inputs
    self.seq_inputs = seq_inputs
    self.ds_inputs = ds_inputs
    print("ds_inputs", ds_inputs)
    assert len(self.ds_inputs) == 1, "Only one merger must represent a dynamical system"
    self.prior_inputs = prior_inputs
    self.node_list = self.seq_inputs + self.ds_inputs + self.prior_inputs
    self.num_expected_inputs = len(seq_inputs) + len(prior_inputs) + len(ds_inputs)    

    # Get state_sizes
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
    
    # shapes
    self.xdim = self.oshapes['main'][-1]

    # dist
    self.dist = None
    
  def _make_list_islot_otuples_from_nodelist(self):
    """
    Make the merge list.
    
    TODO: What if the nodes are normal but variance is not specified through
    prec? Need to account for this
    """
    if self.node_list is None:
      raise ValueError("`node_list` attribute undefined")
    
    # [input sequence, LDS, prior]
    return [[('loc', 'prec')], [('prec', 'A')], [('scale',)]]
    
  def _get_state_sizes(self):
    """
    Define state_sizes from inputs
    """
    ds = self.builder.nodes[self.ds_inputs[0]]
    return [[ds.xdim]]
  
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'usett' : False,
                      'outputname_1' : 'loc',
                      'outputname_2' : 'invscaled',
                      'outputname_3' : 'invscaleoffd'}
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
            'invscaled' : iseq_mainshape + [iseq_mainshape[-1]],
            'invscaleoffd' : iseq_mainshape + [iseq_mainshape[-1]]}

  def __call__(self, *inputs):
    """
    Evaluate the node on a list of inputs.
    """
    raise NotImplementedError
  
  def _build(self):
    """
    Build the MergeNormalSeqLDS
    """
#     samp, loc, invscaled, invscaleoffd = self.build_outputs()
    self.build_outputs()
    
#     self.fill_oslot_with_tensor(0, samp)
#     self.fill_oslot_with_tensor(1, loc)
#     self.fill_oslot_with_tensor(2, invscaled)
#     self.fill_oslot_with_tensor(3, invscaleoffd)

    self._is_built = True 
  
  def build_outputs(self, **inputs):
    """
    Get the outputs of the LDSNode
    """
    print("Building all outputs, ", self.name)
#     invscale, _ = self.build_output('invscale', **inputs)
#     loc, _ = self.build_output('loc', invscale=invscale, **inputs)
#     samp, _ = self.build_output('main', invscale=invscale, loc=loc)
    self.build_output('invscale', **inputs)
    self.build_output('loc', **inputs)
    self.build_output('main', **inputs)
  
  def build_output(self, oname, **inputs):
    """
    Build a single output
    """
    if oname == 'invscale':
      return self.build_invscale(**inputs)
    elif oname == 'loc':
      return self.build_loc(**inputs)
    elif oname == 'main':
      return self.build_main(**inputs)
    else:
      raise ValueError("`oname` {} is not an output name for "
                       "this node".format(oname))
    
  def prepare_inputs(self, **inputs):
    """
    Prepare inputs for building
    """
    true_inputs = {'imain_loc'  : self.get_input_tensor(0, 'loc'),
                   'imain_prec' : self.get_input_tensor(0, 'prec'),
                   'ids_prec' : self.get_input_tensor(1, 'prec'),
                   'ids_A' : self.get_input_tensor(1, 'A'),
                   'iprior_scale' : self.get_input_tensor(2, 'scale')}
    
    if inputs:
      print("\t\tUpdating defaults,", self.name, "with", list(inputs.keys()))
      true_inputs.update(inputs)
    return true_inputs

  def build_invscale(self, **inputs):
    """
    """
    return self.build_invscale_secs(**inputs)[0]
  
  def build_invscale_secs(self, **inputs):
    """
    Build the invscale
    """
    print("\tBuilding invscale,", self.name)
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      NTbins = self.max_steps
      xDim = self.xdim
      
      inputs = self.prepare_inputs(**inputs)
      mu_NxTxd = inputs['imain_loc']
      lmbda_NxTxdxd = inputs['imain_prec']
      invQ = inputs['ids_prec']
      A = inputs['ids_A']
      Q0scale_dxd = inputs['iprior_scale']

      main_sh = infer_shape(mu_NxTxd)
      Nsamps = tf.shape(mu_NxTxd)[0]
  
      # Get node_inputs from the evolution sequence
      invQ_NxTxdxd = match_tensor_shape(main_sh, invQ, 4)
      A_NxTxdxd = match_tensor_shape(main_sh, A, 4)
      
      Q0scale_Nxdxd = match_tensor_shape(main_sh, Q0scale_dxd, 3)
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
    
    if not self._is_built:
      self.fill_oslot_with_tensor(2, invscale_2xxNxTxdxd[0])
      self.fill_oslot_with_tensor(3, invscale_2xxNxTxdxd[1])
    
    return invscale_2xxNxTxdxd, ()
   
  def build_loc(self, **inputs):
    """
    """
    return self.build_loc_secs(**inputs)[0]
  
  def build_loc_secs(self, **inputs):
    """
    Build loc
    """
    print("\tBuilding loc,", self.name)
    
    if not inputs:
      inputs = self.prepare_inputs(**inputs)
      invscale_2xxNxTxdxd = [self.get_output_tensor('invscaled'),
                             self.get_output_tensor('invscaleoffd')]
    else:
      inputs = self.prepare_inputs(**inputs)
      invscale_2xxNxTxdxd = self.build_invscale(**inputs)
    mu_NxTxd = inputs['imain_loc']
    lmbda_NxTxdxd = inputs['imain_prec']
    mu_NxTxdx1 = tf.expand_dims(mu_NxTxd, axis=-1)
    
    lmbdaMu_NxTxd = tf.squeeze(tf.matmul(lmbda_NxTxdxd, mu_NxTxdx1), axis=-1)
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      def postX_from_chol(tc1, tc2, lm):
          """
          loc = (Lambda1 + S)^{-1}.(Lambda1_ij.*Mu_j + X^T_k.*S_kj;i.*X_j)
          """
          return blk_chol_inv(tc1, tc2, blk_chol_inv(tc1, tc2, lm), 
                              lower=False, transpose=True)
      aux_fn2 = lambda _, seqs : postX_from_chol(seqs[0], seqs[1], seqs[2])
      loc = tf.scan(fn=aux_fn2, 
                      elems=[invscale_2xxNxTxdxd[0], invscale_2xxNxTxdxd[1], lmbdaMu_NxTxd],
                      initializer=tf.zeros_like(lmbdaMu_NxTxd[0], dtype=tf.float64) )
    
    if not self._is_built:
      self.fill_oslot_with_tensor(1, loc)
    
    return loc, (invscale_2xxNxTxdxd,)
  
  def build_main(self, **inputs):
    """
    Build main output
    """
    return self.build_main_secs(**inputs)[0]
  
  def build_main_secs(self, **inputs):
    """
    """
    print("\tBuilding main,", self.name)
    
    if not inputs:
      inputs = self.prepare_inputs(**inputs)
      invscaled = self.get_output_tensor('invscaled')
      invscaleoffd = self.get_output_tensor('invscaleoffd')
      loc = self.get_output_tensor('loc')
    else:
      inputs = self.prepare_inputs(**inputs)
      loc, secs = self.build_loc_secs(**inputs)
      invscale = secs[0]
      invscaled, invscaleoffd = invscale[0], invscale[1]
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      xDim = self.xdim
      NTbins = self.max_steps
      Nsamps = tf.shape(loc)[0]
      
#       invscaled = invscale[0]
#       invscaleoffd = invscale[1]
      
      prenoise_NxTxd = tf.random_normal([Nsamps, NTbins, xDim], dtype=tf.float64)
      
      aux_fn = lambda _, seqs : blk_chol_inv(seqs[0], seqs[1], seqs[2],
                                             lower=False, transpose=True)
      noise = tf.scan(fn=aux_fn, elems=[invscaled, invscaleoffd, prenoise_NxTxd],
                      initializer=tf.zeros_like(prenoise_NxTxd[0],
                                                dtype=tf.float64) )
      samp = tf.add(loc, noise, name='sample')

    if not self._is_built:
      self.fill_oslot_with_tensor(0, samp)

    return samp, (loc, [invscaled, invscaleoffd])
  
  def build_entropy(self, name=None, **inputs):
    """
    """
    return self.build_entropy_secs(name=name, **inputs)[0]
  
  def build_entropy_secs(self, name=None, **inputs):
    """
    Build the Entropy. 
    
    Args:
        Input (tf.Tensor):
        Ids (tf.Tensor):
        
    Returns:
        - Entropy: The entropy of the Recognition Model
    """
    print("Building entropy, ", self.name)
    if not self._is_built:
      raise ValueError("Cannot access entropy attribute for unbuilt node")
    
    if not inputs:
      invscaled = self.get_output_tensor('invscaled')
      invscaleoffd = self.get_output_tensor('invscaleoffd')
      invscale = [invscaled, invscaleoffd]
    else:
      inputs = self.prepare_inputs(**inputs)
      invscale = self.build_invscale(**inputs)
      invscaled = invscale[0]
#       if 'invscaled' in inputs:
#         invscaled = inputs['invscaled']
#       else:
#         try:
#           # TODO: Make sure the inputs are providing something new, here and
#           # everywhere else. VERY NASTY BUG!
#           inputs = self.prepare_inputs(**inputs)
#           invscale, _ = self.build_invscale(**inputs)
#           invscaled = invscale[0]
#         except:
#           raise ValueError("Could not define `self.entropy`")

    NTbins = self.max_steps
    xDim = self.xdim
    Nsamps = tf.shape(invscaled)[0]

    invscaled = tf.reshape(invscaled, [Nsamps*NTbins, xDim, xDim])
    
    # compute entropy
    Nsamps = tf.cast(Nsamps, tf.float64)        
    NTbins = tf.cast(NTbins, tf.float64)        
    log_det = -2.0*tf.reduce_sum(tf.log(tf.matrix_determinant(invscaled)))
    entropy = tf.add(0.5*Nsamps*NTbins*(1 + np.log(2*np.pi)), 0.5*log_det,
                     name='entropy') # xdim?

    if name is None:
      name = self.name + ':entropy'
    else:
      name = self.name + ':' + name
    
    if name in self.builder.otensor_names:
      raise ValueError("name {} has already been defined, pass a different"
                       "argument `name`".format(name))
    self.builder.add_to_output_names(name, entropy)
    
    return entropy, (invscale,)
  
  
  