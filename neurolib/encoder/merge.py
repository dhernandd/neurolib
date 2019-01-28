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
  

class MergeNode(InnerNode):
  """
  A MergeNode defines a new InnerNode from a list of InnerNodes
  """
  def __init__(self,
               builder,
               state_size,
               node_list=None,
               node_dict=None,
               parents_to_oslot_tuples=None,
               name_prefix=None,
               is_sequence=False,
               **dirs):
    """
    Initialize the MergeNode
    """
    # Get state_size (TODO: can be derived from mergers)
    self.state_sizes = self.state_sizes_to_list(state_size)
    self.xdim = self.state_sizes[0][0]
    
    super(MergeNode, self).__init__(builder,
                                    is_sequence,
                                    name_prefix=name_prefix,
                                    **dirs)
    
    pots = self._parents_to_otuples(node_list=node_list,
                                    node_dict=node_dict,
                                    parents_to_oslot_tuples=parents_to_oslot_tuples)
    self.parents_to_oslot_merge_tuples = pots
    self.num_mergers = sum([len(pots[key]) for key in pots])
        
  def _parents_to_otuples(self,
                          node_list=None,
                          node_dict=None,
                          parents_to_oslot_tuples=None):
    """
    Define the `parent_to_oslot_merge_tuples` dict for the MergeNodes
    """
    raise NotImplementedError("")
    
  def _update_directives(self, **dirs):
    """
    Update directives
    """
    this_node_dirs = {}
    this_node_dirs.update(dirs)
    print("dirs", dirs)
    super(MergeNode, self)._update_directives(**this_node_dirs)
    print("self.directives", self.directives)

  @staticmethod
  def get_node_oslots_from_output_names(node, *outputs):
    """
    """
    print("node, *outputs", node, *outputs)
    print("node.directives", node.directives.items())
    found_output = [False for _ in range(len(outputs))]
    found_oslots = []
    for i, output in enumerate(outputs):
      for oslot in range(node.num_expected_outputs):
        key = 'output_' + str(oslot) + '_name'
        if node.directives[key] == output:
          if found_output[i] == True:
            raise Exception("The parent node {} has more than one oslot named "
                            " {}. Use the `parent_to_oslot_tuples "
                            "attribute argument to initialize the "
                            "MergeNode".format(node.name, output))
          found_oslots.append(oslot)
          found_output[i] = True
      if found_output[i] == False:
        raise Exception("output {} not found in parent node"
                        "".format(output))
    return found_oslots


class MergeNormals(MergeNode):
  """
  Merge two or more normal distributions.
  """
  def __init__(self,
               builder,
               state_size,
               node_list=None,
               node_dict=None,
               parents_to_oslot_tuples=None,
               name=None,
               name_prefix='MergeNormals',
               **dirs):
    """
    Initialize the MergeNormals Node
    """
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    super(MergeNormals, self).__init__(builder,
                                       state_size,
                                       node_list=node_list,
                                       node_dict=node_dict,
                                       parents_to_oslot_tuples=parents_to_oslot_tuples,
                                       name_prefix=name_prefix,
                                       is_sequence=False,
                                       **dirs)
    # Declare num_expected_inputs, outputs, etc.
    self.num_expected_inputs = 2*self.num_mergers
    self.num_expected_outputs = 3
    self.num_declared_mergers = 0
    self.free_islots = list(range(self.num_expected_inputs))
    self.free_oslots = list(range(self.num_expected_outputs))

    self.dist = None
    self.cov_means = {}
    self._add_links_from_mergers()
    
  def _parents_to_otuples(self,
                          node_list=None,
                          node_dict=None,
                          parents_to_oslot_tuples=None):
    """
    Turn any possible kind of input for the MergeNode into a
    parent_to_oslot_tuples 
    
    Only one out of parents_list, node_dict or parents_to_oslot_tuples can be
    provided.
    """
    if parents_to_oslot_tuples is not None:
      if node_list is not None:
        raise ValueError("The argument `parents_list` is not None, yet "
                         "`parents_to_oslot_tuples` was provided") 
      if node_dict is not None:
        raise ValueError("The argument `node_dict` is not None, yet "
                         "`parents_to_oslot_tuples` was provided") 
      
      self.parents_list = [self.builder.nodes[node] for node in parents_to_oslot_tuples]
      return parents_to_oslot_tuples
    elif node_dict is not None or node_list is not None:
      if node_dict is not None:
        if node_list is not None:
          raise ValueError("`parents_list` is not None, but "
                           "`node_dict` was also provided") 
        self.parents_list = [self.builder.nodes[node] for node in node_dict]
      else:
        self.parents_list = [self.builder.nodes[node] for node in node_list]
      
      parents_to_oslot_tuples = {}
      for node in self.parents_list:
        if node.name not in parents_to_oslot_tuples:
          parents_to_oslot_tuples[node.name] = []
        try:
          oslots = self.get_node_oslots_from_output_names(node, 'loc', 'scale')
          parents_to_oslot_tuples[node.name].append(tuple(oslots))
        except:
          raise Exception("Could not define `parents_to_oslot_tuples`"
                          "from `self.parents_list`")
      
      return parents_to_oslot_tuples

    else:
      raise ValueError("All of `parents_list`, `node_dict` and "
                       "`parents_to_oslot_tuples` are None. "
                       "Exactly one of them must be provided")

  def _update_directives(self, **dirs):
    """
    Update the node directives
    
    Add the directives for specific of this class and propagate up the class
    hierarchy
    """
    this_node_dirs = {'output_1_name' : 'loc',
                      'output_2_name' : 'scale'}
    this_node_dirs.update(dirs)
    super(MergeNormals, self)._update_directives(**this_node_dirs)

  def _add_links_from_mergers(self):
    """
    Create the edges from the parents to this MergeNode
    """
    cur_islot = 0
    for node in self.parents_to_oslot_merge_tuples:
      print("self.parents_to_oslot_merge_tuples", self.parents_to_oslot_merge_tuples)
      for merge_tuple in self.parents_to_oslot_merge_tuples[node]:
        loc_oslot, scale_oslot = merge_tuple[0], merge_tuple[1]
        self.builder.addDirectedLink(node,
                                     self,
                                     islot=cur_islot,
                                     oslot=loc_oslot)
        cur_islot += 1
        self.builder.addDirectedLink(node,
                                     self,
                                     islot=cur_islot,
                                     oslot=scale_oslot)
        cur_islot += 1
        self.num_declared_mergers += 1

  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    Call the MergeNode
    """
    InnerNode.__call__(self, inputs=inputs, islot_to_itensor=islot_to_itensor)
      
  def update_when_linked_as_node1(self):
    """
    """
    return
  
  def update_when_linked_as_node2(self):
    """
    """
    return
        
  def _get_mean_covariance(self):
    """
    Get the total covariance from the input scales
    """
    xdim = self.xdim
    for parent_name in self.parents_to_oslot_merge_tuples:
      parent = self.builder.nodes[parent_name]
      for merge_tuple in self.parents_to_oslot_merge_tuples[parent_name]:
        dist_id = parent.name + str(merge_tuple[0])
        if dist_id not in self.cov_means:
          self.cov_means[dist_id] = [tf.zeros([1, xdim, xdim], dtype=tf.float64),
                                     tf.zeros([1, xdim, 1], dtype=tf.float64)]
        else:
          raise KeyError("The key {} already appears in `self.cov_means"
                         "".format(dist_id))
        loc_oslot, sc_oslot = merge_tuple 
        
        loc = parent.get_output(loc_oslot)
        loc = tf.expand_dims(loc, axis=2)
        self.cov_means[dist_id][0] = loc
        
        sc = parent.get_output(sc_oslot)
        cov = tf.matmul(sc, sc, transpose_b=True)
        self.cov_means[dist_id][1] = cov
    
    print("self.cov_means", self.cov_means)
    total_cov = sum([cov for cov in list(zip(*self.cov_means.values()))[1]])
    post_mean = sum([tf.matmul(tf.matrix_inverse(cov_mean[1]), cov_mean[0]) 
                     for cov_mean in self.cov_means.values()])
    post_mean = tf.matmul(tf.matrix_inverse(total_cov), post_mean)
    post_mean = tf.squeeze(post_mean, axis=2)
        
    return post_mean, total_cov
      
  def _build(self):
    """
    Builds the merged Normal distribution
    """
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      mean, cov = self._get_mean_covariance()
      self.dist = MultivariateNormalFullCovariance(loc=mean,
                                                   covariance_matrix=cov)
    
    o0_name = self.directives['output_0_name']
    samp = self.dist.sample(name='Out' + str(self.label) + '_0')
    self._oslot_to_otensor[0] = tf.identity(samp, name=o0_name)

    o1_name = self.directives['output_1_name']
    self._oslot_to_otensor[1] = tf.identity(mean, name=o1_name)

    o2_name = self.directives['output_2_name']
    self._oslot_to_otensor[2] = tf.identity(self.dist.scale.to_dense(), name=o2_name)
    print("self._oslot_to_otensor[2]", self._oslot_to_otensor[2])
    
    self._is_built = True


class MergeSeqsNormalLDSEv(MergeNode):
  """
  Merge a normal sequence with an LDS evolution sequence
  """
  def __init__(self,
               builder,
               state_size,
               node_list=None,
               node_dict=None,
               parents_to_oslot_tuples=None,
               name=None,
               name_prefix='MergeSeqs',
               **dirs):
    """
    Initialize the MergeSeqsNormalLDSEv 
    """
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    super(MergeSeqsNormalLDSEv, self).__init__(builder,
                                               state_size,
                                               node_list=node_list,
                                               node_dict=node_dict,
                                               parents_to_oslot_tuples=parents_to_oslot_tuples,
                                               is_sequence=True,
                                               name_prefix=name_prefix,
                                               **dirs)
    self._add_additional_inputs()

    self.num_expected_inputs = 5
    self.num_expected_outputs = 4

    self.main_oshape = self.get_state_full_shapes()
    self.D = self.get_state_size_ranks()
    self._oslot_to_shape[0] = self.main_oshape
    self.main_dim = self.state_sizes[0][0]
          
    self.free_islots = list(range(self.num_expected_inputs))
    self.free_oslots = list(range(self.num_expected_outputs))
    
    self.dist = None

    self._add_links_from_mergers()

    self._add_secondary_links()    
    
  def _parents_to_otuples(self,
                          node_list=None,
                          node_dict=None,
                          parents_to_oslot_tuples=None):
    """
    Turn any possible kind of input to the MergeNode into a
    parent_to_oslot_tuples 
    
    Only one out of parents_list, node_dict or parents_to_oslot_tuples can be
    provided.
    """
    if parents_to_oslot_tuples is not None:
      if node_list is not None:
        raise ValueError("The argument `node_list` is not None, yet "
                         "`parents_to_oslot_tuples` was provided") 
      if node_dict is not None:
        raise ValueError("The argument `node_dict` is not None, yet "
                         "`parents_to_oslot_tuples` was provided") 
      
      in_seq_name = parents_to_oslot_tuples['in_seq'][0]
      ev_seq_name = parents_to_oslot_tuples['ev_seq'][0]
      self.parents_list = [self.builder.nodes[in_seq_name],
                           self.builder.nodes[ev_seq_name]]
      return parents_to_oslot_tuples
    elif node_dict is not None or node_list is not None:
      if node_dict is not None:
        if node_list is not None:
          raise ValueError("`parents_list` is not None, but "
                           "`node_dict` was also provided") 
        in_seq_name = node_dict['in_seq']
        ev_seq_name = node_dict['ev_seq']
        self.parents_list = [self.builder.nodes[in_seq_name],
                             self.builder.nodes[ev_seq_name]]
      else:
        self.parents_list = [self.builder.nodes[node] for node in node_list]
      
      parents_to_oslot_tuples = {}
      in_seq, ev_seq = self.parents_list
      try:
        iseq_oslots = self.get_node_oslots_from_output_names(in_seq, 'loc', 'precision')
        eseq_oslots = self.get_node_oslots_from_output_names(ev_seq, 'invQ', 'A')
        
        iseq_oslots.insert(0, in_seq.name)
        eseq_oslots.insert(0, ev_seq.name)
        
#         parents_to_oslot_tuples['in_seq'].append(tuple(iseq_oslots))
#         parents_to_oslot_tuples['ev_seq'].append(tuple(eseq_oslots))
        parents_to_oslot_tuples['in_seq'] = tuple(iseq_oslots)
        parents_to_oslot_tuples['ev_seq'] = tuple(eseq_oslots)
      except:
        raise Exception("Could not define `parents_to_oslot_tuples`"
                        "from `self.parents_list`")
      
      return parents_to_oslot_tuples

    else:
      raise ValueError("All of `parents_list`, `node_dict` and "
                       "`parents_to_oslot_tuples` are None. "
                       "Exactly one of them must be provided")

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'output_1_name' : 'loc',
                      'output_2_name' : 'scale'}
    this_node_dirs.update(dirs)
    super(MergeSeqsNormalLDSEv, self)._update_directives(**this_node_dirs)
    
  def _add_additional_inputs(self):
    """
    """
    ev_seq = self.parents_list[1]
    prior = ev_seq.get_init_inodes()[0]
#     self.parents_to_oslot_merge_tuples['ev_seq_prior'] = []
    try:
      print("prior", prior.name, prior)
      prior_oslot = self.get_node_oslots_from_output_names(prior, 'scale')
      print("prior_oslot", prior_oslot)
      prior_oslot.insert(0, prior.name)
#       self.parents_to_oslot_merge_tuples['ev_seq_prior'].append(tuple(prior_oslot))
      self.parents_to_oslot_merge_tuples['ev_seq_prior'] = tuple(prior_oslot)
    except:
      raise KeyError("Could not define the entry `eq_seq_prior` in "
                       "`self.parents_to_oslot_merge_tuples`")

  def _add_secondary_links(self):
    """
    """
    # Loc oslot
    self._oslot_to_shape[1] = self.main_oshape
    o1 = self.builder.addOutput(name_prefix='Out_'+self.directives['output_1_name'])
    self.builder.addDirectedLink(self, o1, oslot=1)
    
  def _add_links_from_mergers(self):
    """
    Create the edges from the parents to this MergeNode
    """
    prnts_to_otpls = self.parents_to_oslot_merge_tuples
    islot = 0
    self.islot_collections = defaultdict(list)

    # Declare the edges from the input sequence
    iseq_name, loc_oslot, prec_oslot = prnts_to_otpls['in_seq']
    for oslot in [loc_oslot, prec_oslot]:
      self.builder.addDirectedLink(iseq_name,
                                   self,
                                   islot=islot,
                                   oslot=oslot)
      self.islot_collections['in_seq'].append(islot)
      islot += 1

    # Declare the edges from the evolution sequence
    eseq_name, invQ_oslot, A_oslot = prnts_to_otpls['ev_seq']
    for oslot in [invQ_oslot, A_oslot]:
#       print("merge; oslot, name", oslot, name)
      self.builder.addDirectedLink(eseq_name,
                                   self,
                                   islot=islot,
                                   oslot=oslot)
      self.islot_collections['ev_seq'].append(islot)
      islot += 1
  
    # Declare the edge from the evolution sequence prior
    prior_name, sc_oslot = prnts_to_otpls['ev_seq_prior']
    self.builder.addDirectedLink(prior_name,
                                 self,
                                 islot=islot,
                                 oslot=sc_oslot)
    self.islot_collections['ev_seq_prior'].append(islot)
    
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
      # Get islots of the input sequence
      in_seq_islots = self.islot_collections['in_seq']
      mu_NxTxd = self.get_input(in_seq_islots[0])
      mu_NxTxdx1 = tf.expand_dims(mu_NxTxd, axis=-1)
      lmbda_NxTxdxd = self.get_input(in_seq_islots[1])
      lmbdaMu_NxTxd = tf.squeeze(tf.matmul(lmbda_NxTxdxd, mu_NxTxdx1), axis=-1)
      
      Nsamps = tf.shape(mu_NxTxd)[0]
      NTbins = self.max_steps
      xDim = self.main_dim

      # Get islots of the evolution sequence
      ev_seq_islots = self.islot_collections['ev_seq']
      invQ_NxTxdxd = self.get_input(ev_seq_islots[0])
      A_NxTxdxd = self.get_input(ev_seq_islots[1])

      # Get islots of the evolution sequence prior
      ev_seqp_islots = self.islot_collections['ev_seq_prior']
      Q0scale_Nxdxd = self.get_input(ev_seqp_islots[0])
      Q0_Nxdxd = tf.matmul(Q0scale_Nxdxd, Q0scale_Nxdxd, transpose_b=True)
      invQ0_Nxdxd = tf.matrix_inverse(Q0_Nxdxd)
      invQ0_Nx1xdxd = tf.expand_dims(invQ0_Nxdxd, axis=1)

#       for islot in self.islot_collections[input_series_name]:
#         if 'precision' in self.islot_to_name[islot].split('_'):
#           lmbda_NxTxdxd = self.get_input(islot)
#         elif 'loc' in self.islot_to_name[islot].split('_'):
#           mu_NxTxd = self.get_input(islot)
#           mu_NxTxdx1 = tf.expand_dims(mu_NxTxd, axis=-1)
#       lmbdaMu_NxTxd = tf.squeeze(tf.matmul(lmbda_NxTxdxd, mu_NxTxdx1), axis=-1)
#       print('\n GOOD \n')
#       if InputY: _, Lambda_NxTxdxd, self.LambdaMu_NxTxd = self.get_Mu_Lambda(InputY)
#       else: Lambda_NxTxdxd = self.Lambda_NxTxdxd
      
#       if InputX is None and Ids is not None:
#         raise ValueError("Must provide an Input for these Ids")
#       X_NxTxd = self.X if InputX is None else InputX
#       if Ids is None: Ids = self.Ids
      
#       Nsamps = tf.shape(mu_NxTxd)[0]
#       NTbins = self.max_steps
#       xDim = self.main_dim
      
      # Simone Biles level tensorflow gymnastics in the next 150 lines or so
#       ev_seq_name = self.ev_seq.name
#       for islot in self.islot_collections[ev_seq_name]:
#         if 'A' in self.islot_to_name[islot].split('_'):
#           A_NxTxdxd = self.get_input(islot)
#         if 'invQ' in self.islot_to_name[islot].split('_'):
#           invQ_NxTxdxd = self.get_input(islot)
#       for islot in self.islot_collections[self.prior.name]:
#         if 'scale' in self.islot_to_name[islot].split('_'):
#           Q0scale_Nxdxd = self.get_input(islot)
#           Q0_Nxdxd = tf.matmul(Q0scale_Nxdxd, Q0scale_Nxdxd, transpose_b=True)
#           invQ0_Nxdxd = tf.matrix_inverse(Q0_Nxdxd)
#           invQ0_Nx1xdxd = tf.expand_dims(invQ0_Nxdxd, axis=1)
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
  