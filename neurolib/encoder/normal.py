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
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import fully_connected  #pylint: disable=no-name-in-module

from neurolib.encoder.inner import InnerNode
from neurolib.encoder import MultivariateNormalTriL, MultivariateNormalFullCovariance  # @UnresolvedImport
from neurolib.encoder import act_fn_dict
from neurolib.utils.directives import NodeDirectives

# pylint: disable=bad-indentation, no-member, protected-access

class NormalNode(InnerNode):
  """
  An abstract class for Nodes representing a Normal Distribution.
  
  Given some inputs, a Normal Encoder represents a single multidimensional
  Normal distribution, with input-dependent statistics. 
  
  `len(self.state_sizes) = 1` for a Normal Node.
  """
  def __init__(self,
               builder,
               state_sizes,
               num_inputs,
               is_sequence=False,
               name_prefix=None,
               **dirs):
    """
    Initialize the Normal Node
    """
    self.state_sizes = self.state_sizes_to_list(state_sizes) # before InnerNode__init__ call

    # number of inputs
    self.num_expected_inputs = num_inputs
    
    super(NormalNode, self).__init__(builder,
                                     is_sequence,
                                     name_prefix=name_prefix,
                                     **dirs)
    
    # directives object
    self.directives = NodeDirectives(self.directives)
    self.oslot_names = self.directives.output_names
    
    # shapes
    self.oshapes = self._get_all_oshapes()
    self.state_ranks = self.get_state_size_ranks()
    self.xdim = self.state_sizes[0][0] # set when there is only one state
    
    # Initialize list of free i/o slots
    self._islot_to_itensor = [{} for _ in range(self.num_expected_inputs)]
    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))
        
  def __call__(self, *inputs):
    """
    Evaluate the node on a list of inputs.
    
    Args:
        inputs (tf.Tensor or tuple of tensors) : 
    """
    raise NotImplementedError("")
  
  def concat_inputs(self, islot_to_itensor):
    """
    Concat input tensors
    """
    itensors = []
    for elem in islot_to_itensor:
      itensors.append(elem['main'])
    return tf.concat(itensors, axis=-1)

  def get_outputs(self, islot_to_itensor=None):
    """
    Evaluate the node on a dict of inputs.
    """
    raise NotImplementedError("")
          
  @abstractmethod
  def _build(self):
    """
    """
    raise NotImplementedError("")

  def log_prob(self, ipt):
    """
    Get the loglikelihood of the inputs `ipt` for this distribution
    """
    return self.dist.log_prob(ipt)
  
  def entropy(self):
    """
    Get the entropy for this distribution
    """
    return self.dist.entropy()


class NormalTriLNode(NormalNode):
  """
  A NormalNode with input-dependent mean and variance, and the variance
  specified through a lower triangular scale matrix.
  """
  num_expected_outputs = 3
  
  def __init__(self,
               builder,
               state_sizes,
               num_inputs=1,
               is_sequence=False,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize a NormalInputNode
    
    Args:
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes

      state_sizes (int or list of list of ints): The sizes of the outputs

      num_inputs (int): The number of inputs to this node

      is_sequence (bool) :
      
      name (str): A unique string identifier for this node

      name_prefix (str):
      
      dirs (dict): A set of user specified directives for constructing this
          node
    """
    # names
    name_prefix = name_prefix or 'NormalTriL'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    super(NormalTriLNode, self).__init__(builder,
                                         state_sizes=state_sizes,
                                         num_inputs=num_inputs,
                                         is_sequence=is_sequence,
                                         name_prefix=name_prefix,
                                         **dirs)
    
    # expect distribution
    self.dist = None
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'loc_numlayers' : 3,
                      'loc_numnodes' : 128,
                      'loc_activations' : 'leaky_relu',
                      'loc_netgrowrate' : 1.0,
                      'loc_rangeNNws' : 1.0e-5,
                      'scale_numlayers' : 3,
                      'scale_numnodes' : 128,
                      'scale_activations' : 'leaky_relu',
                      'scale_netgrowrate' : 1.0,
                      'scale_rangeNNws' : 0.01,
                      'shareparams' : False,
                      'outputname_1' : 'loc',
                      'outputname_2' : 'scale'}
    this_node_dirs.update(dirs)
    super(NormalTriLNode, self)._update_directives(**this_node_dirs)
    
  def _get_all_oshapes(self):
    """
    Declare the shapes for every output
    """
    bsz = self.batch_size
    mx_stps = self.max_steps
    const_sh = [bsz, mx_stps] if self.is_sequence else [bsz]
    
    xdim = self.state_sizes[0][0]
    return {self.oslot_names[0] : const_sh + [xdim],
            self.oslot_names[1] : const_sh + [xdim],
            self.oslot_names[2] : const_sh + [xdim, xdim]}
  
  def __call__(self, *inputs):
    """
    Call the NormalTriLNode with user-provided inputs
    """
    if not inputs:
      raise ValueError("Inputs are mandatory for calling the NormalTriLNode")
    islot_to_itensor = [{'main' : ipt} for ipt in inputs]
    return self.get_outputs(islot_to_itensor)
    
  def get_loc(self, _input):
    """
    Get the loc of the distribution.
    
    Args:
      _input (tf.Tensor): The inputs to the node
    """
    dirs = self.directives
    
    # get directives
    numlayers = dirs.loc_numlayers
    numnodes = dirs.loc_numnodes
    activations = dirs.loc_activations
    rangeNNws = dirs.loc_rangeNNws

    # get locs
    scope_suffix = "_loc"
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      # 1st layer
      act = act_fn_dict[activations[0]]
      hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(numnodes[0]))
      hid_init_w = tf.random_normal_initializer(stddev=rangeNNws[0])
      hid_layer = fully_connected(_input,
                                  numnodes[0],
                                  activation_fn=act,
#                                   weights_initializer=hid_init_w,
                                  biases_initializer=hid_init_b)
      # 2 -> n-1th layer
      for n in range(1, numlayers-1):
        hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(numnodes[n]))
        hid_init_w = tf.random_normal_initializer(stddev=rangeNNws[n])
        act = act_fn_dict[activations[n]]
        hid_layer = fully_connected(hid_layer,
                                    numnodes[n],
                                    activation_fn=act,
                                    weights_initializer=hid_init_w,
                                    biases_initializer=hid_init_b)
      # nth layer 
      act = act_fn_dict[activations[numlayers-1]]
      loc = fully_connected(hid_layer,
                            self.xdim,
                            activation_fn=act)
    
    return loc, hid_layer
  
  def get_scale_tril(self, _input, hid_layer=None):
    """
    Get the Cholesky decomposition of the variance
    
    Args:
        _input (tf.Tensor) : The inputs to the node
        
        hid_layer (tf.Tensor) : If dirs['share_params'] == True, then 
    """
    dirs = self.directives
    
    # get directives
    numlayers = dirs.scale_numlayers
    numnodes = dirs.scale_numnodes
    activations = dirs.scale_activations
    rangeNNws = dirs.scale_rangeNNws
        
    scope_suffix = "_scale"
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      if dirs.shareparams:
        scale = fully_connected(hid_layer,
                                self.xdim**2,
                                activation_fn=None)
      else:
        # 1st layer
        hid_init_b = tf.random_normal_initializer(stddev=0.1/np.sqrt(numnodes[0]))
#         hid_init_w = tf.random_normal_initializer(stddev=range_scaleNN_w)
        hid_init_w = tf.orthogonal_initializer(gain=rangeNNws[0])
        act = act_fn_dict[activations[0]]
        hid_layer = fully_connected(_input,
                                    numnodes[0],
                                    activation_fn=act,
                                    weights_initializer=hid_init_w,
                                    biases_initializer=hid_init_b)
        # 1 -> n-1th layer
        for n in range(1, numlayers-1):
          hid_init_b = tf.random_normal_initializer(stddev=0.1/np.sqrt(numnodes[n]))
#           hid_init_w = tf.random_normal_initializer(stddev=range_scaleNN_w)
          hid_init_w = tf.orthogonal_initializer(gain=rangeNNws[n])
          act = act_fn_dict[activations[n]]
          hid_layer = fully_connected(hid_layer,
                                      numnodes[n],
                                      activation_fn=act,
                                      weights_initializer=hid_init_w,
                                      biases_initializer=hid_init_b)
        # nth layer
        hid_init_w = tf.orthogonal_initializer(gain=rangeNNws[numlayers-1])
#         hid_init_b = tf.random_normal_initializer(stddev=0.1/np.sqrt(numnodes))
        act = act_fn_dict[activations[numlayers-1]]
        scale = fully_connected(hid_layer,
                                self.xdim**2,
                                activation_fn=act,
                                weights_initializer=hid_init_w)
      
      # select the lower triangular piece
      sc_shape = ([-1, self.max_steps, self.xdim, self.xdim] if self.is_sequence
                  else [-1, self.xdim, self.xdim])
      scale = tf.reshape(scale, shape=sc_shape)
      scale = tf.matrix_band_part(scale, -1, 0)
      
    return scale
  
  def get_dist(self, loc, scale):
    """
    Get tf distribution given loc and scale
    """
    return MultivariateNormalTriL(loc=loc, scale_tril=scale)
    
  def get_outputs(self, islot_to_itensor=None):
    """
    Get the outputs of the NormalTriLNode
    """
    if islot_to_itensor is not None:
      _input = islot_to_itensor
    else:
      _input = self._islot_to_itensor
    _input = self.concat_inputs(_input)
  
    loc, hid_layer = self.get_loc(_input)
    scale = self.get_scale_tril(_input, hid_layer)
    dist = self.get_dist(loc, scale)
    samp = dist.sample()
    
    return samp, loc, scale, dist
  
  def _build(self):
    """
    Build the NormalTriLNode
    """
    samp, loc, scale, self.dist = self.get_outputs()
    
    self.fill_oslot_with_tensor(0, samp)
    self.fill_oslot_with_tensor(1, loc)
    self.fill_oslot_with_tensor(2, scale)

    self._is_built = True

  
class LDSNode(NormalNode):
  """
  A NormalNode with constant variance whose mean is a linear transformation of
  the input.

  TODO: Efficient logprob and entropy
  """
  num_expected_outputs = 4
  
  def __init__(self,
               builder,
               state_sizes,
               num_inputs=1,
               is_sequence=False,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize the LDSNode
    """
    name_prefix = name_prefix or 'LDS'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    super(LDSNode, self).__init__(builder,
                                  state_sizes=state_sizes,
                                  num_inputs=num_inputs,
                                  is_sequence=is_sequence,
                                  name_prefix=name_prefix,
                                  **dirs)
    
    # expect distribution    
    self.dist = None

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'outputname_1' : 'loc',
                      'outputname_2' : 'invQ',
                      'outputname_3' : 'A'}
    this_node_dirs.update(dirs)
    super(LDSNode, self)._update_directives(**this_node_dirs)
    
  def _get_all_oshapes(self):
    """
    Declare the shapes for every output
    """
    bsz = self.batch_size
    mx_stps = self.max_steps
    const_sh = [bsz, mx_stps] if self.is_sequence else [bsz]
    
    xdim = self.state_sizes[0][0]
    return {self.oslot_names[0] : const_sh + [xdim],
            self.oslot_names[1] : const_sh + [xdim],
            self.oslot_names[2] : const_sh + [xdim, xdim],
            self.oslot_names[3] : const_sh + [xdim, xdim]}
  
  def __call__(self, *inputs):
    """
    Evaluate the node on a list of inputs.
    """
    if not inputs:
      raise ValueError("Inputs are mandatory for calling the LDSNode")
    islot_to_itensor = [{'main' : ipt} for ipt in inputs]
    return self.get_outputs(islot_to_itensor)
    
  def get_A(self):
    """
    Declare the evolution matrix A
    """
    xdim = self.xdim
    scope_suffix = "_A"
#     oshape = [1, self.xdim, self.xdim]
    oshape = [xdim, xdim]
    dummy_bsz = tf.concat([self.builder.dummy_bsz, [1, 1]], axis=0)
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
#       eye = np.expand_dims(np.eye(self.xdim, dtype=np.float64), axis=0)
      eye = np.eye(xdim, dtype=np.float64)
      eye = tf.constant_initializer(eye)
      A = tf.get_variable('A',
                          shape=oshape,
                          dtype=tf.float64,
                          initializer=eye)
      A = tf.expand_dims(A, axis=0)
      A = tf.tile(A, dummy_bsz) 
      A.set_shape([None, xdim, xdim]) # tf cannot deduce the shape here
    return A
    
  def get_loc(self, _input, A):
    """
    Declare the loc
    """
    scope_suffix = "_loc"
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      _input = tf.expand_dims(_input, axis=1)
      loc = tf.matmul(_input, A) # in this order A
      loc = tf.squeeze(loc, axis=1) 

    return loc
  
  def get_invQ(self):
    """
    Declare the precision (inverse covariance)
    """
    xdim = self.xdim
    scope_suffix = "_invQ"
#     oshape = [1, self.xdim, self.xdim]
    oshape = [xdim, xdim]
    dummy_bsz = tf.concat([self.builder.dummy_bsz, [1, 1]], axis=0)
    with tf.variable_scope(self.name+scope_suffix, reuse=tf.AUTO_REUSE):
#       eye = np.expand_dims(np.eye(self.xdim, dtype=np.float64), axis=0)
      eye = np.eye(xdim, dtype=np.float64)
      eye = tf.constant_initializer(eye)
      invQscale = tf.get_variable('Q',
                                  shape=oshape,
                                  dtype=tf.float64,
                                  initializer=eye)
      invQscale = tf.expand_dims(invQscale, axis=0)
      invQscale = tf.tile(invQscale, dummy_bsz)
      invQscale.set_shape([None, xdim, xdim]) # tf cannot deduce the shape here
            
      invQ = tf.matmul(invQscale, invQscale, transpose_b=True)
      cov = tf.matrix_inverse(invQ)
    
    return invQ, cov
  
  def get_dist(self, loc, cov):
    """
    Declare the main output
    """
    return MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)

  def get_outputs(self, islot_to_itensor=None):
    """
    Get the outputs of the LDSNode
    """
    if islot_to_itensor is not None:
      _input = islot_to_itensor
    else:
      _input = self._islot_to_itensor
    _input = self.concat_inputs(_input)
      
    A = self.get_A()
    loc = self.get_loc(_input, A)
    invQ, cov = self.get_invQ()
    dist = self.get_dist(loc, cov)
    samp = dist.sample()
    
    return samp, loc, invQ, A, dist
  
  def _build(self):
    """
    Build the LDSNode
    """
    samp, loc, invQ, A, self.dist = self.get_outputs()
    
    self.fill_oslot_with_tensor(0, samp)
    self.fill_oslot_with_tensor(1, loc)
    self.fill_oslot_with_tensor(2, invQ)
    self.fill_oslot_with_tensor(3, A)

    self._is_built = True
    

class NormalPrecisionNode(NormalNode):
  """
  An InnerNode that takes an input and outputs a sample from a normal
  distribution - with input-dependent mean and variance - with the covariance
  specified by its inverse.
  """
  num_expected_outputs = 3

  def __init__(self,
               builder,
               state_sizes,
               num_inputs=1,
               is_sequence=False,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize a NormalInputNode
    
    Args:
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes

      num_inputs (int): The number of inputs to this node

      state_sizes (int or list of list of ints): The sizes of the outputs

      name (str): A unique string identifier for this node

      dirs (dict): A set of user specified directives for constructing this
          node
    """
    # names
    name_prefix = name_prefix or 'NormalPrecision'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    super(NormalPrecisionNode, self).__init__(builder,
                                              state_sizes=state_sizes,
                                              num_inputs=num_inputs,
                                              is_sequence=is_sequence,
                                              name_prefix=name_prefix,
                                              **dirs)

    # expect distribution
    self.dist = None
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'loc_numlayers' : 2,
                      'loc_numnodes' : 128,
                      'loc_activations' : 'softplus',
                      'loc_netgrowrate' : 1.0,
                      'loc_rangeNNws' : 1.0,
                      'prec_numlayers' : 2,
                      'prec_numnodes' : 128,
                      'prec_activations' : 'softplus',
                      'prec_netgrowrate' : 1.0,
                      'prec_rangeNNws' : 1.0,
                      'wconstprecision' : False,
                      'shareparams' : False,
                      'outputname_1' : 'loc',
                      'outputname_2' : 'prec'}
    this_node_dirs.update(dirs)
    super(NormalPrecisionNode, self)._update_directives(**this_node_dirs)

  def _get_all_oshapes(self):
    """
    Declare the shapes for every output
    """
    bsz = self.batch_size
    mx_stps = self.max_steps
    const_sh = [bsz, mx_stps] if self.is_sequence else [bsz]
    
    xdim = self.state_sizes[0][0]
    return {self.oslot_names[0] : const_sh + [xdim],
            self.oslot_names[1] : const_sh + [xdim],
            self.oslot_names[2] : const_sh + [xdim, xdim]}
  
  def __call__(self, *inputs):
    """
    Evaluate the node on a list of inputs.
    """
    if not inputs:
      raise ValueError("Inputs are mandatory for calling the LDSNode")
    islot_to_itensor = [{'main' : ipt} for ipt in inputs]
    return self.get_outputs(islot_to_itensor)
      
  def get_loc(self, _input):
    """
    Get the loc of the distribution.
    
    Args:
      _input (tf.Tensor): The inputs to the node
    """
    dirs = self.directives
    
    # get directives
    numlayers = dirs.loc_numlayers
    numnodes = dirs.loc_numnodes
    activations = dirs.loc_activations
    rangeNNws = dirs.loc_rangeNNws

    scope_suffix = "_loc"
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      # 1st layer
      act = act_fn_dict[activations[0]]
      hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(numnodes[0]))
      hid_init_w = tf.random_normal_initializer(stddev=rangeNNws[0])
      hid_layer = fully_connected(_input,
                                  numnodes[0],
                                  activation_fn=act,
#                                   weights_initializer=hid_init_w,
                                  biases_initializer=hid_init_b)

      # 2 -> n-1th layer
      for n in range(1, numlayers-1):
        hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(numnodes[n]))
        hid_init_w = tf.random_normal_initializer(stddev=rangeNNws[n])
        act = act_fn_dict[activations[n]]
        hid_layer = fully_connected(hid_layer,
                                    numnodes[n],
                                    activation_fn=act,
                                    weights_initializer=hid_init_w,
                                    biases_initializer=hid_init_b)

      # nth layer 
      act = act_fn_dict[activations[numlayers-1]]
      loc = fully_connected(hid_layer,
                            self.xdim,
                            activation_fn=act)
    
    return loc, hid_layer

  def get_precision(self, _input, hid_layer=None):
    """
    Get the Cholesky decomposition of the variance
    
    Args:
        _input (tf.Tensor) : The inputs to the node
        
        hid_layer (tf.Tensor) : If dirs['share_params'] == True, then 
    """
    dirs = self.directives
    
    # get directives
    numlayers = dirs.prec_numlayers
    numnodes = dirs.prec_numnodes
    activations = dirs.prec_activations
    rangeNNws = dirs.prec_rangeNNws
    
    scope_suffix = "_precision"
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      if dirs.wconstprecision:
        lmbda_chol_init = tf.cast(tf.eye(self.xdim), tf.float64)
        lmbda_chol = tf.get_variable('lmbda_chol',
                                     initializer=lmbda_chol_init)
        lmbda = tf.matmul(lmbda_chol, lmbda_chol, transpose_b=True)
      else:
        if dirs.shareparams:
          lmbda_chol = fully_connected(hid_layer,
                                       self.xdim**2,
                                       activation_fn=None)
        else:
          # 1st layer
          act = act_fn_dict[activations[0]]
          hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(numnodes[0]))
          hid_init_w = tf.random_normal_initializer(stddev=rangeNNws[0])
          hid_layer = fully_connected(_input,
                                      numnodes[0],
                                      activation_fn=act,
    #                                   weights_initializer=hid_init_w,
                                      biases_initializer=hid_init_b)
          # 2 -> n-1th layer
          for n in range(1, numlayers-1):
            hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(numnodes[n]))
            hid_init_w = tf.random_normal_initializer(stddev=rangeNNws[n])
            act = act_fn_dict[activations[n]]
            hid_layer = fully_connected(hid_layer,
                                        numnodes[n],
                                        activation_fn=act,
                                        weights_initializer=hid_init_w,
                                        biases_initializer=hid_init_b)

          # nth layer 
          hid_init_w = tf.orthogonal_initializer(gain=1.0)
          hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(self.xdim**2))
          lmbda_chol = fully_connected(hid_layer,
                                       self.xdim**2,
                                       activation_fn=act,
                                       weights_initializer=hid_init_w,
                                       biases_initializer=hid_init_b)
        lmbda_chol = tf.reshape(lmbda_chol,
                                shape=[-1, self.max_steps, self.xdim, self.xdim])
        lmbda = tf.matmul(lmbda_chol, lmbda_chol, transpose_b=True)
      
      cov = tf.matrix_inverse(lmbda)
      
    return lmbda, cov
  
  def get_dist(self, loc, cov):
    """
    Get tf distribution given loc and the covariance
    """
    return MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)
    
  def get_outputs(self, islot_to_itensor=None):
    """
    Get the outputs of the NormalTriLNode
    """
    if islot_to_itensor is not None:
      _input = islot_to_itensor
    else:
      _input = self._islot_to_itensor
    _input = self.concat_inputs(_input)
  
    loc, hid_layer = self.get_loc(_input)
    prec, cov = self.get_precision(_input, hid_layer)
    dist = self.get_dist(loc, cov)
    samp = dist.sample()
    
    return samp, loc, prec, dist

  def _build(self):
    """
    Build the NormalTriLNode
    """
    samp, loc, prec, self.dist = self.get_outputs()
    
    self.fill_oslot_with_tensor(0, samp)
    self.fill_oslot_with_tensor(1, loc)
    self.fill_oslot_with_tensor(2, prec)

    self._is_built = True

  def log_prob(self, ipt):
    """
    Get the loglikelihood of the inputs `ipt` for this distribution
    """
    if not self._is_built:
      raise ValueError("`logprob` method unavailable for unbuilt Node")
    
    loc = self.get_output_tensor('loc')
    prec = self.get_output_tensor('prec')
    
    print("loc", loc)
    print("prec", prec)
    xdim = tf.cast(self.xdim, tf.float64)
    Nsamps = tf.shape(loc)[0]
    Tbins = self.max_steps
    N = tf.cast(Nsamps, tf.float64)
    T = tf.cast(Tbins, tf.float64) if self.is_sequence else 1.0
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      delta = tf.expand_dims(ipt - loc, axis=-2)
      L1 = 0.5*N*T*xdim*tf.cast(tf.log(2*np.pi), tf.float64)
      if self.directives.wconstprecision:
        L2 = 0.5*N*T*tf.log(tf.matrix_determinant(prec))
        prec = tf.expand_dims(tf.expand_dims(prec, 0), 0)
        prec = tf.tile(prec, [Nsamps, Tbins, 1, 1])
      else:
        L2 = 0.5*tf.reduce_sum(tf.log(tf.matrix_determinant(prec)))
      L3 = -0.5*tf.matmul(tf.matmul(delta, prec), delta, transpose_b=True)
      L3 = tf.reduce_sum(L3)
    
      log_prob = tf.identity(L1 + L2 + L3, name='logprob')
      self.builder.add_to_output_names(self.name + ':logprob', log_prob)

    return log_prob
    

#   def _build(self):
#     """
#     Builds the graph corresponding to a NormalPrecision node.
#     
#     """
#     islot_to_itensor = self._islot_to_itensor
#     _input = basic_concatenation(islot_to_itensor)
# 
#     mean, hid_layer = self._get_loc(_input)
#     lmbda, cov = self._get_lmbda(_input, hid_layer)
#     self.dist = MultivariateNormalFullCovariance(loc=mean,
#                                                  covariance_matrix=cov)
#     
#     # Fill the oslots
#     mean_name = self.directives['output_1_name']
#     lmbda_name = self.directives['output_2_name']
#     self._oslot_to_otensor[0] = self.dist.sample(name='Out' + 
#                                                  str(self.label) + '_0')
#     self._oslot_to_otensor[1] = tf.identity(mean, name=mean_name)
#     self._oslot_to_otensor[2] = tf.identity(lmbda, name=lmbda_name)
#     
#     self._is_built = True
        
  
class LocallyLinearNormalNode(NormalNode):
  """
  """
  @abstractmethod
  def _build(self):
    """
    """
    NormalNode._build(self)
  