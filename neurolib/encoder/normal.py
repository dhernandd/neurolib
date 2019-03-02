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
from neurolib.encoder import (MultivariateNormalTriL, # @UnresolvedImport
                              MultivariateNormalFullCovariance) # @UnresolvedImport
                              
from neurolib.encoder import act_fn_dict, layers_dict
from neurolib.utils.directives import NodeDirectives
from neurolib.utils.shapes import infer_shape

# from tensorflow.contrib.distributions.python.ops.mvn_linear_operator import MultivariateNormalLinearOperator

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
               main_inputs,
#                num_inputs,
               is_sequence=False,
               name_prefix=None,
               **dirs):
    """
    Initialize the Normal Node
    """
    self.state_sizes = self.state_sizes_to_list(state_sizes) # before InnerNode__init__ call

    # inputs
    self.main_inputs = main_inputs
    self.num_expected_inputs = len(main_inputs)
    
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
  
  def build_outputs(self, **inputs):
    """
    Evaluate the node on a dict of inputs.
    """
    raise NotImplementedError("")
  
  def prepare_inputs(self, **inputs):
    """
    Concat input tensors
    """
    raise NotImplementedError("")
    
  @abstractmethod
  def _build(self):
    """
    """
    raise NotImplementedError("")
  
#   def sample(self, **ipt):
#     """
#     """
#     try:
#       return self._sample(**ipt)
#     except AttributeError:
#       try:
#         return ipt['dist'].sample()
#       except KeyError:
#         return self.dist.sample()

  def _sample(self, **pars):
    """
    TODO: Add explicit parameters that can be passed to a tensorflow distributionn
    
    TODO: Change this method if more efficient way to sample than tensorflow's
    default.
    """
    try:
      raise NotImplementedError
    except NotImplementedError:
      samp = self.dist.sample(**pars)
      return samp
    
  def log_prob(self, ipt):
    """
    Get the loglikelihood of the inputs `ipt` for this distribution
    """
    try:
      return self._logprob(ipt)
    except AttributeError:
      return self.dist.log_prob(ipt)
  
  def entropy(self):
    """
    Get the entropy for this distribution
    """
    try:
      return self._entropy()
    except AttributeError:
      return self.dist.entropy()
    
  def concat_inputs(self, **inputs):
    """
    Concat input tensors
    """
    input_list = [inputs['imain'+str(i)] for i in range(self.num_expected_inputs)]
    main_input = tf.concat(input_list, axis=-1)

    return main_input
      


class NormalTriLNode(NormalNode):
  """
  A NormalNode with input-dependent mean and variance, and the variance
  specified through a lower triangular scale matrix.
  """
  num_expected_outputs = 3
  
  def __init__(self,
               builder,
               state_size,
               main_inputs,
#                num_inputs=1,
               is_sequence=False,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize a NormalTriLNode
    
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
                                         state_sizes=state_size,
                                         main_inputs=main_inputs,
#                                          num_inputs=num_inputs,
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
                      'scale_numlayers' : 3,
                      'scale_numnodes' : 128,
                      'scale_activations' : 'leaky_relu',
                      'scale_netgrowrate' : 1.0,
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
      raise ValueError("Inputs are mandatory for the NormalTriLNode")
    inputs = {'imain' + str(i) : inputs[i] for i in range(len(inputs))}
    return self.build_outputs(**inputs)
    
  def _build(self):
    """
    Build the NormalTriLNode
    """
    self.build_outputs()
    
    self._is_built = True

  def build_outputs(self, **all_user_inputs):
    """
    Get the outputs of the NormalTriLNode
    """
    print("Building all outputs, ", self.name)
    
    self.build_output('loc', **all_user_inputs)
    self.build_output('scale', **all_user_inputs)
    self.build_output('main', **all_user_inputs)
      
  def build_output(self, oname, **all_user_inputs):
    """
    Build a single output for this node
    """
    if oname == 'loc':
      return self.build_loc(**all_user_inputs)
    elif oname == 'scale':
      return self.build_scale_tril(**all_user_inputs)
    elif oname == 'main':
      return self.build_main(**all_user_inputs)
    else:
      raise ValueError("`oname` {} is not an output name for "
                       "this node".format(oname))
    
  def prepare_inputs(self, **inputs):
    """
    Prepare the inputs
    
    TODO: Use the islots directive to define main_inputs
    """
    islot_to_itensor = self._islot_to_itensor
    main_inputs = {'imain' + str(i) : islot_to_itensor[i]['main'] for i in 
                  range(self.num_expected_inputs)}
    if inputs:
      print("Updating defaults,", self.name, "with", list(inputs.keys()))
      main_inputs.update(inputs)

    return main_inputs
  
  def build_loc(self, **inputs):
    """
    Build the loc of the distribution.
    
    Args:
      _input (tf.Tensor): The inputs to the node
    """
    print("\tBuilding loc, ", self.name)
    
    dirs = self.directives
    
    # get directives
    layers  = dirs.loc_layers
    numlayers = dirs.loc_numlayers
    numnodes = dirs.loc_numnodes
    activations = dirs.loc_activations
    winitializers = dirs.loc_winitializers
    binitializers = dirs.loc_binitializers

    # get locs
    scope_suffix = "_loc"
    inputs = self.prepare_inputs(**inputs)
    _input = self.concat_inputs(**inputs)
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      if numlayers == 1:
        layer = layers_dict[layers[0]]
        act = act_fn_dict[activations[0]]
        winit = winitializers[0]
        binit = binitializers[0]
        loc = layer(_input,
                    self.xdim,
                    activation_fn=act,
                    weights_initializer=winit,
                    biases_initializer=binit)
      else:
        # 1st layer
        layer = layers_dict[layers[0]]     
        act = act_fn_dict[activations[0]]
        winit = winitializers[0]
        binit = binitializers[0]
        hid_layer = layer(_input,
                          numnodes[0],
                          activation_fn=act,
                          weights_initializer=winit,
                          biases_initializer=binit)
        # 2 -> n-1th layer
        for n in range(1, numlayers-1):
          layer = layers_dict[layers[n]]  
          act = act_fn_dict[activations[n]]
          winit = winitializers[n]
          binit = binitializers[n]
          hid_layer = layer(hid_layer,
                            numnodes[n],
                            activation_fn=act,
                            weights_initializer=winit,
                            biases_initializer=binit)
        # nth layer 
        layer = layers_dict[layers[numlayers-1]]
        act = act_fn_dict[activations[numlayers-1]]
        winit = winitializers[numlayers-1]
        binit = binitializers[numlayers-1]
        loc = layer(hid_layer,
                    self.xdim,
                    activation_fn=act,
                    weights_initializer=winit,
                    biases_initializer=binit)
    
    if not self._is_built:
      self.fill_oslot_with_tensor(1, loc)
    
    return loc
  
  def build_scale_tril(self, **inputs):
    """
    Get the Cholesky decomposition of the variance
    
    Args:
        _input (tf.Tensor) : The inputs to the node
        
        hid_layer (tf.Tensor) : If dirs['share_params'] == True, then 
    """
    print("\tBuilding scale, ", self.name)
    
    dirs = self.directives
    
    # get directives
    layers  = dirs.scale_layers
    numlayers = dirs.scale_numlayers
    numnodes = dirs.scale_numnodes
    activations = dirs.scale_activations
    winitializers = dirs.scale_winitializers
    binitializers = dirs.scale_binitializers
        
    scope_suffix = "_scale"
    xdim = self.xdim
    inputs = self.prepare_inputs(**inputs)
    _input = self.concat_inputs(**inputs)
    input_shape = infer_shape(_input)
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      if numlayers == 1:
        scale = fully_connected(_input, xdim**2,
                                activation_fn=None)
      else:
        # 1st layer
        layer = layers_dict[layers[0]]     
        act = act_fn_dict[activations[0]]
        winit = winitializers[0]
        binit = binitializers[0]
        hid_layer = layer(_input,
                          numnodes[0],
                          activation_fn=act,
                          weights_initializer=winit,
                          biases_initializer=binit)
        # 2 -> n-1th layer
        for n in range(1, numlayers-1):
          layer = layers_dict[layers[n]]  
          act = act_fn_dict[activations[n]]
          winit = winitializers[n]
          binit = binitializers[n]
          hid_layer = layer(hid_layer,
                            numnodes[n],
                            activation_fn=act,
                            weights_initializer=winit,
                            biases_initializer=binit)
        # nth layer 
        layer = layers_dict[layers[numlayers-1]]
        act = act_fn_dict[activations[numlayers-1]]
        winit = winitializers[numlayers-1]
        binit = binitializers[numlayers-1]
        scale = layer(hid_layer,
                      self.xdim**2,
                      activation_fn=act,
                      weights_initializer=winit,
                      biases_initializer=binit)
      
      # select the lower triangular piece
      sc_shape = input_shape[:-1] + [xdim, xdim]
      scale = tf.reshape(scale, shape=sc_shape)
      scale = tf.matrix_band_part(scale, -1, 0)
      
    if not self._is_built:
      self.fill_oslot_with_tensor(2, scale)

    return scale
  
  def build_main(self, **inputs):
    """
    Build main output
    """
    return self.build_main_secs(**inputs)[0]
  
  def build_main_secs(self, **inputs):
    """
    Build main and secondary outputs
    """
    print("\tBuilding main, ", self.name)
    
    if inputs:
      loc = self.build_loc(**inputs)
      scale = self.build_scale(**inputs)
      dist = MultivariateNormalTriL(loc=loc, scale_tril=scale)
      samp = dist.sample() # Add parameters here
      return samp, (loc, scale)
    else:
      if not self._is_built:
        loc = self.get_output_tensor('loc')
        scale = self.get_output_tensor('scale')
        self.dist = MultivariateNormalTriL(loc=loc, scale_tril=scale)
        samp = self.dist.sample()  # Add parameters here
        self.fill_oslot_with_tensor(0, samp)
      else:
        samp = self.sample()  # Add parameters here
      
      return samp, ()
        
  
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
               main_inputs,
#                num_inputs=1,
               is_sequence=False,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize a NormalPrecisionNode
    
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
                                              main_inputs=main_inputs,
#                                               num_inputs=num_inputs,
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
                      'loc_numnodes' : 64,
                      'loc_activations' : 'softplus',
                      'loc_netgrowrate' : 1.0,
                      'prec_numlayers' : 3,
                      'prec_numnodes' : 64,
                      'prec_activations' : 'softplus',
                      'prec_netgrowrate' : 1.0,
                      'wconstprecision' : False,
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
    Call the NormalPrecisionNode with user-provided inputs
    """
    if not inputs:
      raise ValueError("Inputs are mandatory for the NormalPrecisionNode")
    inputs = {'imain' + str(i) : inputs[i] for i in range(len(inputs))}
    return self.build_outputs(**inputs)
    
  def _build(self):
    """
    Build the NormalTriLNode
    """
    self.build_outputs()
    
#     self.fill_oslot_with_tensor(0, samp)
#     self.fill_oslot_with_tensor(1, loc)
#     self.fill_oslot_with_tensor(2, prec)

    self._is_built = True

  def build_outputs(self, **all_user_inputs):
    """
    Get the outputs of the NormalPrecisionNode
    """
    print("Building all outputs, ", self.name)
    
    self.build_output('loc', **all_user_inputs)
    self.build_output('prec', **all_user_inputs)
    self.build_output('main', **all_user_inputs)
    
  def build_output(self, oname, **inputs):
    """
    Build a single output
    """
    if oname == 'loc':
      return self.build_loc(**inputs)
    elif oname == 'prec':
      return self.build_prec(**inputs)
    elif oname == 'main':
      return self.build_main(**inputs)
    else:
      raise ValueError("`oname` {} is not an output name for "
                       "this node".format(oname))
    
  def prepare_inputs(self, **inputs):
    """
    Prepare the inputs
    
    TODO: Use the islots directive to define main_inputs
    """
    islot_to_itensor = self._islot_to_itensor
    main_inputs = {'imain' + str(i) : islot_to_itensor[i]['main'] for i in 
                  range(self.num_expected_inputs)}
    if inputs:
      print("\t\tUpdating defaults,", self.name, "with", list(inputs.keys()))
      main_inputs.update(inputs)

    return main_inputs
  
  def build_loc(self, **inputs):
    """
    Get the loc of the distribution.
    
    Args:
      _input (tf.Tensor): The inputs to the node
    """
    print("\tBuilding loc, ", self.name)
    
    dirs = self.directives
    
    # get directives
    layers  = dirs.loc_layers
    numlayers = dirs.loc_numlayers
    numnodes = dirs.loc_numnodes
    activations = dirs.loc_activations
    winitializers = dirs.loc_winitializers
    binitializers = dirs.loc_binitializers
    
    scope_suffix = "_loc"
    inputs = self.prepare_inputs(**inputs)
    _input = self.concat_inputs(**inputs)
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      if numlayers == 1:
        layer = layers_dict[layers[0]]
        act = act_fn_dict[activations[0]]
        winit = winitializers[0]
        binit = binitializers[0]
        loc = layer(_input,
                    self.xdim,
                    activation_fn=act,
                    weights_initializer=winit,
                    biases_initializer=binit)
      else:
        # 1st layer
        layer = layers_dict[layers[0]]
        act = act_fn_dict[activations[0]]
        winit = winitializers[0]
        binit = binitializers[0]
        hid_layer = layer(_input,
                          numnodes[0],
                          activation_fn=act,
                          weights_initializer=winit,
                          biases_initializer=binit)
        # 2 -> n-1th layer
        for n in range(1, numlayers-1):
          layer = layers_dict[layers[n]]
          act = act_fn_dict[activations[n]]
          winit = winitializers[n]
          binit = binitializers[n]
          hid_layer = layer(hid_layer,
                            numnodes[n],
                            activation_fn=act,
                            weights_initializer=winit,
                            biases_initializer=binit)
        # nth layer 
        layer = layers_dict[layers[numlayers-1]]
        act = act_fn_dict[activations[numlayers-1]]
        winit = winitializers[numlayers-1]
        binit = binitializers[numlayers-1]
        loc = layer(hid_layer,
                    self.xdim,
                    activation_fn=act,
                    weights_initializer=winit,
                    biases_initializer=binit)

    if not self._is_built:
      self.fill_oslot_with_tensor(1, loc)

    return loc

  def build_prec(self, **inputs):
    """
    Get the Cholesky decomposition of the variance
    
    Args:
        _input (tf.Tensor) : The inputs to the node
        
        hid_layer (tf.Tensor) : If dirs['share_params'] == True, then 
    """
    print("\tBuilding prec, ", self.name)
    
    dirs = self.directives
    
    # get directives
    layers  = dirs.loc_layers
    numlayers = dirs.prec_numlayers
    numnodes = dirs.prec_numnodes
    activations = dirs.prec_activations
    winitializers = dirs.prec_winitializers
    binitializers = dirs.prec_binitializers
    
    scope_suffix = "_precision"
    inputs = self.prepare_inputs(**inputs)
    _input = self.concat_inputs(**inputs)
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      if dirs.wconstprecision:
        lmbda_chol_init = tf.cast(tf.eye(self.xdim), tf.float64)
        lmbda_chol = tf.get_variable('lmbda_chol',
                                     initializer=lmbda_chol_init)
        lmbda = tf.matmul(lmbda_chol, lmbda_chol, transpose_b=True)
      else:
        if numlayers == 1:
          layer = layers_dict[layers[0]]
          act = act_fn_dict[activations[0]]
          winit = winitializers[0]
          binit = binitializers[0]
          lmbda_chol = layer(_input,
                             self.xdim**2,
                             activation_fn=act,
                             weights_initializer=winit,
                             biases_initializer=binit)
        else:
          # 1st layer          
          act = act_fn_dict[activations[0]]
          winit = winitializers[0]
          binit = binitializers[0]
          hid_layer = fully_connected(_input,
                                      numnodes[0],
                                      activation_fn=act,
                                      weights_initializer=winit,
                                      biases_initializer=binit)
          # 2 -> n-1th layer
          for n in range(1, numlayers-1):
            act = act_fn_dict[activations[n]]
            winit = winitializers[n]
            binit = binitializers[n]
            hid_layer = fully_connected(hid_layer,
                                        numnodes[n],
                                        activation_fn=act,
                                        weights_initializer=winit,
                                        biases_initializer=binit)

          # nth layer 
          act = act_fn_dict[activations[numlayers-1]]
          winit = winitializers[numlayers-1]
          binit = binitializers[numlayers-1]
          lmbda_chol = fully_connected(hid_layer,
                                       self.xdim**2,
                                       activation_fn=act,
                                       weights_initializer=winit,
                                       biases_initializer=binit)
        lmbda_chol = tf.reshape(lmbda_chol,
                                shape=[-1, self.max_steps, self.xdim, self.xdim])
        lmbda = tf.matmul(lmbda_chol, lmbda_chol, transpose_b=True)
      
    if not self._is_built:
      self.fill_oslot_with_tensor(2, lmbda)

    return lmbda
  
  def build_main(self, **inputs):
    """
    Build main output
    """
    return self.build_main_secs(**inputs)[0]
  
  def build_main_secs(self, **inputs):
    """
    Build main and secondary outputs
    """
    print("\tBuilding main, ", self.name)
    if inputs:
      loc = self.build_loc(**inputs)
      prec = self.build_prec(**inputs)
      cov = tf.matrix_inverse(prec)
      dist = MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)
      samp = dist.sample() # Add parameters here
      return samp, (loc, prec, cov)
    else:
      if not self._is_built:
        loc = self.get_output_tensor('loc')
        prec = self.get_output_tensor('prec')
        cov = tf.matrix_inverse(prec)
        self.dist = MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)
        samp = self.dist.sample()  # Add parameters here
        self.fill_oslot_with_tensor(0, samp)
      else:
        samp = self.sample()  # Add parameters here
      
      return samp, ()
    
  def logprob(self, Y):
    """
    Build the loglikelihood given an input tensor
    
    Args:
        X (tf.Tensor) :
    """
    return self.build_logprob(Y)
    
  def build_logprob(self, Y, name=None, **inputs):
    """
    Build the loglikelihood given a dictionary of user inputs
    """
    return self.build_logprob_secs(Y, name=name, **inputs)[0]
    
  def build_logprob_secs(self, Y, name=None, **inputs):
    """
    Get the loglikelihood of the inputs `ipt` for this distribution
    """
    if self.num_expected_inputs > 1:
      raise NotImplementedError
        
    print("Building logprob, ", self.name)
    if not self._is_built:
      raise ValueError("`logprob` method unavailable for unbuilt Node")
    
    if not inputs:
      loc = self.get_output_tensor('loc')
      prec = self.get_output_tensor('prec')
    else:
      loc = self.build_loc(**inputs)
      prec = self.build_prec(**inputs)
    
    xdim = tf.cast(self.xdim, tf.float64)
    Nsamps = tf.shape(loc)[0]
    Tbins = self.max_steps
    N = tf.cast(Nsamps, tf.float64)
    T = tf.cast(Tbins, tf.float64) if self.is_sequence else 1.0
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      delta = tf.expand_dims(Y - loc, axis=-2)
      L1 = 0.5*N*T*xdim*tf.cast(tf.log(2*np.pi), tf.float64)
      if self.directives.wconstprecision:
        L2 = 0.5*N*T*tf.log(tf.matrix_determinant(prec))
        prec = tf.expand_dims(tf.expand_dims(prec, 0), 0)
        prec = tf.tile(prec, [Nsamps, Tbins, 1, 1])
      else:
        L2 = 0.5*tf.reduce_sum(tf.log(tf.matrix_determinant(prec)))
      L3 = -0.5*tf.matmul(tf.matmul(delta, prec), delta, transpose_b=True)
      L3 = tf.reduce_sum(L3)
    
      logprob = tf.identity(L1 + L2 + L3, name='logprob')
    
    if name is None:
      name = self.name + ':logprob'
    else:
      name = self.name + ':' + name
    
    if name in self.builder.otensor_names:
      raise ValueError("name {} has already been defined, pass a different"
                       "argument `name`".format(name))
    self.builder.add_to_output_names(name, logprob)

    return logprob, (loc, prec)


class LinearTransformNormal(NormalNode):
  """
  """
  num_expected_outputs = 4
  
  def __init__(self,
               builder,
               state_sizes,
               num_inputs=2,
               is_sequence=True,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize the LDSNode
    """
    name_prefix = name_prefix or 'LTN'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    super(LinearTransformNormal, self).__init__(builder,
                                                state_sizes=state_sizes,
                                                num_inputs=num_inputs,
                                                is_sequence=is_sequence,
                                                name_prefix=name_prefix,
                                                **dirs)
    
    # expect distribution
    self.dist = None
    
  def __call__(self, *inputs):
    pass

  def prepare_inputs(self, **inputs):
    NormalNode.prepare_inputs(self, **inputs)
    
  def build_outputs(self, **inputs):
    NormalNode.build_outputs(self, **inputs)
    
  @abstractmethod
  def _build(self):
    NormalNode._build(self)
    
  def _get_all_oshapes(self):
    NormalNode._get_all_oshapes(self)
