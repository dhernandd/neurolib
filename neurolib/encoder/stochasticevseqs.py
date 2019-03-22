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
from neurolib.encoder import (MultivariateNormalLinearOperator) # @UnresolvedImport
from neurolib.encoder import act_fn_dict, layers_dict
from neurolib.utils.directives import NodeDirectives
from neurolib.utils.shapes import infer_shape

# from tensorflow.contrib.distributions.python.ops.mvn_linear_operator import MultivariateNormalLinearOperator

# pylint: disable=bad-indentation, no-member, protected-access

class DSEvolution(InnerNode):
  """
  An abstract class for Nodes representing Dynamical Systems evolving with
  Gaussian noise.
  
  Given some inputs, a Normal Encoder represents a single multidimensional
  Normal distribution, with input-dependent statistics. 
  
  `len(self.state_sizes) = 1` for a Normal Node.
  """
  def __init__(self,
               builder,
               state_sizes,
               main_inputs,
               prior_inputs,
               sec_inputs=None,
               name_prefix=None,
               **dirs):
    """
    Initialize the Normal Node
    """
    self.state_sizes = self.state_sizes_to_list(state_sizes) # before InnerNode__init__ call

    # inputs
    self.main_inputs = main_inputs
    self.priors = prior_inputs
    self.num_expected_inputs = len(main_inputs) + len(prior_inputs)
    if sec_inputs is not None:
      self.sec_inputs = sec_inputs
      self.num_sec_inputs = len(sec_inputs)
      self.num_expected_inputs += self.num_sec_inputs
    
    super(DSEvolution, self).__init__(builder,
                                      is_sequence=True,
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
  
  def sample(self, **ipt):
    """
    """
    try:
      return self._sample(**ipt)
    except AttributeError:
      try:
        return ipt['dist'].sample()
      except KeyError:
        return self.dist.sample()
    
  def concat_inputs(self, **inputs):
    """
    Concat input tensors
    """
    input_list = [inputs['imain'+str(i)] for i in range(self.num_expected_inputs)]
    main_input = tf.concat(input_list, axis=-1)

    return main_input
      
    
class LDSEvolution(DSEvolution):
  """
  A NormalNode with constant variance whose mean is a linear transformation of
  the input.
  """
  num_expected_outputs = 5
  
  def __init__(self,
               builder,
               state_sizes,
               main_inputs,
               prior_inputs,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize the LDSNode
    """
    name_prefix = name_prefix or 'LDS'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    super(LDSEvolution, self).__init__(builder,
                                       state_sizes=state_sizes,
                                       main_inputs=main_inputs,
                                       prior_inputs=prior_inputs,
                                       name_prefix=name_prefix,
                                       **dirs)

    # expect distribution
    self.dist = None

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'outputname_1' : 'loc',
                      'outputname_2' : 'A',
                      'outputname_3' : 'prec',
                      'outputname_4' : 'scale'}
    this_node_dirs.update(dirs)
    super(LDSEvolution, self)._update_directives(**this_node_dirs)
    
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
    
    TODO: Probably NOT the behavior we want from call, I think we just want to
    pass it 'imain0'
    """
    if not inputs:
      raise ValueError("Inputs are mandatory for calling an LDSEvolution"
                       "instance")
    inputs = {'iprior_loc' : inputs[0],
              'imain0' : inputs[1]}
    
    return self.build_outputs(**inputs)
    
  def _build(self):
    """
    Build the LDSNode
    """
    self.build_outputs()
    
    self._is_built = True
    
  def build_outputs(self, **all_user_inputs):
    """
    Get the outputs of the LDSEvolution
    """
    print("Building all outputs, ", self.name)
    
    self.build_output('A')
    self.build_output('loc', **all_user_inputs)
    self.build_output('prec')
    self.build_output('main', **all_user_inputs)
  
  def build_output(self, oname, **inputs):
    """
    Build a single output for this node
    """
    if oname == 'A':
      return self.build_A()
    elif oname == 'loc':
      return self.build_loc(**inputs)
    elif oname == 'prec':
      return self.build_prec()
    elif oname == 'main':
      return self.build_main(**inputs)
    else:
      raise ValueError("`oname` {} is not an output name for "
                       "this node".format(oname))    
  
  def prepare_inputs(self, **inputs):
    """
    Prepare inputs for building
    """
    islot_to_itensor = self._islot_to_itensor
    
    true_inputs = {'iprior_loc'  : islot_to_itensor[0]['loc'],
                   'imain0' : islot_to_itensor[1]['main']}
    if inputs:
      print("\t\tUpdating defaults with", list(inputs.keys()), ",", self.name, )
      true_inputs.update(inputs)
    
    return true_inputs

  def build_A(self):
    """
    Declare the evolution matrix A
    """
    print("\tBuilding A, ", self.name)
    xdim = self.xdim
    oshape = [xdim, xdim]
    with tf.variable_scope(self.name + "_A", reuse=tf.AUTO_REUSE):
      eye = np.eye(xdim, dtype=np.float64)
      eye = tf.constant_initializer(eye)
      A = tf.get_variable('A',
                          shape=oshape,
                          dtype=tf.float64,
                          initializer=eye)
    self.fill_oslot_with_tensor(2, A)
    
  def build_loc(self, **inputs):
    """
    Declare the loc
    
    TODO: Rethink this, should loc really truncate the series at time -1, maybe
    it should return the evolved series and forget about the prior
    """
    print("\tBuilding loc, ", self.name)
    A = self.get_output_tensor('A')
    inputs = self.prepare_inputs(**inputs)
    
    prior_input = inputs['iprior_loc']
    seq_input = inputs['imain0']
    seq_shape = infer_shape(seq_input)
    r = len(seq_shape)    
    with tf.variable_scope(self.name + "_loc", reuse=tf.AUTO_REUSE):
      seq_input = tf.expand_dims(seq_input, axis=-2)
      for _ in seq_shape[r-2::-1]:
        A = tf.expand_dims(A, axis=-3)
      A = tf.tile(A, seq_shape[:-1] + [1, 1])
      loc = tf.matmul(seq_input, A) # in this order A
      loc = tf.squeeze(loc, axis=-2)
      
      
      prior_input = tf.expand_dims(prior_input, axis=1)
      loc = tf.concat([prior_input, loc], axis=1)
      loc = loc[:,:-1]

    if not self._is_built:
      self.fill_oslot_with_tensor(1, loc)
    
    return loc
  
  def build_prec(self):
    """
    Declare the prec (inverse covariance)
    """
    print("\tBuilding prec, scale, ", self.name)
    xdim = self.xdim
    oshape = [xdim, xdim]
    with tf.variable_scope(self.name + "_prec", reuse=tf.AUTO_REUSE):
      eye = np.eye(xdim, dtype=np.float64)
      eye_init = tf.constant_initializer(eye)
      eye_init = tf.get_variable('eye_init',
                                       shape=oshape,
                                       dtype=tf.float64,
                                       initializer=eye_init)
      invscale = tf.linalg.band_part(eye_init, -1, 0)
      prec = tf.matmul(invscale, invscale, transpose_b=True)

      scale = tf.matrix_inverse(invscale) # uses that inverse of LT is LT

    self.fill_oslot_with_tensor(3, prec)
    self.fill_oslot_with_tensor(4, scale)
      
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
    scale = self.get_output_tensor('scale')
    scale = tf.linalg.LinearOperatorLowerTriangular(scale)
    if inputs:
      loc = self.build_loc(**inputs)
      dist = MultivariateNormalLinearOperator(loc=loc, scale=scale)
      samp = dist.sample() # Add parameters here
    else:
      loc = self.get_output_tensor('loc')
      self.dist = MultivariateNormalLinearOperator(loc=loc, scale=scale)
      samp = self.sample() # Add parameters here
      self.fill_oslot_with_tensor(0, samp)
    
    return samp, (loc,)
      
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
    
  def logprob(self, X):
    """
    Build the loglikelihood given an input tensor
    
    Args:
        X (tf.Tensor) :
    """
    return self.build_logprob(imain0=X)
    
  def build_logprob(self, name=None, **inputs):
    """
    Build the loglikelihood given a dictionary of user inputs
    """
    return self.build_logprob_secs(name=name, **inputs)[0]
    
  def build_logprob_secs(self, name=None, **inputs):
    """
    Build the loglikelihood of this node. Return all the tensors that are built
    in the process (loc).
    """
    print("Building logprob, ", self.name)
    
    with tf.variable_scope(self.builder.scope, reuse=tf.AUTO_REUSE):
      if not inputs:
        X = self.get_input_tensor(1, 'main')
        loc = self.get_output_tensor('loc')
      else:
        X = inputs['imain0']
        loc = self.build_loc(**inputs)
        
      loc_sh = infer_shape(loc)
      N, T = loc_sh[0], loc_sh[1]
      l5 = -0.5*np.log(2*np.pi)*tf.cast(N*T*self.xdim, tf.float64)
  
      prior_sc = self._islot_to_itensor[0]['scale']
      invscale = tf.matrix_inverse(prior_sc)
      invQ0 = tf.matmul(invscale, invscale, transpose_b=True)
      l3 = 0.5*(tf.log(tf.matrix_determinant(invQ0))*tf.cast(N, tf.float64))

      deltax0 = X[:,0] - loc[:,0]
      l1 = -0.5*tf.reduce_sum(deltax0*tf.matmul(deltax0, invQ0))
  
      invQ = self.get_output_tensor('prec')
      l4 = 0.5*(tf.log(tf.matrix_determinant(invQ))*tf.cast(N*(T-1), tf.float64))
  
      loc = tf.expand_dims(loc, axis=-2)
      X = tf.expand_dims(X, axis=-2)
      r = len(loc_sh)
      for _ in loc_sh[r-2::-1]:
        invQ = tf.expand_dims(invQ, axis=-3)
      invQ = tf.tile(invQ, loc_sh[:-1] + [1, 1])
      deltax = X[:,1:] - loc[:,1:]
      l2 = -0.5*tf.reduce_sum(deltax*tf.matmul(deltax, invQ[:,1:]))
      
      logprob = l1 + l2 + l3 + l4 + l5

    if name is None:
      name = self.name + ':logprob'
    else:
      name = self.name + ':' + name
    
    if name in self.builder.otensor_names:
      raise ValueError("name {} has already been defined, pass a different"
                       "argument `name`".format(name))
    self.builder.add_to_output_names(name, logprob)
          
    return logprob, (loc,)


class LLDSEvolution(DSEvolution):
  """
  A NormalNode with constant variance whose loc is a nonlinear, NN-parameterized
  transformation of the input.
  """
  num_expected_outputs = 6
  
  def __init__(self,
               builder,
               state_sizes,
               main_inputs,
               prior_inputs,
               sec_inputs=None,
#                num_inputs=2,
#                is_sequence=False,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize the LLDSNode
    """
    name_prefix = name_prefix or 'LLDS'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    super(LLDSEvolution, self).__init__(builder,
                                        state_sizes=state_sizes,
                                        main_inputs=main_inputs,
                                        prior_inputs=prior_inputs,
                                        sec_inputs=sec_inputs,
#                                         num_inputs=num_inputs,
#                                         is_sequence=True,
                                        name_prefix=name_prefix,
                                        **dirs)
    
    # expect distribution
    self.dist = None
        
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'A_numlayers' : 3,
                      'A_numnodes' : 256,
                      'A_activations' : 'relu',
                      'A_netgrowrate' : 1.0,
                      'outputname_1' : 'loc',
                      'outputname_2' : 'A',
                      'outputname_3' : 'prec',
                      'outputname_4' : 'Alinear',
                      'outputname_5' : 'scale',
                      'alpha' : 0.1}
    this_node_dirs.update(dirs)
    super(LLDSEvolution, self)._update_directives(**this_node_dirs)
    
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
            self.oslot_names[3] : const_sh + [xdim, xdim],
            self.oslot_names[3] : const_sh + [xdim, xdim]}

  def __call__(self, *inputs):
    """
    Evaluate the node on a list of inputs.
    
    TODO: Probably NOT the behavior we want from call, I think we just want to
    pass it 'imain0'
    """
    if not inputs:
      raise ValueError("Inputs are mandatory for calling an LDSEvolution"
                       "instance")
    inputs = {'iprior_loc' : inputs[0],
              'imain0' : inputs[1]}
    
    for i, ipt in enumerate(inputs[2:]):
      inputs['isec'+str(i)] = ipt
    
    return self.build_outputs(**inputs)
    
  def _build(self):
    """
    Build the LLDSEvolution
    """
    self.build_outputs()
    
#     self.fill_oslot_with_tensor(0, samp)
#     self.fill_oslot_with_tensor(1, loc)
#     self.fill_oslot_with_tensor(2, A)
#     self.fill_oslot_with_tensor(3, prec)
#     self.fill_oslot_with_tensor(4, Alinear)

    self._is_built = True
    
  def build_outputs(self, **all_user_inputs):
    """
    Build the outputs of the LLDSEvolution
    """
    print("Building all outputs, ", self.name)
    
    self.build_Alinear()
    self.build_output('A', **all_user_inputs)
    self.build_output('loc', **all_user_inputs)
    self.build_output('prec')
    self.build_output('main', **all_user_inputs)

#     if islot_to_itensor is not None:
#       inputs = self.prepare_inputs(islot_to_itensor=islot_to_itensor)
#     else:
#       inputs = self.prepare_inputs()
#       
#     Alinear, _ = self.build_output('Alinear')
#     A, _ = self.build_output('A', Alinear=Alinear, **inputs)
#     loc, _ = self.build_output('loc', A=A, **inputs)
#     prec, sec_output = self.build_output('prec')
#     scale = sec_output[0]
# 
#     dist = self.build_dist(loc, scale)
#     samp, _ = self.build_output('main', loc=loc, scale=scale, dist=dist)
#     
#     return samp, loc, A, prec, Alinear, dist
#   
  def prepare_inputs(self, **inputs):
    """
    Prepare inputs for building
    """
    islot_to_itensor = self._islot_to_itensor
    
    true_inputs = {'iprior_loc'  : islot_to_itensor[0]['loc'],
                   'imain0' : islot_to_itensor[1]['main']}
    for i, ipt in enumerate(islot_to_itensor[2:]):
      true_inputs['isec'+str(i)] = ipt['main']
    
    if inputs:
      print("\t\tUpdating defaults with", list(inputs.keys()), ",", self.name, )
      true_inputs.update(inputs)
    
    return true_inputs

  def build_output(self, oname, **inputs):
    """
    Build a single output
    """
    if oname == 'Alinear':
      return self.build_Alinear()
    elif oname == 'A':
      return self.build_A(**inputs)
    elif oname == 'loc':
      return self.build_loc(**inputs)
    elif oname == 'prec':
      return self.build_prec()
    elif oname == 'main':
      return self.build_main(**inputs)
    else:
      raise ValueError("")
    
  def concat_inputs(self, **inputs):
    """
    Concat input tensors
    """
    input_list = ([inputs['imain0']] 
                  + [inputs['isec'+str(i)] for i in range(self.num_sec_inputs)])
    main_input = tf.concat(input_list, axis=-1)
    return main_input

  def build_Alinear(self):
    """
    Build the linear component of the dynamics
    
    TODO: Batch size in all these constant tensors is NOT needed! Just return
    Alinear of shape [1, xdim, xdim] and broadcast
    """
    print("\tBuilding Alinear, ", self.name)
    xdim = self.xdim
    scope_suffix = "_Alinear"
    oshape = [xdim, xdim]
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      eye = np.eye(xdim, dtype=np.float64)
      eye = tf.constant_initializer(eye)
      Al = tf.get_variable('Alinear',
                           shape=oshape,
                           dtype=tf.float64,
                           initializer=eye)
    
    self.fill_oslot_with_tensor(4, Al)
#     return Al, ()

  def build_A(self, **inputs):
    """
    """
    return self.build_A_secs(**inputs)[0]

  def build_A_secs(self, **inputs):
    """
    Declare the evolution matrix A
    """
    print("\tBuilding A,", self.name)
    
    dirs = self.directives

    # get directives
    layers  = dirs.A_layers
    numlayers = dirs.A_numlayers
    numnodes = dirs.A_numnodes
    activations = dirs.A_activations
    winitializers = dirs.A_winitializers
    binitializers = dirs.A_binitializers

    inputs = self.prepare_inputs(**inputs)
    _input = self.concat_inputs(**inputs)
    
    xdim = self.xdim
    scope_suffix = "_A"
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      if numlayers == 1:
        layer = layers_dict[layers[0]]
        act = act_fn_dict[activations[0]]
        winit = winitializers[0]
        binit = binitializers[0]
        B = layer(_input, xdim**2,
                  activation_fn=act,
                  weights_initializer=winit,
                  biases_initializer=binit)
      
      else:
        # 1st layer
        layer = layers_dict[layers[0]]
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
          layer = layers_dict[layers[n]]
          act = act_fn_dict[activations[n]]
          winit = winitializers[n]
          binit = binitializers[n]
          hid_layer = fully_connected(hid_layer,
                                      numnodes[n],
                                      activation_fn=act,
                                      weights_initializer=winit,
                                      biases_initializer=binit)
        # nth layer 
        layer = layers_dict[layers[numlayers-1]]
        act = act_fn_dict[activations[numlayers-1]]
        winit = winitializers[numlayers-1]
        binit = binitializers[numlayers-1]
        B = fully_connected(hid_layer, xdim**2,
                            activation_fn=act,
                            weights_initializer=winit,
                            biases_initializer=binit)
      
      # reshape
      input_shape = infer_shape(_input)
      r = len(input_shape)
      B_shape = input_shape[:-1] + [xdim, xdim]
      B = tf.reshape(B, shape=B_shape, name='B')

      Alinear = self.get_output_tensor('Alinear')
      for _ in input_shape[r-2::-1]:
        Alinear = tf.expand_dims(Alinear, axis=-3)

      alpha = dirs.alpha
      A = alpha*B + Alinear # Broadcast Alinear
      
    if not self._is_built:
      self.fill_oslot_with_tensor(2, A)
    
    return A, (B,)

  def build_loc(self, **inputs):
    """
    """
    return self.build_loc_secs(**inputs)[0]
  
  def build_loc_secs(self, **inputs):
    """
    Build the loc
    """
    print("\tBuilding loc,", self.name)
    if not inputs:
      A = self.get_output_tensor('A')
      prior_loc = self.get_input_tensor(0, 'loc')
      imain = self.get_input_tensor(1, 'main')
#       imain = inputs['iseq_main']
    else:
      A = self.build_A(**inputs)
      inputs = self.prepare_inputs(**inputs)
      prior_loc = inputs['iprior_loc']
      imain = inputs['imain0']
    
    scope_suffix = "_loc"
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      imain = tf.expand_dims(imain, axis=-2)
      loc = tf.matmul(imain, A) # in this order A
      loc = tf.squeeze(loc, axis=-2)
      
      prior_loc = tf.expand_dims(prior_loc, axis=1)
      loc = tf.concat([prior_loc, loc], axis=1)
      loc = loc[:,:-1]
      
    if not self._is_built:
      self.fill_oslot_with_tensor(1, loc)

    return loc, (A,)

  def build_prec(self):
    """
    Declare the prec (inverse covariance)
    """
    print("\tBuilding prec, scale, ", self.name)
    xdim = self.xdim
    oshape = [xdim, xdim]
    with tf.variable_scope(self.name + "_prec", reuse=tf.AUTO_REUSE):
      eye = np.eye(xdim, dtype=np.float64)
      eye_init = tf.constant_initializer(eye)
      eye_init = tf.get_variable('eye_init',
                                       shape=oshape,
                                       dtype=tf.float64,
                                       initializer=eye_init)
      invscale = tf.linalg.band_part(eye_init, -1, 0)
      prec = tf.matmul(invscale, invscale, transpose_b=True)

      scale = tf.matrix_inverse(invscale) # uses that inverse of LT is LT

    self.fill_oslot_with_tensor(3, prec)
    self.fill_oslot_with_tensor(5, scale)

#   def build_prec(self):
#     """
#     Declare the prec (inverse covariance)
#     """
#     print("\tBuilding prec,", self.name)
#     xdim = self.xdim
#     oshape = [xdim, xdim]
#     with tf.variable_scope(self.name + "_precision", reuse=tf.AUTO_REUSE):
#       eye = np.eye(xdim, dtype=np.float64)
#       eye = tf.constant_initializer(eye)
#       eye_init = tf.get_variable('invscale',
#                                  shape=oshape,
#                                  dtype=tf.float64,
#                                  initializer=eye)
#       invscale = tf.linalg.band_part(eye_init, -1, 0)
#       prec = tf.matmul(invscale, invscale, transpose_b=True)
# 
#       scale = tf.matrix_inverse(invscale) # uses that inverse of lower triangular is lt
#       scale = tf.linalg.LinearOperatorLowerTriangular(scale)
# 
#     return prec, (scale,)
  
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
    scale = self.get_output_tensor('scale')
    scale = tf.linalg.LinearOperatorLowerTriangular(scale)
    if inputs:
      loc = self.build_loc(**inputs)
      dist = MultivariateNormalLinearOperator(loc=loc, scale=scale)
      samp = dist.sample() # Add parameters here
    else:
      loc = self.get_output_tensor('loc')
      self.dist = MultivariateNormalLinearOperator(loc=loc, scale=scale)
      samp = self.sample() # Add parameters here
      self.fill_oslot_with_tensor(0, samp)
    
    return samp, (loc,)

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
    
  def logprob(self, X):
    """
    Build the loglikelihood given an input tensor
    
    Args:
        X (tf.Tensor) :
    """
    return self.build_logprob(imain0=X)
    
  def build_logprob(self, name=None, **inputs):
    """
    Build the loglikelihood given a dictionary of user inputs
    """
    return self.build_logprob_secs(name=name, **inputs)[0]
    
  def build_logprob_secs(self, name=None, **inputs):
    """
    Define the loglikelihood for this node.
    """
    print("Building logprob, ", self.name)

    with tf.variable_scope(self.builder.scope, reuse=tf.AUTO_REUSE):
      if not inputs:
        X = self.get_input_tensor(1, 'main')
        A = self.get_output_tensor('A')
        loc = self.get_output_tensor('loc')
      else:
        inputs = self.prepare_inputs(**inputs)
        X = inputs['imain0']
        A = self.build_A(**inputs)
        loc = self.build_loc(**inputs)
        
      loc_sh = infer_shape(loc)
      N, T = loc_sh[0], loc_sh[1]
      l5 = -0.5*np.log(2*np.pi)*tf.cast(N*T*self.xdim, tf.float64)
  
      prior_sc = self._islot_to_itensor[0]['scale']
      invscale = tf.matrix_inverse(prior_sc)
      invQ0 = tf.matmul(invscale, invscale, transpose_b=True)
      l3 = 0.5*(tf.log(tf.matrix_determinant(invQ0))*tf.cast(N, tf.float64))

      deltax0 = X[:,0] - loc[:,0]
      l1 = -0.5*tf.reduce_sum(deltax0*tf.matmul(deltax0, invQ0))
  
      invQ = self.get_output_tensor('prec')
      l4 = 0.5*(tf.log(tf.matrix_determinant(invQ))*tf.cast(N*(T-1), tf.float64))
  
      loc = tf.expand_dims(loc, axis=-2)
      X = tf.expand_dims(X, axis=-2)
      r = len(loc_sh)
      for _ in loc_sh[r-2::-1]:
        invQ = tf.expand_dims(invQ, axis=-3)
      invQ = tf.tile(invQ, loc_sh[:-1] + [1, 1])
      deltax = X[:,1:] - loc[:,1:]
      l2 = -0.5*tf.reduce_sum(deltax*tf.matmul(deltax, invQ[:,1:]))
      
      logprob = l1 + l2 + l3 + l4 + l5
      
    if name is None:
      name = self.name + ':logprob'
    else:
      name = self.name + ':' + name
    
    if name in self.builder.otensor_names:
      raise ValueError("name {} has already been defined, pass a different"
                       "argument `name`".format(name))
    self.builder.add_to_output_names(name, logprob)
      
    return logprob, (A, loc)
