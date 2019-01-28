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

from neurolib.encoder.basic import InnerNode
from neurolib.encoder import MultivariateNormalTriL, MultivariateNormalFullCovariance  # @UnresolvedImport
from neurolib.utils.utils import basic_concatenation

act_fn_dict = {'relu' : tf.nn.relu,
               'leaky_relu' : tf.nn.leaky_relu,
               'softplus' : tf.nn.softplus}

# pylint: disable=bad-indentation, no-member, protected-access

class NormalNode(InnerNode):
  """
  An abstract class representing Normal Encoders.
  
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
    self.state_sizes = self.state_sizes_to_list(state_sizes)
    
    super(NormalNode, self).__init__(builder,
                                     is_sequence,
                                     name_prefix=name_prefix,
                                     **dirs)
    
    self.num_expected_inputs = num_inputs

    self.main_oshapes = self.get_state_full_shapes()
    self.state_ranks = self.get_state_size_ranks()
    self.xdim = self.state_sizes[0][0]
    self._oslot_to_shape[0] = self.main_oshapes[0]
    
  def _declare_secondary_outputs(self):
    """
    Declare outputs for the statistics of the distribution (mean and standard
    deviation)
    """
    main_oshape = self.state_sizes[0]
    add_name = lambda x : self.name + '_' + x
    
    # Loc oslot
    self._oslot_to_shape[1] = main_oshape
    o1 = self.builder.addOutput(name=add_name(self.directives['output_1_name']))
    self.builder.addDirectedLink(self, o1, oslot=1)
    
    # Stddev oslot
    self._oslot_to_shape[2] = main_oshape + [self.xdim]
    o2 = self.builder.addOutput(name=add_name(self.directives['output_2_name']))
    self.builder.addDirectedLink(self, o2, oslot=2)
        
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
  
  @abstractmethod
  def _build(self):
    """
    """
    raise NotImplementedError("")
      
  def __call__(self, inputs=None):
    """
    Call the node on inputs
    
    Args:
        inputs (tf.Tensor or list of tensors) : 
        
        islot_to_itensor (dict) :
    """
    if inputs is not None:
      islot_to_itensor = (dict(enumerate(list)) if isinstance(inputs, list)
                          else inputs)
      
    return self._build_output(islot_to_itensor)
  
  
class LDSNode(NormalNode):
  """
  An InnerNode that outputs a sample from a normal distribution with input-
  dependent mean and variance.
  
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
    self.dist = None

    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))
    
    self._declare_secondary_outputs()
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'num_layers' : 2,
                      'num_nodes' : 128,
                      'activation' : 'leaky_relu',
                      'net_grow_rate' : 1.0,
                      'share_params' : False,
                      'output_1_name' : 'loc',
                      'output_2_name' : 'invQ',
                      'output_3_name' : 'A'}
    this_node_dirs.update(dirs)
    super(LDSNode, self)._update_directives(**this_node_dirs)
    
    # Deal with directives that map to tensorflow objects hidden from the client
    self.directives['activation'] = act_fn_dict[self.directives['activation']]

  def _declare_secondary_outputs(self):
    """
    Declare outputs for the statistics of the distribution (mean and standard
    deviation)
    """
    # Loc oslot
    self._oslot_to_shape[1] = self.state_sizes[0]
    o1 = self.builder.addOutput(name_prefix='Out_'+self.directives['output_1_name'])
    self.builder.addDirectedLink(self, o1, oslot=1)
     
    # invQ oslot
    self._oslot_to_shape[2] = self.state_sizes[0] + [self.xdim]
    o2 = self.builder.addOutput(name_prefix='Out_'+self.directives['output_2_name'])
    self.builder.addDirectedLink(self, o2, oslot=2)

    # A oslot
    self._oslot_to_shape[3] = self.state_sizes[0] + [self.xdim]
    o3 = self.builder.addOutput(name_prefix='Out_'+self.directives['output_3_name'])
    self.builder.addDirectedLink(self, o3, oslot=3)
    
  def _get_A(self, scope_suffix=None):
    """
    Declare the evolution matrix A
    """
    scope_suffix = ("_A" if scope_suffix is None else "_" 
                    + scope_suffix + "_A")
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      eye = np.expand_dims(np.eye(self.xdim, dtype=np.float64), axis=0)
      eye = tf.constant_initializer(eye)
      A = tf.get_variable('A',
                          shape=[1] + self._oslot_to_shape[3],
                          dtype=tf.float64,
                          initializer=eye)      
    return A
    
  def _get_loc(self, _input, A, scope_suffix=None):
    """
    Declare the loc
    """
    scope_suffix = ("_loc" if scope_suffix is None 
                            else "_" + scope_suffix + "_loc")
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      _input = tf.expand_dims(_input, axis=1)
      loc = tf.matmul(_input, A) # in this order A
      loc = tf.squeeze(loc, axis=1) 

    return loc
  
  def _get_invQ(self, scope_suffix=None):
    """
    Declare the precision (inverse covariance)
    """
    scope_suffix = ("_invQ" if scope_suffix is None 
                            else "_" + scope_suffix + "_invQ")
    with tf.variable_scope(self.name+scope_suffix, reuse=tf.AUTO_REUSE):
      eye = np.expand_dims(np.eye(self.xdim, dtype=np.float64), axis=0)
      eye = tf.constant_initializer(eye)
      invQscale = tf.get_variable('Q',
                                  shape=[1]+self._oslot_to_shape[2],
                                  dtype=tf.float64,
                                  initializer=eye)
      invQ = tf.matmul(invQscale, invQscale, transpose_b=True)
      cov = tf.matrix_inverse(invQ)
    
    return invQ, cov
  
  def _build_output(self, islot_to_itensor=None):
    """
    Get the outputs from the distribution
    
    Args:
        inputs (tf.Tensor or list of tensors or list of list of tensors) : 
        
        islot_to_itensor (dict) :
    """
    _input = basic_concatenation(islot_to_itensor)
    
    A = self._get_A()
    mean = self._get_loc(_input, A)
    _, cov = self._get_invQ()
    
    return MultivariateNormalFullCovariance(loc=mean,
                                            covariance_matrix=cov).sample()
  
  def _build(self):
    """
    Builds the graph corresponding to a NormalTriL encoder.
    
    TODO: Expand this a lot, many more specs necessary.
    """
    islot_to_itensor = self._islot_to_itensor
    _input = basic_concatenation(islot_to_itensor)

    A = self._get_A()
    mean = self._get_loc(_input, A)
    invQ, cov = self._get_invQ()
    self.dist = MultivariateNormalFullCovariance(loc=mean,
                                                 covariance_matrix=cov)
    
    # Fill the oslots
    mean_name = self.directives['output_1_name']
    invQ_name = self.directives['output_2_name']
    self._oslot_to_otensor[0] = self.dist.sample(name='Out' + str(self.label) + '_0')
    self._oslot_to_otensor[1] = tf.identity(mean, name=mean_name)
    self._oslot_to_otensor[2] = tf.identity(invQ, name=invQ_name)
    self._oslot_to_otensor[3] = A
    
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
    name_prefix = name_prefix or 'NormalPrecision'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    super(NormalPrecisionNode, self).__init__(builder,
                                              state_sizes=state_sizes,
                                              num_inputs=num_inputs,
                                              is_sequence=is_sequence,
                                              name_prefix=name_prefix,
                                              **dirs)
    self.dist = None

    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))

    self._declare_secondary_outputs()
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'num_layers' : 2,
                      'num_nodes' : 128,
                      'activation' : 'softplus',
                      'net_grow_rate' : 1.0,
                      'share_params' : False,
                      'output_1_name' : 'loc',
                      'output_2_name' : 'precision',
                      'with_constant_precision' : False,
                      'range_locNN_w' : 1.0
                      }
    this_node_dirs.update(dirs)
    super(NormalPrecisionNode, self)._update_directives(**this_node_dirs)
    
    # Deal with directives that map to tensorflow objects hidden from the client
    self.directives['activation'] = act_fn_dict[self.directives['activation']]

  def _declare_secondary_outputs(self):
    """
    Declare outputs for the statistics of the distribution (mean and standard
    deviation)
    
    TODo: Can also be done in the parent class!
    """
    main_oshape = self.state_sizes[0]
    osize = main_oshape[0]
    
    # Loc oslot
    self._oslot_to_shape[1] = main_oshape
    o1 = self.builder.addOutput(name_prefix='Out_'+self.directives['output_1_name'])
    self.builder.addDirectedLink(self, o1, oslot=1)
    
    # Precision oslot
    self._oslot_to_shape[2] = main_oshape + [osize]
    o2 = self.builder.addOutput(name_prefix='Out_'+self.directives['output_2_name'])
    self.builder.addDirectedLink(self, o2, oslot=2)
    
  def _get_loc(self, _input, scope_suffix=None):
    """
    Get the loc of the distribution.
    
    Args:
      _input (tf.Tensor): The inputs to the node
    """
    dirs = self.directives
    num_layers = dirs['num_layers']
    num_nodes = dirs['num_nodes']
    activation = dirs['activation']
    net_grow_rate = dirs['net_grow_rate']
    range_locNN_w = dirs['range_locNN_w']

    scope_suffix = ("_loc" if scope_suffix is None 
                            else "_" + scope_suffix + "_loc")
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      # Define the Means
      hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes))
      hid_init_w = tf.random_normal_initializer(stddev=range_locNN_w)
      hid_layer = fully_connected(_input,
                                  num_nodes,
                                  activation_fn=activation,
                                  weights_initializer=hid_init_w,
                                  biases_initializer=hid_init_b)
      for _ in range(num_layers-1):
        num_nodes = int(num_nodes*net_grow_rate)
        hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes))
        hid_init_w = tf.random_normal_initializer(stddev=range_locNN_w)
        hid_layer = fully_connected(hid_layer,
                                    num_nodes,
                                    activation_fn=activation,
                                    weights_initializer=hid_init_w,
                                    biases_initializer=hid_init_b)
      loc = fully_connected(hid_layer,
                            self.xdim,
                            activation_fn=None)
    
    return loc, hid_layer
  
  def _get_lmbda(self, _input, hid_layer=None, scope_suffix=None):
    """
    Get the Cholesky decomposition of the variance
    
    Args:
        _input (tf.Tensor) : The inputs to the node
        
        hid_layer (tf.Tensor) : If dirs['share_params'] == True, then 
    """
    dirs = self.directives
    num_layers = dirs['num_layers']
    num_nodes = dirs['num_nodes']
    activation = dirs['activation']
    net_grow_rate = dirs['net_grow_rate']
    range_locNN_w = dirs['range_locNN_w']
    with_constant_precision = dirs['with_constant_precision']
    share_params = dirs['share_params']

    scope_suffix = ("_precision" if scope_suffix is None 
                              else "_" + scope_suffix + "_precision")
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      if with_constant_precision:
        lmbda_chol_init = tf.cast(tf.eye(self.xdim), tf.float64)
        lmbda_chol = tf.get_variable('lmbda_chol',
                                     initializer=lmbda_chol_init)
        lmbda = tf.matmul(lmbda_chol, lmbda_chol, transpose_b=True)
      else:
        if share_params:
          lmbda_chol = fully_connected(hid_layer,
                                       self.xdim**2,
                                       activation_fn=None)
        else:
          hid_init_w = tf.random_normal_initializer(stddev=range_locNN_w)
          hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes))
          hid_layer = fully_connected(_input,
                                      num_nodes,
                                      activation_fn=activation,
                                      weights_initializer=hid_init_w,
                                      biases_initializer=hid_init_b)
          for _ in range(num_layers-1):
            hid_init_w = tf.random_normal_initializer(stddev=range_locNN_w)
            hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes))
            num_nodes = int(num_nodes*net_grow_rate)
            hid_layer = fully_connected(hid_layer,
                                        num_nodes,
                                        activation_fn=activation,
                                        weights_initializer=hid_init_w,
                                        biases_initializer=hid_init_b)
          hid_init_w = tf.orthogonal_initializer(gain=1.0)
          hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(self.xdim**2))
          lmbda_chol = fully_connected(hid_layer,
                                       self.xdim**2,
                                       activation_fn=None,
                                       weights_initializer=hid_init_w,
                                       biases_initializer=hid_init_b)
        lmbda_chol = tf.reshape(lmbda_chol,
                                shape=[-1, self.max_steps, self.xdim, self.xdim])
        lmbda = tf.matmul(lmbda_chol, lmbda_chol, transpose_b=True)
      
      cov = tf.matrix_inverse(lmbda)
      
    return lmbda, cov

  def _build_output(self, islot_to_itensor=None):
    """
    Get the outputs from the distribution
    
    Args:
        inputs (tf.Tensor or list of tensors or list of list of tensors) : 
        
        islot_to_itensor (dict) :
    """
    _input = basic_concatenation(islot_to_itensor)

    mean, hid_layer = self._get_loc(_input)
    _, cov = self._get_lmbda(_input, hid_layer)
    
    return MultivariateNormalFullCovariance(loc=mean,
                                            covariance_matrix=cov).sample()

  def _build(self):
    """
    Builds the graph corresponding to a NormalPrecision node.
    
    TODO: Expand this a lot, many more specs necessary.
    """
    islot_to_itensor = self._islot_to_itensor
    _input = basic_concatenation(islot_to_itensor)

    mean, hid_layer = self._get_loc(_input)
    lmbda, cov = self._get_lmbda(_input, hid_layer)
    self.dist = MultivariateNormalFullCovariance(loc=mean,
                                                 covariance_matrix=cov)
    
    # Fill the oslots
    mean_name = self.directives['output_1_name']
    lmbda_name = self.directives['output_2_name']
    self._oslot_to_otensor[0] = self.dist.sample(name='Out' + 
                                                 str(self.label) + '_0')
    self._oslot_to_otensor[1] = tf.identity(mean, name=mean_name)
    self._oslot_to_otensor[2] = tf.identity(lmbda, name=lmbda_name)
    
    self._is_built = True
    

class NormalTriLNode(NormalNode):
  """
  An InnerNode that takes an input and outputs a sample from a normal
  distribution, with input-dependent mean and variance, with the variance
  specified via a lower triangular scale matrix.
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
    name_prefix = name_prefix or 'NormalTriL'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    super(NormalTriLNode, self).__init__(builder,
                                         state_sizes=state_sizes,
                                         num_inputs=num_inputs,
                                         is_sequence=is_sequence,
                                         name_prefix=name_prefix,
                                         **dirs)
    self.dist = None

    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))
    
    self._declare_secondary_outputs()
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'num_layers' : 2,
                      'num_nodes' : 128,
                      'activation' : 'leaky_relu',
                      'net_grow_rate' : 1.0,
                      'share_params' : False,
                      'output_0_name' : 'main',
                      'output_1_name' : 'loc',
                      'output_2_name' : 'scale',
                      'range_locNN_w' : 1.0e-5,
                      'range_scaleNN_w' : 1.0}
    this_node_dirs.update(dirs)
    super(NormalTriLNode, self)._update_directives(**this_node_dirs)
    
    # Deal with directives that map to tensorflow objects hidden from the client
    self.directives['activation'] = act_fn_dict[self.directives['activation']]
    
  def _get_loc(self, _input, scope_suffix=None):
    """
    Get the mean of the distribution.
    
    Args:
      _input (tf.Tensor): The inputs to the node
    """
    dirs = self.directives
    num_layers = dirs['num_layers']
    num_nodes = dirs['num_nodes']
    activation = dirs['activation']
    net_grow_rate = dirs['net_grow_rate']
    scope_suffix = ("_mean" if scope_suffix is None 
                            else "_" + scope_suffix + "_mean")
    range_locNN_w = dirs['range_locNN_w']

    output_dim = self.state_sizes[0][-1]
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      # Define the Means
      hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes))
      hid_init_w = tf.random_normal_initializer(stddev=range_locNN_w)
      hid_layer = fully_connected(_input,
                                  num_nodes,
                                  activation_fn=activation,
#                                   weights_initializer=hid_init_w,
                                  biases_initializer=hid_init_b)
      for _ in range(num_layers-1):
        num_nodes = int(num_nodes*net_grow_rate)
        hid_init_b = tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes))
        hid_init_w = tf.random_normal_initializer(stddev=range_locNN_w)
        hid_layer = fully_connected(hid_layer,
                                    num_nodes,
                                    activation_fn=activation,
                                    weights_initializer=hid_init_w,
                                    biases_initializer=hid_init_b)
      mean = fully_connected(hid_layer,
                             output_dim,
                             activation_fn=None)
    
    return mean, hid_layer
  
  def _get_scale_tril(self, _input, hid_layer=None, scope_suffix=None):
    """
    Get the Cholesky decomposition of the variance
    
    Args:
        _input (tf.Tensor) : The inputs to the node
        
        hid_layer (tf.Tensor) : If dirs['share_params'] == True, then 
    """
    dirs = self.directives
    num_layers = dirs['num_layers']
    num_nodes = dirs['num_nodes']
    activation = dirs['activation']
    net_grow_rate = dirs['net_grow_rate']
    range_scaleNN_w = dirs['range_scaleNN_w']
    
    scope_suffix = ("_scale" if scope_suffix is None 
                              else "_" + scope_suffix + "_scale")
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      if dirs['share_params']:
        scale = fully_connected(hid_layer,
                                self.xdim**2,
                                activation_fn=None)
      else:
        hid_init_b = tf.random_normal_initializer(stddev=0.1/np.sqrt(num_nodes))
#         hid_init_w = tf.random_normal_initializer(stddev=range_scaleNN_w)
        hid_init_w = tf.orthogonal_initializer(gain=0.01)
        hid_layer = fully_connected(_input,
                                    num_nodes,
                                    activation_fn=activation,
                                    weights_initializer=hid_init_w,
                                    biases_initializer=hid_init_b)
        for _ in range(num_layers-1):
          num_nodes = int(num_nodes*net_grow_rate)
          hid_init_b = tf.random_normal_initializer(stddev=0.1/np.sqrt(num_nodes))
#           hid_init_w = tf.random_normal_initializer(stddev=range_scaleNN_w)
          hid_init_w = tf.orthogonal_initializer(gain=0.001)
          hid_layer = fully_connected(hid_layer,
                                      num_nodes,
                                      activation_fn=activation,
                                      weights_initializer=hid_init_w,
                                      biases_initializer=hid_init_b)
        hid_init_w = tf.orthogonal_initializer(gain=0.00001)
        hid_init_b = tf.random_normal_initializer(stddev=0.1/np.sqrt(num_nodes))
        scale = fully_connected(hid_layer,
                                self.xdim**2,
                                activation_fn=None,
                                weights_initializer=hid_init_w,
                                biases_initializer=hid_init_b)
      sc_shape = ([-1, self.max_steps, self.xdim, self.xdim] if self.is_sequence
                  else [-1, self.xdim, self.xdim])
      scale = tf.reshape(scale, shape=sc_shape)
      scale = tf.matrix_band_part(scale, -1, 0) # Select the lower triangular piece
      
    return scale
  
  def _build_output(self, islot_to_itensor=None):
    """
    Get the outputs from the distribution
    
    Args:
        inputs (tf.Tensor or list of tensors or list of list of tensors) : 
        
        islot_to_itensor (dict) :
    """
    _input = basic_concatenation(islot_to_itensor)

    mean, hid_layer = self._get_loc(_input)
    scale = self._get_scale_tril(_input, hid_layer)
    
    return MultivariateNormalTriL(loc=mean, scale_tril=scale).sample()
     
  def _build(self):
    """
    Builds the graph corresponding to a NormalTriL encoder.
    
    TODO: Expand this a lot, many more specs necessary.
    """
    islot_to_itensor = self._islot_to_itensor
    _input = basic_concatenation(islot_to_itensor)

    mean, hid_layer = self._get_loc(_input)
    scale = self._get_scale_tril(_input, hid_layer)
    self.dist = MultivariateNormalTriL(loc=mean, scale_tril=scale)
    samp = self.dist.sample(name='Out' + str(self.label) + '_0')

    # Fill the oslots
    o0_name = self.name + '_' + self.directives['output_0_name']
    self._oslot_to_otensor[0] = tf.identity(samp, name=o0_name)
    o1_name = self.name + '_' + self.directives['output_1_name']
    self._oslot_to_otensor[1] = tf.identity(mean, name=o1_name)
    o2_name = self.name + '_' + self.directives['output_2_name']
    self._oslot_to_otensor[2] = tf.identity(scale, name=o2_name)
    
    self._is_built = True
    
  
class LocallyLinearNormalNode(NormalNode):
  """
  """
  @abstractmethod
  def _build(self):
    """
    """
    NormalNode._build(self)
  