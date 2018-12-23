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
from tensorflow.contrib.layers.python.layers import fully_connected  #pylint: disable=no-name-in-module

from neurolib.encoder.basic import InnerNode
from neurolib.encoder import MultivariateNormalTriL  # @UnresolvedImport
from neurolib.utils.utils import basic_concatenation

act_fn_dict = {'relu' : tf.nn.relu,
               'leaky_relu' : tf.nn.leaky_relu}

# pylint: disable=bad-indentation, no-member, protected-access

class NormalTriLNode(InnerNode):
  """
  An InnerNode that outputs a sample from a normal distribution with input-
  dependent mean and variance.
  """
  num_expected_outputs = 3
  
  def __init__(self,
               builder,
               state_sizes,
               num_inputs=1,
               is_sequence=False,
               name=None,
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
    super(NormalTriLNode, self).__init__(builder,
                                         is_sequence)
    
    self.num_expected_inputs = num_inputs
    self.main_output_sizes = self.get_output_sizes(state_sizes)
    self.main_oshape, self.D = self.get_main_oshapes() 
    self._oslot_to_shape[0] = self.main_oshape[0]
    
    self.name = "NormalTril_" + str(self.label) if name is None else name

    self._update_directives(**dirs)
    
    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))
    
    self._declare_secondary_outputs()
    
    self.dist = None

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    self.directives = {'num_layers' : 2,
                      'num_nodes' : 128,
                      'activation' : 'leaky_relu',
                      'net_grow_rate' : 1.0,
                      'share_params' : False,
                      'output_mean_name' : self.name + '_' + str(1) + '_mean',
                      'output_scale_name' : self.name + '_' + str(2) + '_scale'}
    self.directives.update(dirs)
    
    # Deal with directives that map to tensorflow objects hidden from the client
    self.directives['activation'] = act_fn_dict[self.directives['activation']]
    
  def _declare_secondary_outputs(self):
    """
    Declare outputs for the statistics of the distribution (mean and standard
    deviation)
    """
    main_oshape = self.main_output_sizes[0]
    osize = main_oshape[0]
    
    # Mean oslot
    self._oslot_to_shape[1] = main_oshape
    o1 = self.builder.addOutput(name=self.directives['output_mean_name'])
    self.builder.addDirectedLink(self, o1, oslot=1)
    
    # Stddev oslot
    self._oslot_to_shape[2] = main_oshape + [osize]
    o2 = self.builder.addOutput(name=self.directives['output_scale_name'])
    self.builder.addDirectedLink(self, o2, oslot=2)
    
  def _get_mean(self, _input, scope_suffix=None):
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

    output_dim = self.main_output_sizes[0][-1]
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      # Define the Means
      hid_layer = fully_connected(_input, num_nodes, activation_fn=activation,
            biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
      for _ in range(num_layers-1):
        num_nodes = int(num_nodes*net_grow_rate)
        hid_layer = fully_connected(hid_layer, num_nodes, activation_fn=activation,
            biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
      mean = fully_connected(hid_layer, output_dim, activation_fn=None)
    
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
    scope_suffix = ("_scale" if scope_suffix is None 
                              else "_" + scope_suffix + "_scale")

    output_dim = self.main_output_sizes[0][-1]
    with tf.variable_scope(self.name + scope_suffix, reuse=tf.AUTO_REUSE):
      if dirs['share_params']:
        scale = fully_connected(hid_layer, output_dim**2, activation_fn=None)
      else:
        hid_layer = fully_connected(_input, num_nodes, activation_fn=activation,
            biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
        for _ in range(num_layers-1):
          num_nodes = int(num_nodes*net_grow_rate)
          hid_layer = fully_connected(hid_layer, num_nodes, activation_fn=activation,
            biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
        scale = fully_connected(hid_layer, output_dim**2,
              activation_fn=None,
              weights_initializer=tf.random_normal_initializer(stddev=1e-4),
              biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(output_dim**2)))
      scale = tf.reshape(scale, shape=[-1, output_dim, output_dim])
      scale = tf.matrix_band_part(scale, -1, 0) # Select the lower triangular piece
      
    return scale
  
  def _get_output(self, inputs=None, islot_to_itensor=None):
    """
    Get the outputs from the distribution
    
    Args:
        inputs (tf.Tensor or list of tensors or list of list of tensors) : 
        
        islot_to_itensor (dict) :
    """
    if inputs is not None:
#       print("inputs", inputs)
      if not isinstance(inputs, list):
        _input = inputs
      elif not isinstance(inputs[0], list):
        _input = basic_concatenation(inputs)
      else:
        _input = basic_concatenation(inputs[0])
    else:
      _input = basic_concatenation(islot_to_itensor)

    mean, hid_layer = self._get_mean(_input)
    scale = self._get_scale_tril(_input, hid_layer)
    
    return MultivariateNormalTriL(loc=mean, scale_tril=scale).sample()
      
  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    Call the node on inputs
    
    Args:
        inputs (tf.Tensor or list of tensors) : 
        
        islot_to_itensor (dict) :
    """
    return self._get_output(inputs, islot_to_itensor)
     
  def _build(self):
    """
    Builds the graph corresponding to a NormalTriL encoder.
    
    TODO: Expand this a lot, many more specs necessary.
    """
    islot_to_itensor = self._islot_to_itensor
    _input = basic_concatenation(islot_to_itensor)

    mean, hid_layer = self._get_mean(_input)
    scale = self._get_scale_tril(_input, hid_layer)

    mean_name = self.directives['output_mean_name']
    scale_name = self.directives['output_scale_name']
    
    cholesky_tril = tf.identity(scale, name=scale_name)
    
    # Get the tensorflow distribution for this node
    self.dist = MultivariateNormalTriL(loc=mean, scale_tril=cholesky_tril)

    # Fill the oslots
    self._oslot_to_otensor[0] = self.dist.sample(name='Out' + 
                                                 str(self.label) + '_0')
    self._oslot_to_otensor[1] = tf.identity(mean, name=mean_name)
    self._oslot_to_otensor[2] = cholesky_tril
    
    self._is_built = True
    
  def log_prob(self, ipt):
    """
    Get the loglikelihood of the inputs `ipt` for this distribution
    """
    return self.dist.log_prob(ipt)
  
  def entropy(self, ipt):
    """
    Get the entropy of the inputs `ipt` for this distribution
    """
    return self.dist.log_prob(ipt)
  