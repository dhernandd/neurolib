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
import tensorflow as tf

from neurolib.encoder.basic import InnerNode
from neurolib.encoder import act_fn_dict, layers_dict
from neurolib.utils.utils import basic_concatenation

# pylint: disable=bad-indentation, no-member

class DeterministicNNNode(InnerNode):
  """
  An InnerNode representing a deterministic transformation in the Model Graph
  (MG).
  
  A DeterministicNNNode is represented as a neural net with a single output.
  This is the simplest node that transforms information from one representation
  to another. DeterministicNNNodes can have an arbitrary number of inputs. If
  `self.num_expected_inputs > 1`, a concatenation utility is called.
  
  Class attributes:
    num_expected_outputs = 1    
  """
  num_expected_outputs = 1
  
  def __init__(self,
               builder,
               state_sizes,
               num_inputs=1,
               is_sequence=False,
               name=None,
               **dirs):
    """
    Initialize a DeterministicNNNode.
    
    Args:
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes

      state_sizes (int or list of ints): The shape of the output encoding.
          This excludes the 0th dimension - batch size - and the 1st dimension
          when the data is a sequence - number of steps
      
      num_inputs (int): The number of inputs to this node
      
      is_sequence (bool): Is the input a sequence?
      
      name (str): A unique string identifier for this node
      
      dirs (dict): A set of user specified directives for constructing this
          node
    """
    super(DeterministicNNNode, self).__init__(builder, is_sequence)
    
    self.name = "Det_" + str(self.label) if name is None else name
    
    self.main_output_sizes = self.get_output_sizes(state_sizes)
    self.num_expected_inputs = num_inputs
    
    # Define main shape
    self.main_oshapes, self.D = self.get_main_oshapes()
    self._oslot_to_shape[0] = self.main_oshapes
    
    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))

    self._update_directives(**dirs)
    
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    self.directives = {'num_layers' : 2,
                       'num_nodes' : 128,
                       'activation' : 'relu',
                       'net_grow_rate' : 1.0}
    self.directives.update(dirs)
  
  def _get_output(self, inputs=None, islot_to_itensor=None):
    """
    """
    if inputs is not None:
      _input = inputs
    elif islot_to_itensor is not None:
      _input = islot_to_itensor
    else:
      _input = self._islot_to_itensor
    _input = basic_concatenation(_input)
#     if inputs is not None:
#       _input = basic_concatenation(inputs)
#     elif islot_to_itensor is not None:
#       _input = basic_concatenation(islot_to_itensor)
#     else:
#       _input = basic_concatenation(self._islot_to_itensor)

    dirs = self.directives
    
    # Get directives
    try:
      if 'layers' in dirs:
        num_layers = len(dirs['layers'])
        layers = [layers_dict[dirs['layers'][i]] for i in range(num_layers)]
      else:
        num_layers = dirs['num_layers']
        layers = [layers_dict['full'] for i in range(num_layers)]
      if 'activations' in dirs:
        activations = [act_fn_dict[dirs['activations'][i]] 
                                      for i in range(num_layers)]
      else:
        activation = dirs['activation']
        activations = [act_fn_dict[activation]
                                      for i in range(num_layers-1)]
        activations.append(None)
      num_nodes = dirs['num_nodes']
      net_grow_rate = dirs['net_grow_rate']
    except AttributeError as err:
      raise err
    
    # Build
#     if islot_to_itensor is None:
#       islot_to_itensor = self._islot_to_itensor
#     itensors = list(zip(*sorted(islot_to_itensor.items())))[1] # make sure the inputs are ordered
#     _input = tf.concat(itensors, axis=-1)
#     output_dim = self._oslot_to_shape[0][-1]
    output_dim = self.main_output_sizes[0][0]
    if num_layers == 1:
      output = layers[0](_input, output_dim, activation_fn=activations[0])
    else:
      for n, layer in enumerate(layers):
        if n == 0:
          hid_layer = layer(_input, num_nodes, activation_fn=activations[n])
        elif n == num_layers-1:
          output = layer(hid_layer, output_dim, activation_fn=activations[n])
        else:
          hid_layer = layer(hid_layer, int(num_nodes*net_grow_rate),
                            activation_fn=activations[n])
    return output
  
  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    Call the DeterministicNNNode.
    """
    return self._get_output(inputs, islot_to_itensor)

  def _build(self):
    """
    Build the DeterministicNNNode
    """
    output = self._get_output(islot_to_itensor=self._islot_to_itensor)
    
    output_name = self.name + '_out'
    self._oslot_to_otensor[0] = tf.identity(output, output_name) 
      
    self._is_built = True
    
    return output