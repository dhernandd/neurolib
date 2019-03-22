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

from neurolib.encoder.inner import InnerNode
from neurolib.encoder import act_fn_dict, layers_dict
from neurolib.utils.directives import NodeDirectives

# pylint: disable=bad-indentation, no-member

class DeterministicNN(InnerNode):
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
               state_size,
               main_inputs,
               lmbda=None,
               is_sequence=False,
               name=None,
               name_prefix=None,
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
    # set name
    name_prefix = name_prefix or 'Det'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    self.state_sizes = self.state_sizes_to_list(state_size) # before InnerNode__init__ call

    # inputs
    self.main_inputs = self.islot_to_input = main_inputs
    self.num_expected_inputs = len(main_inputs)
    
    super(DeterministicNN, self).__init__(builder,
                                          is_sequence,
                                          name_prefix=name_prefix,
                                          **dirs)

    # directives object
    self.directives = NodeDirectives(self.directives)
    self.oslot_names = self.directives.output_names
    
    # lmbda
    self.lmbda = lmbda
    
    # shapes
    self.oshapes = self._get_all_oshapes()
    self.state_ranks = self.get_state_size_ranks()
    self.xdim = self.oshapes['main'][-1] # set when there is only one state
    
    # init list of free i/o slots
    self._islot_to_itensor = [{} for _ in range(self.num_expected_inputs)]
    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))

  def _update_directives(self, **directives):
    """
    Update the node directives
    """
    this_node_dirs = {'numlayers' : 2,
                      'numnodes' : 128,
                      'activations' : 'relu',
                      'netgrowrate' : 1.0,
                      'islots' : [['main'] for _ in range(self.num_expected_inputs)]}
    this_node_dirs.update(directives)
    super(DeterministicNN, self)._update_directives(**this_node_dirs)
  
  def _get_all_oshapes(self):
    """
    Declare the shapes for every output
    """
    bsz = self.batch_size
    mx_stps = self.max_steps
    const_sh = [bsz, mx_stps] if self.is_sequence else [bsz]
    
    xdim = self.state_sizes[0][0]
    return {self.oslot_names[0] : const_sh + [xdim]}
    
  def __call__(self, *inputs):
    """
    Call the DeterministicNN on a list of inputs.
    """
    if not inputs:
      raise ValueError("Inputs are mandatory for the DeterministicNN")
    inputs = {'imain' + str(i) : inputs[i] for i in range(len(inputs))}
    
    return self.build_main(**inputs)

  def _build(self):
    """
    Build the DeterministicNNNode
    """
    self.build_outputs()

    self._is_built = True 
  
  def build_outputs(self, **inputs):
    """
    Get all the outputs
    """
    print("Building all outputs, ", self.name)
    
    self.build_output('main', **inputs)

  def build_output(self, oname, **inputs):
    """
    Build a single output
    """
    if oname == 'main':
      return self.build_main(**inputs)
    else:
      raise ValueError("`oname` {} is not an output name for "
                       "this node".format(oname))
    
  def get_build_name(self, islot, name=None): #pylint: disable=unused-argument
    """
    """
    dirs = self.directives
    if hasattr(dirs, 'buildnames'):
      raise NotImplementedError("")
    else:
      return 'imain' + str(islot) 
    
  def prepare_inputs(self, **inputs):
    """
    Prepare the inputs
    
    TODO: Use the islots directive to define main_inputs
    
    TODO: Use self.get_build_name everywhere
    """
    islot_to_itensor = self._islot_to_itensor
    
    def get_input_tensor(islot):
      """
      Find and return the right input tensor for this islot.
      
      The procedure is divided into cases:
      
      i) self.islot_to_input[islot] is an int and an input for this islot is
      provided. This can only happen if the node is a Custom Node... FINISH
      
      ii) self.islot_to_input[islot] is not an int and an input for this islot
      is provided
      
      iii) An input for the islot is not provided. In this case try to use the
      default: self._islot_to_itensor[islot].
      """
      _input = self.islot_to_input[islot]
      if isinstance(_input, int) and 'icust' + str(_input) in inputs:
        print('\tUpdating default {} with {}'.format(_input, 'icust' + str(_input)))
        return inputs['icust'+str(_input)]
      elif 'imain' + str(islot) in inputs:
        print('\tUpdating default {}'.format('imain' + str(islot)))
        return inputs['imain'+str(islot)]
      else:
        try:
          return islot_to_itensor[islot]['main']
        except KeyError:
          raise KeyError("Could not obtain an input tensor for islot {}"
                         "for building this Node".format(islot))
      
    main_inputs = {self.get_build_name(i) : get_input_tensor(i) for i
                   in range(self.num_expected_inputs)}
    
    return main_inputs
  
  def build_main(self, **inputs):
    """
    Build output 'main'
    """
    print("\tBuilding main, ", self.name)
    
    # get the directives object
    dirs = self.directives
    
    # Get directives
    layers = dirs.layers
    numlayers = dirs.numlayers
    activations = dirs.activations
    numnodes = dirs.numnodes + [self.xdim]
    winitializers = dirs.winitializers
    binitializers = dirs.binitializers
    
    # build output
    inputs = self.prepare_inputs(**inputs)
    _input = self.concat_inputs(**inputs)
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      if numlayers == 1:
        layer = layers_dict[layers[0]]
        act = act_fn_dict[activations[0]]
        winit = winitializers[0]
        binit = binitializers[0]
        output = layer(_input,
                       self.xdim,
                       activation_fn=act,
                       weights_initializer=winit,
                       biases_initializer=binit)
      else:
        for n, layer in enumerate(layers):
          layer  = layers_dict[layer]
          act = act_fn_dict[activations[n]]
          winit = winitializers[n]
          binit = binitializers[n]
          nnodes = numnodes[n]
          if n == 0:
            hid_layer = layer(_input,
                              nnodes,
                              activation_fn=act,
                              weights_initializer=winit,
                              biases_initializer=binit)
          else:
            hid_layer = layer(hid_layer,
                              nnodes,
                              activation_fn=act,
                              weights_initializer=winit,
                              biases_initializer=binit)
        output = hid_layer
      
      if self.lmbda is not None:
        output = self.lmbda(output)
    
    if not self._is_built:
      self.fill_oslot_with_tensor(0, output)
    
    return output
  
  def concat_inputs(self, **inputs):
    """
    Concat input tensors
    """
    input_list = [inputs['imain'+str(i)] for i in range(self.num_expected_inputs)]
    main_input = tf.concat(input_list, axis=-1)

    return main_input
  