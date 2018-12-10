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

import tensorflow as tf

from neurolib.encoder.basic import InnerNode
from neurolib.encoder import cell_dict
from neurolib.encoder.input import NormalInputNode
from neurolib.encoder.seq_cells import CustomCell

# pylint: disable=bad-indentation, no-member, protected-access

class EvolutionSequence(InnerNode):
  """
  A sequential InnerNode with Markovian internal dynamics. 
  
  An EvolutionSequence is an InnerNode representing internally a sequence of
  mappings, each mapping taking the output of their predecessor as input. This makes
  them appropriate to represent the evolution, possibly in time, of information, .
  
  RNNs are children of EvolutionSequence.
  """
  num_expected_outputs = 1
  def __init__(self,
               builder,
               state_sizes,
               num_inputs=2,
               name=None,
               mode='forward'):
    """
    Initialize an EvolutionSequence
    """
    super(EvolutionSequence, self).__init__(builder,
                                            is_sequence=True)
    self.name = 'EvSeq_' + str(self.label) if name is None else name    

    self.main_output_sizes = self.get_output_sizes(state_sizes)
    self.main_oshapes = self.get_main_oshapes()
    self._oslot_to_shape[0] = self.main_oshapes[0]
    
    self.num_expected_inputs = num_inputs
    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))

    self.mode = mode
    
  @abstractmethod
  def _build(self):
    """
    """
    raise NotImplementedError("Please implement me.")


class BasicRNNEvolutionSequence(EvolutionSequence):
  """
  Define an Evolution Sequence with a single latent state and output.
  
  BasicRNNEvolutionSequence is the simplest possible EvolutionSequence. It is an
  evolution sequence characterized by a single latent state. In particular this
  implies that only one initial state needs to be provided.
  """
  def __init__(self,
               builder,
               state_sizes,
               num_inputs=2,
               name=None,
               cell_class='basic',
               mode='forward',
               **dirs):
    """
    Initialize the BasicRNNEvolutionSequence
    """
    super(BasicRNNEvolutionSequence, self).__init__(builder,
                                                    state_sizes,
                                                    num_inputs=num_inputs,
                                                    name=name,
                                                    mode=mode)
    
    # Get cell_class and cell
    self.cell_class = cell_class = (cell_dict[cell_class] if isinstance(cell_class, str) 
                                    else cell_class) 
    osize = self.main_output_sizes[0][0]
    if issubclass(cell_class, CustomCell): 
      self.cell = cell_class(state_sizes, builder=self.builder)  #pylint: disable=not-callable
    else:
      self.cell = cell_class(osize)
    
    self._update_default_directives(**dirs)
    
    # Add the init_inode and the init_inode -> ev_seq edge
    self._declare_init_state()

  def _declare_init_state(self):
    """
    Declare the initial state of the Evolution Sequence.
    
    Requires that the BasicRNNEvolutionSequence has only one evolution input
    """
    builder = self.builder
    osize = self.main_output_sizes[0][0]
    try:
      self.init_inode = self.cell.get_init_states(ext_builder=builder)[0]
      self.init_inode = builder.nodes[self.init_inode]
      builder.addDirectedLink(self.init_inode, self, islot=0)
    except AttributeError:
      self.init_inode = builder.addInput(osize, iclass=NormalInputNode)
      builder.addDirectedLink(self.init_inode, self, islot=0)
      self.init_inode = builder.nodes[self.init_inode]
    
  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {}
    self.directives.update(dirs)
        
  def _build(self):
    """
    Build the Evolution Sequence
    """
    cell = self.cell
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      init_state = self.init_inode()

      sorted_inputs = sorted(self._islot_to_itensor.items())
#       print(sorted_inputs)
      inputs_series = tuple(zip(*sorted_inputs[1:]))[1]
      if len(inputs_series) == 1:
        inputs_series = inputs_series[0]
      else:
        inputs_series = tf.concat(inputs_series, axis=-1)
        
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_state)
    
    self._oslot_to_otensor[0] = tf.identity(states_series, name=self.name)
    
    self._is_built = True

  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    """
    raise NotImplementedError("")

class LSTMEvolutionSequence(EvolutionSequence):
  """
  """
  def __init__(self,
               builder,
               state_sizes,
#                init_states,
               num_inputs=3,
               name=None,
               cell_class='lstm',
               mode='forward',
               **dirs):
    """
    Initialize the LSTMEvolutionSequence
    """
    super(LSTMEvolutionSequence, self).__init__(builder,
                                                state_sizes,
                                                num_inputs=num_inputs,
                                                name=name,
                                                mode=mode)
    
    self.cell_class = cell_class = (cell_dict[cell_class] if isinstance(cell_class, str) 
                                    else cell_class) 
    self.num_units = self.main_output_sizes[0][0]
    self._update_default_directives(**dirs)
    
    # Add the init_inode and the init_inode -> ev_seq edge
    if issubclass(cell_class, CustomCell):
      self.is_custom = True
      self.cell = cell_class(state_sizes, builder=self.builder)  #pylint: disable=not-callable
    else:
      self.is_custom = False
      self.cell = cell_class(self.num_units)
    
    self._declare_init_state()

#     builder.addDirectedLink(self.init_inode, self, islot=0)
#     builder.addDirectedLink(self.init_hidden_state, self, islot=1)
    
  def _declare_init_state(self):
    """
    Declare the initial state of the Evolution Sequence.
    
    Uses that the BasicRNNEvolutionSequence has only one evolution input
    """
    builder = self.builder
    
    if self.is_custom:
      self.init_inodes = self.cell.get_init_states(ext_builder=builder)[0]
      
#     try:
#       self.init_inode = builder.nodes[self.init_inode]
      for islot, init_node in self.init_inodes.items():
        builder.addDirectedLink(init_node, self, islot=islot)
#     except AttributeError:
    else:
      hidden_dim = self.main_output_sizes[1][0]
      init_node0_name = builder.addInput(self.num_units, iclass=NormalInputNode)
      init_node1_name = builder.addInput(hidden_dim, iclass=NormalInputNode)
      builder.addDirectedLink(init_node0_name, self, islot=0)
      builder.addDirectedLink(init_node1_name, self, islot=1)
      
      self.init_nodes = [builder.nodes[init_node0_name],
                         builder.nodes[init_node1_name]]
      
  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {'cell' : 'lstm'}
    self.directives.update(dirs)
    
    self.directives['cell'] = cell_dict[self.directives['cell']]
    
  def _build(self):
    """
    Build the Evolution Sequence
    """
    sorted_inputs = sorted(self._islot_to_itensor.items())
    
    init_state = tf.nn.rnn_cell.LSTMStateTuple(sorted_inputs[0][1], sorted_inputs[1][1])
    
    inputs_series = tuple(zip(*sorted_inputs[2:]))[1]
    if len(inputs_series) == 1:
      inputs_series = inputs_series[0]
    else:
      inputs_series = tf.concat(inputs_series, axis=-1)
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units,
                                          state_is_tuple=True)  #pylint: disable=not-callable
      
      print("inputs_series", inputs_series)
      print("sorted_inputs", sorted_inputs)
      print("state tuple", init_state)
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_state)
    
    self._oslot_to_otensor[0] = tf.identity(states_series, name=self.name)
    
    self._is_built = True

  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    """
    raise NotImplementedError("")

class LinearNoisyDynamicsEvSeq(EvolutionSequence):
  """
  """
  def __init__(self,
               label, 
               num_features,
               init_states,
               num_islots=1,
               max_steps=30,
               batch_size=1,
               name=None,
               builder=None,
               mode='forward',
               **dirs):
    """
    """
    self.init_inode = init_states[0]
    super(LinearNoisyDynamicsEvSeq, self).__init__(label,
                                                   num_features,
                                                   init_states=init_states,
                                                   num_inputs=num_islots,
                                                   max_steps=max_steps,
                                                   batch_size=batch_size,
                                                   name=name,
                                                   builder=builder,
                                                   mode=mode)

    builder.addDirectedLink(self.init_inode, self, islot=0)
    self._update_default_directives(**dirs)
    
  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {}
    self.directives.update(dirs)
        
  def _build(self):
    """
    Build the Evolution Sequence
    """
    sorted_inputs = sorted(self._islot_to_itensor.items())
    
    print("sorted_inputs", sorted_inputs)
    init_state = sorted_inputs[0][1]
    inputs_series = tuple(zip(*sorted_inputs[1:]))[1]
    if len(inputs_series) == 1:
      inputs_series = inputs_series[0]
    else:
      inputs_series = tf.concat(inputs_series, axis=-1)
    print("self.state_size", self.state_size)
    
    rnn_cell = self.directives['cell']
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      cell = rnn_cell(self.state_size)  #pylint: disable=not-callable
      
      print("inputs_series", inputs_series)
      print("init_inode", init_state)
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_state)
      print("states_series", states_series)
    
    self._oslot_to_otensor[0] = tf.identity(states_series, name=self.name)
    
    self._is_built = True
    
  def __call__(self,  inputs=None, islot_to_itensor=None):
    """
    """
    raise NotImplementedError("")
  
class CustomEvolutionSequence():
  """
  """
  pass