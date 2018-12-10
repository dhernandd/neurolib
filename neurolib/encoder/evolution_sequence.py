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
from neurolib.encoder.input import NormalInputNode
from neurolib.encoder.seq_cells import CustomCell
from neurolib.utils.utils import basic_concatenation

# pylint: disable=bad-indentation, no-member, protected-access

class EvolutionSequence(InnerNode):
  """
  A sequential InnerNode with Markovian internal dynamics. 
  
  An EvolutionSequence is an InnerNode representing internally a sequence of
  mappings, each mapping taking the output of their predecessor as input. This makes
  them appropriate to represent the evolution, possibly in time, of information, .
  
  RNNs are children of EvolutionSequence.
  """
  cell_dict = {'basic' : tf.nn.rnn_cell.BasicRNNCell,
               'lstm' : tf.nn.rnn_cell.BasicLSTMCell}

  def __init__(self,
               builder,
               state_sizes,
               num_inputs=2,
               num_outputs=1,
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
    
    self.num_expected_outputs = num_outputs
    self.num_expected_inputs = num_inputs
    self.num_states = len(self.main_output_sizes)
    self.num_input_seqs = self.num_expected_inputs - self.num_states
    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))

    self.mode = mode
    
  @abstractmethod
  def _build(self):
    """
    Build the EvolutionSequence
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
               num_outputs=1,
               name=None,
               cell_class='basic',
               **dirs):
    """
    Initialize the BasicRNNEvolutionSequence
    """
    super(BasicRNNEvolutionSequence, self).__init__(builder,
                                                    state_sizes,
                                                    num_inputs=num_inputs,
                                                    num_outputs=num_outputs,
                                                    name=name,
                                                    mode='forward')
    # Get cell_class and cell
    if isinstance(cell_class, str):
      self.cclass = cell_class
      self._check_dims_default_rnns(cell_class)
      self.cell_class = cell_class = self.cell_dict[cell_class]  
    else:
      self.cell_class = cell_class
    
    if issubclass(cell_class, CustomCell): 
      self.cell = cell_class(state_sizes, builder=self.builder)  #pylint: disable=not-callable
    else:
      osize = self.main_output_sizes[0][0]
      self.cell = cell_class(osize)
    
    self._update_default_directives(**dirs)
    
    # Add the init_inode_names and the init_inode_names -> ev_seq edge
    self._declare_init_state()

  def _check_dims_default_rnns(self, cclass):
    """
    """
    if cclass == 'basic':
      pass
    if cclass == 'lstm':
      if len(self.main_output_sizes) == 1:
        self.main_output_sizes = self.main_output_sizes*2
      if len(self.main_output_sizes) == 2:
        if self.main_output_sizes[0] != self.main_output_sizes[1]:
          raise ValueError("`self.main_output_sizes[0] != self.main_output_sizes[1]` "
                           "({}, {})".format(self.main_output_sizes[0],
                                             self.main_output_sizes[1]))
      if len(self.main_output_sizes) > 2:
        raise ValueError("`len(self.main_output_sizes) == 1 or 2` for an LSTM cell")
      
    
  def _declare_init_state(self):
    """
    Declare the initial state of the Evolution Sequence.
    
    Requires that the BasicRNNEvolutionSequence has only one evolution input
    """
    builder = self.builder
    self.init_inodes = []
    try:
      self.init_inode_names = self.cell.get_init_states(ext_builder=builder)[0]
      for islot, name in enumerate(self.init_inode_names):
        init_inode = builder.nodes[name]
        builder.addDirectedLink(init_inode, self, islot=islot)
        self.init_inodes.append(init_inode)
    except AttributeError:
      for islot, osize in enumerate(self.main_output_sizes):
        init_inode_name = builder.addInput(osize[0], iclass=NormalInputNode)
        builder.addDirectedLink(init_inode_name, self, islot=islot)
        self.init_inodes.append(builder.nodes[init_inode_name])
    
  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {}
    self.directives.update(dirs)
  
  def get_init_state_tuple(self):
    """
    Get the initial states for this Evolution Sequence
    """
    if len(self.init_inodes) == 1:
      return self.init_inodes[0]()
    init_states = tuple(node() for node in self.init_inodes)
    if self.cclass == 'lstm':
      init_states = tf.nn.rnn_cell.LSTMStateTuple(init_states[0],
                                                  init_states[1])
    return init_states
    
  def _build(self):
    """
    Build the Evolution Sequence
    """
    cell = self.cell
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      init_states = self.get_init_state_tuple()
      inputs_series = basic_concatenation(self._islot_to_itensor,
                                          start_from=self.num_states)
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_states)
    
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
    
    self.cell_class = cell_class = (self.cell_dict[cell_class] if isinstance(cell_class, str) 
                                    else cell_class) 
    self.num_units = self.main_output_sizes[0][0]
    self._update_default_directives(**dirs)
    
    # Add the init_inode_names and the init_inode_names -> ev_seq edge
    if issubclass(cell_class, CustomCell):
      self.is_custom = True
      self.cell = cell_class(state_sizes, builder=self.builder)  #pylint: disable=not-callable
    else:
      self.is_custom = False
      self.cell = cell_class(self.num_units)
    
    self._declare_init_state()

#     builder.addDirectedLink(self.init_inode_names, self, islot=0)
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
#       self.init_inode_names = builder.nodes[self.init_inode_names]
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
    
    self.directives['cell'] = self.cell_dict[self.directives['cell']]
    
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
    self.init_inode_names = init_states[0]
    super(LinearNoisyDynamicsEvSeq, self).__init__(label,
                                                   num_features,
                                                   init_states=init_states,
                                                   num_inputs=num_islots,
                                                   max_steps=max_steps,
                                                   batch_size=batch_size,
                                                   name=name,
                                                   builder=builder,
                                                   mode=mode)

    builder.addDirectedLink(self.init_inode_names, self, islot=0)
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
      print("init_inode_names", init_state)
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