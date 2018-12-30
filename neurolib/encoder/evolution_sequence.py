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

from neurolib.encoder.basic import InnerNode
from neurolib.encoder.input import NormalInputNode
from neurolib.encoder.seq_cells import CustomCell, NormalTriLCell
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
#                name=None,
               mode='forward'):
    """
    Initialize an EvolutionSequence
    """
    super(EvolutionSequence, self).__init__(builder,
                                            is_sequence=True)
#     self.name = 'EvSeq_' + str(self.label) if name is None else name    

    self.main_output_sizes = self.get_output_sizes(state_sizes)
    self.main_oshapes, self.D = self.get_main_oshapes()
    for oslot, shape in enumerate(self.main_oshapes):
      print("oslot, shape", oslot, shape)
      self._oslot_to_shape[oslot] = shape
    
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


class RNNEvolutionSequence(EvolutionSequence):
  """
  Defines an RNN based EvolutionSequence either in 'forward' or 'backward' mode.
  
  RNNEvolutionSequence is the simplest possible EvolutionSequence.
  """
  def __init__(self,
               builder,
               state_sizes,
#                num_inputs=2,
#                num_outputs=1,
#                cell_class='basic',
#                name=None,
               **dirs):
    """
    Initialize the RNNEvolutionSequence
    """
    try:
      num_inputs = dirs.pop('num_inputs')
    except KeyError:
      raise KeyError("Missing argument `num_inputs`")
    num_outputs = dirs.pop('num_outputs', 1)

    super(RNNEvolutionSequence, self).__init__(builder,
                                               state_sizes,
                                               num_inputs=num_inputs,
                                               num_outputs=num_outputs,
#                                                name=name,
                                               mode='forward')
    if not hasattr(self, 'name'):
      self.name = dirs.pop('name', 'RNN_' + str(self.label))

    # Get cell_class and cell
    cell_class = dirs.pop('cell_class', 'basic')
    if isinstance(cell_class, str):
      self.cclass_name = cell_class
      self._check_dims_default_rnns(cell_class)
      self.cell_class = cell_class = self.cell_dict[cell_class]  
    else:
      self.cell_class = cell_class
      self._set_cclass_name()
    
    if issubclass(cell_class, CustomCell): 
      self.cell = cell_class(state_sizes,
                             builder=self.builder)  #pylint: disable=not-callable
      self._declare_secondary_outputs()
    else:
      osize = self.main_output_sizes[0][0]
      self.cell = cell_class(osize)
    
    self._update_default_directives(**dirs)
    
    # Add the init_inode_names and the init_inode_names -> ev_seq edge
    self._declare_init_state()

  def _set_cclass_name(self):
    """
    Set the `cclass_name` attribute for tensorflow RNNs
    """
    if self.cell_class.__name__ == 'BasicLSTMCell':
      self.cclass_name = 'lstm'
    elif self.cell_class.__name__ == 'BasicRNNCell':
      self.cclass_name = 'basic'
    else:
      self.cclass_name = None
  
  def _declare_secondary_outputs(self):
    """
    Declare secondary output nodes and link them
    """
    for oslot in self.cell.secondary_output_slots:
      oname = self.get_output_tensor_name(oslot)
      self._oslot_to_shape[oslot] = self.cell._oslot_to_shape[oslot][:].insert(1, self.max_steps)
      o = self.builder.addOutputSequence(name="Out_" + str(oslot) + '_' + oname)
      self.builder.addDirectedLink(self, o, oslot=oslot)
      
  def _check_dims_default_rnns(self, cclass):
    """
    Check the EvolutionSequence dimension attributes for tensorflow RNNs
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
    Declare the initial states of the EvolutionSequence.
    
    The initial states are not considered to be elements of the
    EvolutionSequence, hence they are defined outside of it. Specifically,
    custom cells do NOT use their internal Builder to build the nodes
    corresponding to their initial state. Instead, they have a method
    `get_init_states` that takes as argument the EvolutionSequence builder and
    uses it to build its initial states.
    """
    builder = self.builder
    self.init_inodes = []
    try:
      self.init_inode_names = self.cell.init_states
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
    self.directives = {'out_' + str(i) + '_name' : self.name + '_' + str(i) 
                       for i in range(self.num_expected_outputs)}
    self.directives.update(dirs)
  
  def get_init_state_tuple(self):
    """
    Get the initial states for this Evolution Sequence
    """
    if len(self.init_inodes) == 1:
      return self.init_inodes[0]()
    
    init_states = tuple(node() for node in self.init_inodes)
    if self.cclass_name == 'lstm':
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
      print("ev_seq; states_series", states_series)
      try:
        for oslot, state in enumerate(states_series):
#           print("ev_seq; oslot, state", oslot, state)
          out_name = self.get_output_tensor_name(oslot)
          self._oslot_to_otensor[oslot] = tf.identity(state, name=out_name)
      except TypeError:
        out_name = self.directives['out_0' + '_name']
        self._oslot_to_otensor[0] = tf.identity(states_series, name=out_name)
          
    self._is_built = True

  def get_output_tensor_name(self, oslot):
    """
    Get the name of the output tensor corresponding to an oslot, based on the
    underlying Cell
    """
    try:
      return self.cell.directives['out_' + str(oslot) + '_name']
    except KeyError:
      return self.directives['out_' + str(oslot) + '_name']
    
  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    TODO:
    """
    raise NotImplementedError("")


class NonlinearDynamicswGaussianNoise(RNNEvolutionSequence):
  """
  """
  def __init__(self,
               builder,
               state_sizes,
               **dirs):
    """
    Initialize the NonlinearDynamicswGaussianNoise EvolutionSequence
    """
    self.name = dirs.pop('name', 'NLDS_wGnoise')
#     print("dirs", dirs)
    super(NonlinearDynamicswGaussianNoise, self).__init__(builder,
                                                          state_sizes,
                                                          num_outputs=3,
                                                          cell_class=NormalTriLCell,
                                                          **dirs)
    self.state_dim = self.main_output_sizes[0][0]
    
    # Slot names
    self._oslot_to_name[1] = 'loc'
    self._oslot_to_name[2] = 'scale'

  def log_prob(self, Y):
    """
    Return the log_probability for the NonlinearDynamicswGaussianNoise
    """
    assert self._is_built, ("`self._is_built == False. "
                            "Method `log_prob` can only be accessed once the "
                            "node is built")
    means = self._oslot_to_otensor[1]
    means, Y = tf.expand_dims(means, -1), tf.expand_dims(Y, -1)
    scales = self._oslot_to_otensor[2]
    covs = tf.matmul(scales, scales, transpose_b=True)
    T = self.max_steps
    D = self.state_dim
    
    t1 = np.log(np.pi)*T*D
    t2 = 0.5*tf.reduce_sum(tf.log(tf.linalg.det(covs)), axis=1)
    t3 = -0.5*tf.reduce_sum(tf.multiply(tf.linalg.triangular_solve(scales, (Y - means)),
                                        tf.linalg.triangular_solve(scales, (Y - means))))
    
    return t1 + t2 + t3
    
  def entropy(self):
    """
    Return the entropy of the NonlinearDynamicswGaussianNoise
    """
    assert self._is_built, ("`self._is_built == False. "
                            "Method `entropy` can only be accessed once the "
                            "node is built")
    scales = self._oslot_to_otensor[2]
#     print(scales)
#     print(tf.linalg.det(scales))
    covs = tf.matmul(scales, scales, transpose_b=True)
#     return 0.5*tf.reduce_sum(tf.log(tf.linalg.det(scales)**2), axis=1)
    return 0.5*tf.reduce_sum(tf.log(tf.linalg.det(covs)), axis=1)
    
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
    
    init_state = sorted_inputs[0][1]
    inputs_series = tuple(zip(*sorted_inputs[1:]))[1]
    if len(inputs_series) == 1:
      inputs_series = inputs_series[0]
    else:
      inputs_series = tf.concat(inputs_series, axis=-1)
    
    rnn_cell = self.directives['cell']
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      cell = rnn_cell(self.state_size)  #pylint: disable=not-callable
      
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_state)
    
    for oslot, states in enumerate(states_series):
      self._oslot_to_otensor[oslot] = tf.identity(states, name=self.name + str(oslot))
    
    self._is_built = True
    
  def __call__(self,  inputs=None, islot_to_itensor=None):
    """
    """
    raise NotImplementedError("")
  
class CustomEvolutionSequence():
  """
  """
  pass
