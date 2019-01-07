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
               mode='forward'):
    """
    Initialize an EvolutionSequence
    """
    super(EvolutionSequence, self).__init__(builder,
                                            is_sequence=True)

    self.state_sizes = self.state_sizes_to_list(state_sizes)
    self.main_oshapes = self.get_state_full_shapes()
    self.D = self.get_state_size_ranks()
    for oslot, shape in enumerate(self.main_oshapes):
      print("ev seq; oslot, shape", oslot, shape)
      self._oslot_to_shape[oslot] = shape
      self.oslot_to_name[oslot] = 'main_' + str(self.label) + '_' + str(oslot)
    
    self.num_expected_outputs = num_outputs
    self.num_expected_inputs = num_inputs
    self.num_states = len(self.state_sizes)
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
               num_inputs=2,
               num_outputs=1,
               name_prefix=None,
               **dirs):
    """
    Initialize the RNNEvolutionSequence
    """
    super(RNNEvolutionSequence, self).__init__(builder,
                                               state_sizes,
                                               num_inputs=num_inputs,
                                               num_outputs=num_outputs,
                                               mode='forward')
    self.name = ('RNN_' + str(self.label) if name_prefix is None 
                 else name_prefix + '_' + str(self.label))

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
      print("evs; state_sizes", state_sizes)
      print("evs; self.builder", self.builder)
      self.cell = cell_class(state_sizes,
                             builder=self.builder)  #pylint: disable=not-callable
    else:
      osize = self.state_sizes[0][0]
      self.cell = cell_class(osize)
    
    self._update_default_directives(**dirs)
    
    # Add the init_inode_names and the init_inode_names -> ev_seq edge
    self._declare_init_state()
    
    if isinstance(self.cell, CustomCell):
      self._declare_secondary_outputs()
    
    if self.num_input_seqs == 0:
      self.dummy_bsz = tf.placeholder(tf.int32, [self.batch_size],
                                      self.name + '_dummy_bsz')
      inseq_dummy = tf.zeros([self.max_steps], dtype=tf.float64)
      inseq_dummy = tf.tile(inseq_dummy, self.dummy_bsz)
      self.dummy_input_series = tf.reshape(inseq_dummy, [-1, self.max_steps, 1])
#       dummy_is = tf.placeholder(dtype=tf.float64,
#                                 shape=[1, self.max_steps, 1],
#                                 name=self.name + '_dummy_is')
      self.builder.dummies[self.dummy_bsz.name] = [self.batch_size]
#       self.dummy_input_series = tf.zeros_like(dummy_is, dtype=tf.float64)

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
      desc = self.cell.directives['out_' + str(oslot) + '_name']
      oname = str(self.label) + '_' + str(oslot) + '_' + desc
      self.oslot_to_name[oslot] = oname
#       oname = self.get_output_tensor_name(oslot)
      self._oslot_to_shape[oslot] = self.cell._oslot_to_shape[oslot][:].insert(1, self.max_steps)
      o = self.builder.addOutputSequence(name="Out_" + str(self.label) + '_' 
                                         + str(oslot) + '_' + desc)
      self.builder.addDirectedLink(self, o, oslot=oslot)
      
  def _check_dims_default_rnns(self, cclass):
    """
    Check the EvolutionSequence dimension attributes for tensorflow RNNs
    """
    if cclass == 'basic':
      pass
    if cclass == 'lstm':
      if len(self.state_sizes) == 1:
        self.state_sizes = self.state_sizes*2
      if len(self.state_sizes) == 2:
        if self.state_sizes[0] != self.state_sizes[1]:
          raise ValueError("`self.state_sizes[0] != self.state_sizes[1]` "
                           "({}, {})".format(self.state_sizes[0],
                                             self.state_sizes[1]))
      if len(self.state_sizes) > 2:
        raise ValueError("`len(self.state_sizes) == 1 or 2` for an LSTM cell")
      
  def _declare_init_state(self):
    """
    Declare the initial states of the EvolutionSequence.
    
    The initial states are not considered to be elements of the
    EvolutionSequence, hence they are defined outside of it. Specifically,
    custom cells do NOT use their internal Builder to build the nodes
    corresponding to their initial state. Instead, they have a method
    `get_init_states` that takes as argument the external EvolutionSequence
    builder and uses it to build its initial states.
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
      for islot, osize in enumerate(self.state_sizes):
        init_inode_name = builder.addInput(osize[0], iclass=NormalInputNode)
        builder.addDirectedLink(init_inode_name, self, islot=islot)
        self.init_inodes.append(builder.nodes[init_inode_name])
  
  def get_init_inodes(self):
    """
    """
    return self.init_inodes
  
  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {'output_0_name' : 'main'}
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
#       print("es; self.num_input_seqs", self.num_input_seqs)
      if self.num_input_seqs > 0:
        inputs_series = basic_concatenation(self._islot_to_itensor,
                                            start_from=self.num_states)
      else:
        inputs_series = self.dummy_input_series 
#       print("evseq; init_states", init_states)
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_states)
#       print("ev_seq; states_series", states_series)
      
    try:
      for oslot, state in enumerate(states_series):
        if 'output_' + str(oslot) + '_name' in self.directives:
          out_name = (self.name + '_' 
                      + self.directives['output_' + str(oslot) + '_name'])
        else:
          out_name = self.name + '_' + str(oslot)
        self._oslot_to_otensor[oslot] = tf.identity(state, name=out_name)
    except TypeError:
      out_name = self.directives['output_0' + '_name']
      self._oslot_to_otensor[0] = tf.identity(states_series, name=out_name)
          
    self._is_built = True
    
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
               name=None,
               **dirs):
    """
    Initialize the NonlinearDynamicswGaussianNoise EvolutionSequence
    """
    ni = dirs.pop('num_inputs', None)
    if ni and ni != 2: 
      raise ValueError("Provided `num_inputs` {} inconsistent with "
                       "evseq class {}".format(ni, 2))
    super(NonlinearDynamicswGaussianNoise, self).__init__(builder,
                                                          state_sizes,
                                                          num_inputs=2,
                                                          num_outputs=3,
                                                          cell_class=NormalTriLCell,
                                                          name_prefix='NLDS_wGnoise',
                                                          **dirs)
    if name is not None: self.name = name # override default
    
    # Slot names
    self.oslot_to_name[1] = 'loc_' + str(self.label) + '_1' 
    self.oslot_to_name[2] = 'scale_' + str(self.label) + '_2'
    
    self.state_dim = self.state_sizes[0][0]

  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {'output_0_name' : 'main',
                       'output_1_name' : 'loc',
                       'output_2_name' : 'scale'}
    self.directives.update(dirs)

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
    covs = tf.matmul(scales, scales, transpose_b=True)
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
