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

from neurolib.encoder.basic import InnerNode
from neurolib.encoder.input import NormalInputNode
from neurolib.encoder.seq_cells import CustomCell, NormalTriLCell
from neurolib.utils.utils import basic_concatenation

# pylint: disable=bad-indentation, no-member, protected-access

class RNNEvolutionSequence(InnerNode):
  """
  An RNN based EvolutionSequence that can be either in 'forward' or 'backward'
  mode.
  
  The directives expected by the RNNEvolutionSequence comprise directive for the
  sequence node and directives for its cell. The following directives are expected:
  
  Evolution Sequence directives:  
    
  Cell directives:
    num_outputs : TODO: Rethink, a cell should know how many outputs it has??
  """  
  cell_dict = {'basic' : tf.nn.rnn_cell.BasicRNNCell,
               'lstm' : tf.nn.rnn_cell.BasicLSTMCell}

  def __init__(self,
               builder,
               state_sizes,
               cell_class,
               num_inputs=2,
               name=None,
               name_prefix=None,
               mode='forward',
               **dirs):
    """
    Initialize the RNNEvolutionSequence
    """
    name_prefix = name_prefix or 'RNN'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    self.state_sizes = self.state_sizes_to_list(state_sizes)
    
    if cell_class is None:
      cell_class = 'basic'
    self.cell = self._get_cell_instance(cell_class,
                                        builder,
                                        **dirs)
    dirs = self._pull_cell_directives(**dirs)
    
    self.mode = mode
    
    super(RNNEvolutionSequence, self).__init__(builder,
                                               is_sequence=True,
                                               name_prefix=name_prefix,
                                               **dirs)

    self.num_expected_inputs = num_inputs
    self.num_states = len(self.state_sizes)
    if isinstance(self.cell, CustomCell):
      self.num_expected_outputs = self.cell.num_expected_outputs
    else:
      self.num_expected_outputs = self.num_states
    self.num_input_seqs = self.num_expected_inputs - self.num_states

    self.main_oshapes = self.get_state_full_shapes()
    self.state_ranks = self.get_state_size_ranks()
    for oslot, shape in enumerate(self.main_oshapes):
      self._oslot_to_shape[oslot] = shape
    
    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))

    # Add the init_inode_names and the init_inode_names -> ev_seq edge
    self._declare_init_state()
    print(self.init_inodes)
    
    # Add secondary OutputNodes if CustomCell
    if isinstance(self.cell, CustomCell):
      self._declare_secondary_outputs()
    
    # Add a dummy placeholder for the batch size if it cannot be deduced
    if self.num_input_seqs == 0:
      self.dummy_bsz = tf.placeholder(tf.int32, [self.batch_size],
                                      self.name + '_dummy_bsz')
      inseq_dummy = tf.zeros([self.max_steps], dtype=tf.float64)
      inseq_dummy = tf.tile(inseq_dummy, self.dummy_bsz)
      self.dummy_input_series = tf.reshape(inseq_dummy, [-1, self.max_steps, 1])
      self.builder.dummies[self.dummy_bsz.name] = [self.batch_size]

  def _get_cell_instance(self, 
                         cell_class,
                         builder,
                         **dirs):
    """
    Define the cell object
    """
    def _set_cclass_name(cell_class):
      """
      Set the `cclass_name` attribute for tensorflow RNNs
      """
      if cell_class.__name__ == 'BasicLSTMCell':
        cclass_name = 'lstm'
      elif cell_class.__name__ == 'BasicRNNCell':
        cclass_name = 'basic'
      else:
        cclass_name = cell_class.__name__
      return cclass_name
    
    if isinstance(cell_class, str):
      self.cclass_name = cell_class
      self._check_dims_default_rnns(cell_class)
      self.cell_class = cell_class = self.cell_dict[cell_class]  
    else:
      self.cell_class = cell_class
      self.cclass_name = _set_cclass_name(cell_class)

    if issubclass(cell_class, CustomCell):
      cell = cell_class(builder,
                        self.state_sizes,
                        **dirs)  #pylint: disable=not-callable
    else:
      osize = self.state_sizes[0][0]
      cell = cell_class(osize)
    return cell

  def _pull_cell_directives(self, **dirs):
    """
    Add the cell directives to the Evolution Sequence
    """
    if isinstance(self.cell, CustomCell):
      self.cell.directives.update(dirs)
      return self.cell.directives
    else:
      return dirs
    
  def _update_directives(self, **dirs):
    """
    Update the default directives
    """
    this_node_dirs = {'output_0_name' : 'main'}
    this_node_dirs.update(dirs)
    super(RNNEvolutionSequence, self)._update_directives(**this_node_dirs)
    print("self.directives", self.directives)

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
        
  def _declare_secondary_outputs(self):
    """
    Declare secondary output nodes and link them
    """
    for oslot in self.cell.secondary_outputs:
      desc = self.cell.directives['output_' + str(oslot) + '_name']
      oshape = self.cell.get_oshapes(oslot)[:]
      self._oslot_to_shape[oslot] = oshape.insert(1, self.max_steps)
      o = self.builder.addOutputSequence(name_prefix='Out_' + desc)
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
  
  def get_init_inodes(self):
    """
    """
    return self.init_inodes
  
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
      if self.num_input_seqs > 0:
        inputs_series = basic_concatenation(self._islot_to_itensor,
                                            start_from=self.num_states)
      else:
        inputs_series = self.dummy_input_series 
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_states)
      
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
    
    print("self.directives", self.directives)
    self._is_built = True
    
  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    TODO:
    """
    raise NotImplementedError("")


class NonlinearDynamicswGaussianNoise(RNNEvolutionSequence):
  """
  """
  default_cell_class = NormalTriLCell
  def __init__(self,
               builder,
               state_sizes,
               num_inputs=2,
               cell_class=None,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize the NonlinearDynamicswGaussianNoise EvolutionSequence
    """
    if num_inputs < 2:
      raise ValueError("Provided `num_inputs` ({}) is incompatible with "
                        "evseq class, (`num_inputs >= 2`)".format(num_inputs))
    if cell_class is None:
      cell_class = self.default_cell_class      
    
    name_prefix = name_prefix or 'NLDS_wGnoise'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    super(NonlinearDynamicswGaussianNoise, self).__init__(builder,
                                                          state_sizes,
                                                          num_inputs=num_inputs,
                                                          cell_class=cell_class,
                                                          name_prefix=name_prefix,
                                                          **dirs)    
    self.state_dim = self.state_sizes[0][0]

  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    this_node_dirs = {'output_0_name' : 'main',
                      'output_1_name' : 'loc',
                      'output_2_name' : 'scale'}
    this_node_dirs.update(dirs)
    super(NonlinearDynamicswGaussianNoise, self)._update_default_directives(this_node_dirs)

  def log_prob(self, Y):
    """
    Return the log_probability for the NonlinearDynamicswGaussianNoise
    """
    assert self._is_built, ("`self._is_built == False. "
                            "Accessed method `log_prob` but node is not yet built")
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
    TODO:
    """
    raise NotImplementedError("")

    
class LinearNoisyDynamicsEvSeq(RNNEvolutionSequence):
  """
  """
  def __init__(self,
               label, 
               num_features,
               init_states,
               num_islots=1,
               max_steps=30,
               batch_size=1,
               builder=None,
               mode='forward',
               name=None,
               name_prefix='LDS_EvSeq',
               **dirs):
    """
    """
    self.init_inode_names = init_states[0]
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    super(LinearNoisyDynamicsEvSeq, self).__init__(label,
                                                   num_features,
                                                   init_states=init_states,
                                                   num_inputs=num_islots,
                                                   max_steps=max_steps,
                                                   batch_size=batch_size,
                                                   builder=builder,
                                                   mode=mode,
                                                   name=name,
                                                   name_prefix=name_prefix)

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
