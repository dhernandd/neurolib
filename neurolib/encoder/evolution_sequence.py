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

from neurolib.encoder.inner import InnerNode
from neurolib.encoder.input import NormalInputNode
from neurolib.encoder.seq_cells import CustomCell, NormalTriLCell
from neurolib.utils.directives import NodeDirectives

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
    # names
    name_prefix = name_prefix or 'RNN'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    # state sizes
    self.state_sizes = self.state_sizes_to_list(state_sizes)
    self.num_states = len(self.state_sizes)
    
    # define cell before call to super
    self.mode = mode
    if cell_class is None: cell_class = 'basic'
    self.cell = self._get_cell_instance(cell_class,
                                        builder,
                                        **dirs)
    dirs = self._pull_cell_directives(**dirs) # pull in required cell directives
    
    super(RNNEvolutionSequence, self).__init__(builder,
                                               is_sequence=True,
                                               name_prefix=name_prefix,
                                               **dirs)
    # number of inputs/outputs
    self.num_expected_inputs = num_inputs
    if isinstance(self.cell, CustomCell):
      self.num_expected_outputs = self.cell.num_expected_outputs
    else:
      self.num_expected_outputs = self.num_states
    self.num_input_seqs = self.num_expected_inputs - self.num_states

    # directives object
    self.directives = NodeDirectives(self.directives)
    self.oslot_names = self.directives.output_names
    
    # define shapes
    self.main_oshapes = self.get_state_full_shapes()
    self.state_ranks = self.get_state_size_ranks()
      
    # Initialize list of free i/o slots
    self._islot_to_itensor = [{} for _ in range(self.num_expected_inputs)]
    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))

    # Add init_inode_names and the init_inode_names -> ev_seq edge
    self._declare_init_states()
    
    # declare state sequence
    if isinstance(self.cell, CustomCell):
      self._declare_state_sequence()
    
    # Add a dummy placeholder for the batch size if it cannot be deduced
    if self.num_input_seqs == 0:
      self.dummy_bsz = self.builder.dummy_bsz
      inseq_dummy = tf.zeros([self.max_steps], dtype=tf.float64)
      inseq_dummy = tf.tile(inseq_dummy, self.dummy_bsz)
      self.dummy_input_series = tf.reshape(inseq_dummy, [-1, self.max_steps, 1])

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
      print("evseq, builder.nodes", builder.nodes)
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
    this_node_dirs = {}
    this_node_dirs.update(dirs)
    super(RNNEvolutionSequence, self)._update_directives(**this_node_dirs)
    
  def _get_all_oshapes(self):
    """
    Declare the shapes for every node output
    """
    InnerNode._get_all_oshapes(self)

  def _declare_init_states(self):
    """
    Declare the initial states of the EvolutionSequence.
    
    In an EvolutionSequence with N states, they occupy the first N islots. 
    
    NOTE: The initial states are not considered to be elements of the
    EvolutionSequence, hence they are defined outside of it. In particular,
    custom cells do NOT use their internal Builder to build the nodes
    corresponding to their initial state. Instead, they have the method
    `get_init_states` that uses the external EvolutionSequence builder to build
    its initial states.
    """
    builder = self.builder
    try:
      # If cell is Custom, it will have the `init_states` attribute
#       self.init_inode_names = self.cell.init_states
      self.init_inodes = self.cell.init_states
      for islot, name in enumerate(self.init_inode_names):
        init_inode = builder.nodes[name]
        builder.addDirectedLink(init_inode, self, islot=islot)
    except AttributeError:
      # Otherwise, it must be a tensorflow cell
      self.init_inodes = []
      for islot, osize in enumerate(self.state_sizes):
        init_inode_name = builder.addInput(osize[0], iclass=NormalInputNode)
        builder.addDirectedLink(init_inode_name, self, islot=islot)
#         self.init_inodes.append(builder.nodes[init_inode_name])
        self.init_inodes.append(init_inode_name)
    
  def _declare_state_sequence(self):
    """
    Declare state sequence
    """
    for i, state in enumerate(self.state_sizes):
      self.builder.addInputSequence([state], name="StateInSeq"+str(i))
    
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
    Get the init input nodes
    """
    return self.init_inodes
  
  def get_init_state_tuple(self, islot_to_itensor=None):
    """
    Get the initial states for this Evolution Sequence
    
    The initial states must be a tuple of tensors.
    """
    if islot_to_itensor is None:
      islot_to_itensor = self._islot_to_itensor
    if self.num_states == 1:
      return islot_to_itensor[0]['main']
    init_states = tuple([islot_to_itensor[i]['main'] for i in range(self.num_states)])
      
    # The tensorflow lstm implementation requires special treatment
    if self.cclass_name == 'lstm':
      assert len(init_states) == 2
      init_states = tf.nn.rnn_cell.LSTMStateTuple(init_states[0],
                                                  init_states[1])
    return init_states

  def __call__(self, *inputs):
    """
    TODO:
    """
    if not inputs:
      raise ValueError("Inputs are mandatory for the DeterministicNNNode")
    islot_to_itensor = [{'main' : ipt} for ipt in inputs]
    return self.build_outputs(islot_to_itensor)

  @staticmethod
  def concat_input_series(islot_to_itensor, start_from=0):
    """
    Concatenate a list of tensors or a dictionary with items (slot, tensor)
    """
    input_series = [elem['main'] for elem in islot_to_itensor[start_from:]]
    return tf.concat(input_series, axis=-1)

  def build_outputs(self, islot_to_itensor=None):
    """
    Get the Evolution Sequence outputs
    """
    if islot_to_itensor is not None:
      _input = islot_to_itensor
    else:
      _input = self._islot_to_itensor
  
    cell = self.cell
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      init_states = self.get_init_state_tuple(_input)
#       print("evseq; init_states", init_states)
      if self.num_input_seqs > 0:
        inputs_series = self.concat_input_series(_input, start_from=self.num_states)
      else:
        inputs_series = self.dummy_input_series
#       print("evseq; inputs_series", inputs_series)
#       print("evseq; init_states", init_states)
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_states)
#       print("evseq; states_series", states_series)
      
    return states_series
  
  def build_outputs_winit(self):
    """
    """
    i_to_itensor = []
    for i, ss in enumerate(self.state_sizes):
      sdim = ss[0]
      inits = tf.placeholder(tf.float64, [None, sdim], 'initstate'+str(i))
      i_to_itensor.append({'main' : inits})
      
    for i in range(self.num_states, self.num_expected_inputs):
      i_to_itensor.append(self._islot_to_itensor[i])
    
    return self.build_outputs(i_to_itensor)
  
  def _build(self):
    """
    Build the Evolution Sequence
    """
    states_series = self.build_outputs()
    states_series_winit = self.build_outputs_winit()
    print("evseq; states_series_winit", states_series_winit)

    if self.num_expected_outputs == 1 or self.cclass_name == 'lstm':
      name_init = self.oslot_names[0] + '_winit'
      self.fill_oslot_with_tensor(0, states_series)
      self.fill_oslot_with_tensor(1, states_series_winit, name=name_init)
    else:
      for oslot, state in enumerate(states_series):
        name_init = self.oslot_names[oslot] + '_winit'
        if isinstance(state, list): # hideous, due to inconsistent cell behaviour. Keep for now
          self.fill_oslot_with_tensor(oslot, state[0])
          self.fill_oslot_with_tensor(oslot+self.num_states,
                                      states_series_winit[oslot][0],
                                      name=name_init)
        else:
          self.fill_oslot_with_tensor(oslot, state)
          self.fill_oslot_with_tensor(oslot+self.num_states,
                                      states_series_winit[oslot],
                                      name=name_init)
      for oslot in range(len(states_series), self.num_expected_outputs):
        oname = self.oslot_names[oslot]
        tensor, _ = self.cell.build_output(oname) # build_output returns a tuple
        self.fill_oslot_with_tensor(oslot, tensor)
          
    self._is_built = True
    
  def _build_model_outputs(self):
    """
    """
    model_outputs = self.cell.model_outputs
    for item in model_outputs:
      oname = item[0]
      _inputs = self.get_model_inputs(item[1])
      output, _ = self.cell.encoder.build_output(oname, _inputs)
      self.builder.add_to_output_names(item[2], output)
      

class NonlinearDynamicswGaussianNoise(RNNEvolutionSequence):
  """
  """
  def __init__(self,
               builder,
               state_sizes,
               cell_class,
               num_inputs=2,
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize the NonlinearDynamicswGaussianNoise EvolutionSequence
    """
    # names
    name_prefix = name_prefix or 'NLDS_wGnoise'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)

    if num_inputs < 2:
      raise ValueError("Provided `num_inputs` ({}) is incompatible with "
                        "EvolutionSequence class, (`num_inputs >= 2`)".format(num_inputs))
    
    if cell_class is None: cell_class = NormalTriLCell 
    
    super(NonlinearDynamicswGaussianNoise, self).__init__(builder,
                                                          state_sizes,
                                                          cell_class,
                                                          num_inputs=num_inputs,
                                                          name_prefix=name_prefix,
                                                          **dirs)    
    
    # shapes
    self.xdim = self.state_sizes[0][0]

  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    this_node_dirs = {}
    this_node_dirs.update(dirs)
    super(NonlinearDynamicswGaussianNoise, self)._update_default_directives(**this_node_dirs)

  def log_prob(self, Y):
    """
    Return the log_probability for the NonlinearDynamicswGaussianNoise
    """
    assert self._is_built, ("`self._is_built == False. "
                            "Accessed method `log_prob` but node is not yet built")
#     means = self._oslot_to_otensor[1]
#     scales = self._oslot_to_otensor[2]
    means = self.get_output_tensor('loc')
    scales = self.get_output_tensor('scale')

    means, Y = tf.expand_dims(means, -1), tf.expand_dims(Y, -1)
    covs = tf.matmul(scales, scales, transpose_b=True)
    T = self.max_steps
    xdim = self.xdim
    
    t1 = np.log(np.pi)*T*xdim
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
    print("self._oslot_to_otensor", self._oslot_to_otensor)
#     scales = self._oslot_to_otensor[2]
    scales = self.get_output_tensor('scale')
    covs = tf.matmul(scales, scales, transpose_b=True)
    return 0.5*tf.reduce_sum(tf.log(tf.linalg.det(covs)), axis=1)

#     
# class LinearNoisyDynamicsEvSeq(RNNEvolutionSequence):
#   """
#   """
#   def __init__(self,
#                label, 
#                num_features,
#                init_states,
#                num_islots=1,
#                max_steps=30,
#                batch_size=1,
#                builder=None,
#                mode='forward',
#                name=None,
#                name_prefix='LDS_EvSeq',
#                **dirs):
#     """
#     """
#     self.init_inode_names = init_states[0]
#     name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
#     super(LinearNoisyDynamicsEvSeq, self).__init__(label,
#                                                    num_features,
#                                                    init_states=init_states,
#                                                    num_inputs=num_islots,
#                                                    max_steps=max_steps,
#                                                    batch_size=batch_size,
#                                                    builder=builder,
#                                                    mode=mode,
#                                                    name=name,
#                                                    name_prefix=name_prefix)
# 
#     builder.addDirectedLink(self.init_inode_names, self, islot=0)
#     self._update_default_directives(**dirs)
#     
#   def _update_default_directives(self, **dirs):
#     """
#     Update the default directives
#     """
#     self.directives = {}
#     self.directives.update(dirs)
#         
#   def _build(self):
#     """
#     Build the Evolution Sequence
#     """
#     sorted_inputs = sorted(self._islot_to_itensor.items())
#     
#     init_state = sorted_inputs[0][1]
#     inputs_series = tuple(zip(*sorted_inputs[1:]))[1]
#     if len(inputs_series) == 1:
#       inputs_series = inputs_series[0]
#     else:
#       inputs_series = tf.concat(inputs_series, axis=-1)
#     
#     rnn_cell = self.directives['cell']
#     with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
#       cell = rnn_cell(self.state_size)  #pylint: disable=not-callable
#       
#       states_series, _ = tf.nn.dynamic_rnn(cell,
#                                            inputs_series,
#                                            initial_state=init_state)
#     
#     for oslot, states in enumerate(states_series):
#       self._oslot_to_otensor[oslot] = tf.identity(states, name=self.name + str(oslot))
#     
#     self._is_built = True
# 
#   def __call__(self, *inputs):
#     """
#     """
#     raise NotImplementedError("")
#   
#   def build_outputs(self, islot_to_itensor=None):
#     """
#     """
#     raise NotImplementedError("")
#   
