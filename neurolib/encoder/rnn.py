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
from neurolib.encoder.seq_cells import CustomCell, NormalTriLCell
from neurolib.utils.directives import NodeDirectives

# pylint: disable=bad-indentation, no-member, protected-access


class RNNEvolution(InnerNode):
  """
  An RNN-based EvolutionSequence that can be either in 'forward' or 'backward'
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
               main_inputs,
               state_inputs,
               cell_class,
               mode='forward',
               name=None,
               name_prefix=None,
               **dirs):
    """
    Initialize the RNNEvolutionSequence
    
    TODO: Revisit cell initialization. There is something fishy
    """
    # names
    name_prefix = name_prefix or 'RNN'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    # state_sizes
    self.main_inputs = main_inputs
    self.state_inputs = state_inputs
    self.state_sizes = self._pull_in_state_sizes(builder)

    # define cell before call to super for directives
    self.mode = mode
    if cell_class is None: cell_class = 'basic'
    cell_dirs = self.get_cell_directives(**dirs)
    print("RNNEvolution; cell_dirs", cell_dirs)
    self.cell = self._get_cell_instance(cell_class,
                                        builder,
                                        **cell_dirs)
    dirs = self._pull_in_cell_directives(**dirs) # pull in required cell directives
    
    super(RNNEvolution, self).__init__(builder,
                                       is_sequence=True,
                                       name_prefix=name_prefix,
                                       **dirs)

    # inputs
    self.num_main_inputs = len(self.main_inputs)
    self.num_states = len(self.state_inputs)
    self.num_expected_inputs = self.num_main_inputs + self.num_states
    
    # outputs
    if isinstance(self.cell, CustomCell):
      self.num_expected_outputs = self.cell.num_expected_outputs
    else:
      self.num_expected_outputs = self.num_states
      
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
    
  def _pull_in_state_sizes(self, builder):
    """
    TODO: Change to calling the states `state_sizes` attribute instead?  
    """
    return [[builder.nodes[state].xdim] for state in self.state_inputs]
  
  def get_cell_directives(self, **dirs):
    """
    """
    return {'_'.join(d.split('_')[1:]) : dirs[d] for d in dirs if d.startswith('cell')}
    
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

  def _pull_in_cell_directives(self, **dirs):
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
    super(RNNEvolution, self)._update_directives(**this_node_dirs)
    
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
  
  def __call__(self, *inputs):
    """
    TODO:
    """
    raise NotImplementedError

  def _build(self):
    """
    Build the RNN outputs
    """
    self.build_outputs()
    
    self._is_built = True
    
  def build_outputs(self, **inputs):
    """
    Get the Evolution Sequence outputs
    """
    print("Building all outputs, ", self.name)
    
    self.build_outputs_rnn(**inputs)
    
  def build_outputs_rnn(self, **inputs):
    """
    Build the RNN outputs
    """
    cell = self.cell
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      state_inputs =  {key : inputs[key] for key in inputs 
                       if key.startswith('istate')}
      init_states = self.get_init_state_tuple(**state_inputs)

      main_inputs = {key : inputs[key] for key in inputs 
                       if key.startswith('imain')}
      main_inputs = self.prepare_main_inputs(**main_inputs)
      
      # build the tf ops
      if self.num_main_inputs > 0:
        inputs_series = self.concat_main_input_series(**main_inputs)
      else:
        inputs_series = self.dummy_input_series
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_states)

    if self.num_expected_outputs == 1 or self.cclass_name == 'lstm':
      self.fill_oslot_with_tensor(0, states_series)
    else:
      for oslot, state in enumerate(states_series):
        if isinstance(state, list): # hideous, due to inconsistent cell behaviour. Keep for now
          self.fill_oslot_with_tensor(oslot, state[0])
        else:
          self.fill_oslot_with_tensor(oslot, state)

    return states_series
  
  def get_init_state_tuple(self, **inputs):
    """
    Get the initial states for this Evolution Sequence
    
    The initial states must be a tuple of tensors.
    """
    inputs = self.prepare_state_inputs(**inputs)
    if self.num_states == 1:
      return inputs['istate0']
    
    init_states = tuple([inputs['istate'+str(i)] for i in range(self.num_states)])
      
    print("RNNEvolution; self.cclass_name", self.cclass_name)
    # The tensorflow lstm implementation requires special treatment
    if self.cclass_name == 'lstm':
      assert len(init_states) == 2
      init_states = tf.nn.rnn_cell.LSTMStateTuple(init_states[0],
                                                  init_states[1])
      print("RNNEvolution; init_states", init_states)
    return init_states

  def prepare_main_inputs(self, **inputs):
    """
    Prepare inputs for building
    """
    nss, nms = self.num_states, self.num_main_inputs
    islot_to_itensor = self._islot_to_itensor
    true_main_inputs = {'imain'+str(i-nss) : islot_to_itensor[i]['main'] for
                         i in range(nss, nss+nms)}
    
    if inputs:
      print("\t\tUpdating defaults with", list(inputs.keys()), ",", self.name, )
      true_main_inputs.update(inputs)
    
    return true_main_inputs

  def prepare_state_inputs(self, **inputs):
    """
    Prepare inputs for building
    """
    islot_to_itensor = self._islot_to_itensor
    true_state_inputs = {'istate'+str(i) : islot_to_itensor[i]['main'] for
                         i in range(self.num_states)}
    
    if inputs:
      print("\t\tUpdating defaults with", list(inputs.keys()), ",", self.name, )
      true_state_inputs.update(inputs)
    
    return true_state_inputs
  
  def concat_main_input_series(self, **main_inputs):
    """
    Concatenate a list of tensors or a dictionary with items (slot, tensor)
    """
    input_series = [main_inputs['imain'+str(i)] for i in range(self.num_main_inputs)]
    
    return tf.concat(input_series, axis=-1)
  
  
class NormalRNN(RNNEvolution):
  """
  """
  def __init__(self,
               builder,
               main_inputs,
               state_inputs,
               cell_class=NormalTriLCell,
               mode='forward',
               name=None,
               name_prefix=None,
               **dirs):
    """
    """
    # names
    name_prefix = name_prefix or 'NormalRNN'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    super(NormalRNN, self).__init__(builder=builder,
                                    main_inputs=main_inputs,
                                    state_inputs=state_inputs,
                                    cell_class=cell_class,
                                    mode=mode,
                                    name=name,
                                    name_prefix=name_prefix,
                                    **dirs)
  
  def entropy(self):
    """
    Return the entropy of the NonlinearDynamicswGaussianNoise
    """
    assert self._is_built, ("`self._is_built == False. "
                            "Method `entropy` can only be accessed once the "
                            "node is built")
    scales = self.get_output_tensor('scale')
    covs = tf.matmul(scales, scales, transpose_b=True)
    return 0.5*tf.reduce_sum(tf.log(tf.linalg.det(covs)), axis=1)

  def __call__(self, *inputs):
    """
    TODO:
    """
    raise NotImplementedError
