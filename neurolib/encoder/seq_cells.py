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

from neurolib.encoder.input import NormalInputNode
from neurolib.encoder.anode import ANode
from neurolib.builders.static_builder import StaticBuilder
from neurolib.encoder.normal import NormalTriLNode


# pylint: disable=bad-indentation, no-member, protected-access

class CustomCell(ANode, tf.nn.rnn_cell.RNNCell):
  """
  A class for custom cells to be used with sequential models
  """
  def __init__(self,
               num_inputs,
               num_outputs,
               builder):

    """
    Initialize the CustomCell
    """
    super(CustomCell, self).__init__()
    self.ext_builder = builder

    self.builder = StaticBuilder(scope='Cell')

    self.num_expected_inputs = num_inputs
    self.num_expected_outputs = num_outputs
    
  def _get_init_states(self):
    """
    """
    raise NotImplementedError

  @property
  def state_size(self):
    """
    """
    raise NotImplementedError

  @property
  def output_size(self):
    """
    """
    raise NotImplementedError
  
  def compute_output_shape(self, input_shape):
    """
    Compute output shape.
    """
    tf.nn.rnn_cell.RNNCell.compute_output_shape(self, input_shape)


class TwoEncodersCell(CustomCell):
  """
  A CustomCell with 2 inner states, each evolved by means of a deterministic RNN.
  """
  def __init__(self,
               state_sizes,
               builder,
               num_inputs=3,
               num_outputs=2,
               name=None,
               **dirs):
    """
    Initialize the TwoEncodersCell
    
    Args:
        state_sizes (list of list of ints):
        
        builder :
               
        num_inputs :

        num_outputs :
  
        name :
        
        dirs :
    """
    super(TwoEncodersCell, self).__init__(num_inputs=num_inputs,
                                          num_outputs=num_outputs,
                                          builder=builder)
    self.state_dims = tuple(state_sizes)
    self.cname = 'TwoEncCell' if name is None else name

    self._update_directives(**dirs)

    self.encoder = self.declare_cell_encoder()
    
    self.init_states = self._get_init_states()
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    self.directives = {}
    self.directives.update(dirs)
    
  @property
  def state_size(self):
    """
    """
    return self.state_dims
  
  @property
  def output_size(self):
    """
    """
    return self.state_dims
    
  def declare_cell_encoder(self):
    """
    Declare the cell inner encoder
    """
    builder = self.builder
    num_inputs = self.num_expected_inputs
    num_outputs = self.num_expected_outputs
    
    cust = builder.createCustomNode(num_inputs, num_outputs, name="Custom")
    cust_in1 = cust.addInner(self.state_size[0:1], num_inputs=2)
    cust_in2 = cust.addInner(self.state_size[1:2], num_inputs=2)
    cust.addDirectedLink(cust_in1, cust_in2, islot=1)

    cust.declareIslot(islot=0, innernode_name=cust_in1, inode_islot=0)
    cust.declareIslot(islot=1, innernode_name=cust_in2, inode_islot=0)
    cust.declareIslot(islot=2, innernode_name=cust_in1, inode_islot=1)

    cust.declareOslot(oslot=0, innernode_name=cust_in1, inode_oslot=0)
    cust.declareOslot(oslot=1, innernode_name=cust_in2, inode_oslot=0)
    cust.commit()
    
    return cust
    
  def __call__(self, inputs, state,  scope=None):
    """
    Evaluate the cell encoder on a set of inputs    
    """
#     print("inputs, state", inputs, state)
    try:
      num_init_states = len(state)
    except TypeError:
      num_init_states = 1
      state = (state,)
    
    islot_states = dict(enumerate(state))
    try:
      islot_states.update(dict(enumerate(inputs, num_init_states)))
      inputs = islot_states
    except TypeError:
      islot_states.update({num_init_states : inputs})
      inputs = islot_states
      
    output, _ = self.encoder(islot_to_itensor=inputs)

    return (output, output)

  def _get_init_states(self):
    """
    Get the init states of the cell (which will become the init_states of the EvolutionSequence)
    
    The init_states are returned as a list indexed by islot.
    """
    init1 = self.ext_builder.addInput(self.state_dims[:1],
                                 iclass=NormalInputNode)
    init2 = self.ext_builder.addInput(self.state_dims[1:],
                                 iclass=NormalInputNode)
    self.init_states = init_states = [init1, init2]
    
    return init_states
    
    
class NormalTriLCell(CustomCell):
  """
  A CustomCell with 1 inner state, evolved by means of a stochastic normal RNN. 
  """
  def __init__(self,
               state_sizes,
               builder,
               num_inputs=2,
               name=None,
               **dirs):
    """
    Initialize the NormalTriLCell.
    """
    super(NormalTriLCell, self).__init__(num_inputs=num_inputs,
                                         num_outputs=3,
                                         builder=builder)
    try:
      self.state_dims = state_sizes
    except TypeError:
      self.state_dims = [[state_sizes]]
      
    self.cname = 'NormalTriLCell' if name is None else name

    self._update_directives(**dirs)

    self.encoder = self.declare_cell_encoder()
    
    self.init_states = self._get_init_states()
    

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    self.directives = {}
    self.directives.update(dirs)
    
  @property
  def state_size(self):
    """
    Get the states dimensions
    """
    return self.state_dims
  
  @property
  def output_size(self):
    """
    """
    return self.state_dims
  
  def declare_cell_encoder(self):
    """
    Declare the cell inner encoder.
    """
    builder = self.builder
    num_inputs = self.num_expected_inputs
    num_outputs = self.num_expected_outputs
    
    cust = builder.createCustomNode(num_inputs, num_outputs, name="Custom")
    enc_name = cust.addInner(list(self.state_dims),
                             num_inputs=num_inputs,
                             node_class=NormalTriLNode,
                             name='NormalTriLCell')
    
    cust.declareIslot(islot=0, innernode_name=enc_name, inode_islot=0)
    cust.declareIslot(islot=1, innernode_name=enc_name, inode_islot=1)
    cust.declareOslot(oslot=0, innernode_name=enc_name, inode_oslot=0)
    cust.declareOslot(oslot=1, innernode_name=enc_name, inode_oslot=1)
    cust.declareOslot(oslot=2, innernode_name=enc_name, inode_oslot=2)
    cust.commit()

    return cust 
  
  def __call__(self, inputs, state, scope=None):
    """
    Evaluate the cell encoder on a set of inputs
    """
    try:
      num_init_states = len(state)
    except TypeError:
      num_init_states = 1
      state = (state, )
    
    islot_states = dict(enumerate(state))
    try:
      islot_states.update(dict(enumerate(inputs, num_init_states)))
      inputs = islot_states
    except TypeError:
      islot_states.update({num_init_states : inputs})
      inputs = islot_states
      
    output, _ = self.encoder(islot_to_itensor=inputs)
    return (output, output[0])
    
  def on_linked_as_node_2(self, islot):
    """
    """
    if islot == 0:
      if self._islot_to_shape[islot][-1] != self.num_features:
        raise ValueError("Input dimension of BasicCustomCell in islot 0"
                         "must equal self.num_features")
        
  def _get_init_states(self):
    """
    Get the init states of the cell (which will become the init_states of the EvolutionSequence)
    
    The init_states are returned as a list indexed by islot.
    """
    return [self.ext_builder.addInput(self.state_dims,
                                      iclass=NormalInputNode,
                                      **self.directives)]
