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
from neurolib.builders.static_builder import StaticBuilder
from neurolib.encoder.normal import NormalTriLNode, LDSNode
from tensorflow.python.framework.tensor_shape import TensorShape  #pylint: disable=no-name-in-module


# pylint: disable=bad-indentation, no-member, protected-access

class CustomCell(tf.nn.rnn_cell.RNNCell):
  """
  A class for custom cells to be used with sequential models
  """
  def __init__(self,
               builder,
               state_sizes,
               num_inputs,
               num_outputs,
               **dirs):

    """
    Initialize the CustomCell
    """
    super(CustomCell, self).__init__()

    self.ext_builder = builder
    self.builder = StaticBuilder(scope='Cell')
    self.batch_size = self.builder.batch_size

    self.num_expected_inputs = num_inputs
    self.num_expected_outputs = num_outputs
    
    self._oslot_to_shape = {}
    
    # Set the state sizes
    self.state_sizes = self.state_sizes_to_list(state_sizes)
    self.state_dims = tuple(state_sizes)
    self.num_states = len(state_sizes)
    self.main_oshape = self.get_state_full_shapes()
    self.D = self.get_state_size_ranks()
    
    self._update_directives(**dirs)

    # Declare the cell encoder 
    self.encoder = self.declare_cell_encoder()
    self.init_states = self._get_init_states()
    
    # Declare secondary outputs
    self.secondary_outputs = list(range(self.num_states, self.num_expected_outputs))

  def _update_directives(self, **dirs):
    """
    Update the cell directives
    """
    raise NotImplementedError("")
  
  def declare_cell_encoder(self):
    """
    Declare the cell inner encoder.
    """
    raise NotImplementedError("")
  
  def _get_init_states(self):
    """
    Get the cell init_states
    """
    raise NotImplementedError("")
  
  @staticmethod
  def state_sizes_to_list(state_sizes):
    """
    Get a list of output sizes corresponding to each oslot
    """
    if isinstance(state_sizes, int):
      state_sizes = [[state_sizes]]
    elif isinstance(state_sizes, list):
      if not all([isinstance(size, list) for size in state_sizes]):
        raise TypeError("`state_sizes` argument must be an int of a list of lists "
                        "of int")
    return state_sizes

  def get_state_full_shapes(self):
    """
    Get the state size shapes for this ANode    
    """
    return [[self.batch_size] + sz for sz in self.state_sizes]
  
  def get_state_size_ranks(self): 
    """
    Get the ranks of the states for this ANode
    """
    return [len(sz) for sz in self.state_sizes]
  
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

  def get_oshapes(self, oslot):
    """
    Get the cell output shapes
    """
    return self.encoder.get_oslot_shape(oslot)
   
  def __call__(self, inputs, state, scope=None):
    """
    Evaluate the cell encoder on a set of inputs    
    """
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
    super(TwoEncodersCell, self).__init__(builder,
                                          state_sizes,
                                          num_inputs=num_inputs,
                                          num_outputs=num_outputs,
                                          **dirs)
    self.cname = 'TwoEncCell' if name is None else name

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    self.directives = {}
    self.directives.update(dirs)
    
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

  def _get_init_states(self):
    """
    Get the init states of the cell (which become the initial states of the
    EvolutionSequence)
    
    The init_states are returned as a list indexed by islot.
    """
    init1 = self.ext_builder.addInput(self.state_dims[:1], iclass=NormalInputNode)
    init2 = self.ext_builder.addInput(self.state_dims[1:], iclass=NormalInputNode)
    
    return [init1, init2]
    
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
    

class TwoEncodersCell2(CustomCell):
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
    super(TwoEncodersCell2, self).__init__(builder,
                                           state_sizes,
                                           num_inputs=num_inputs,
                                           num_outputs=num_outputs,
                                           **dirs)
    self.cname = 'TwoEncCell' if name is None else name

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    self.directives = {}
    self.directives.update(dirs)
    
  @property
  def state_size(self):
    """
    The size of the inner state
    """
    return self.state_dims
  
  @property
  def output_size(self):
    """
    The size of the outputs
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
    cust_in1 = cust.addInner(self.state_size[0:1], num_inputs=3)
    cust_in2 = cust.addInner(self.state_size[1:2], num_inputs=2)
    cust.addDirectedLink(cust_in1, cust_in2, islot=1)

    # These lines are the only difference between TwoEncodersCell2 and
    # TwoEncodersCell above. Specifically, islot 1 of the CustomCell, is an
    # input here for BOTH InnerNodes. This results in forward links between
    # different Encoders when the RNN is unrolled.
    cust.declareIslot(islot=0, innernode_name=cust_in1, inode_islot=0)
    cust.declareIslot(islot=1, innernode_name=cust_in1, inode_islot=1)
    cust.declareIslot(islot=1, innernode_name=cust_in2, inode_islot=0)
    cust.declareIslot(islot=2, innernode_name=cust_in1, inode_islot=2)

    cust.declareOslot(oslot=0, innernode_name=cust_in1, inode_oslot=0)
    cust.declareOslot(oslot=1, innernode_name=cust_in2, inode_oslot=0)
    cust.commit()
    
    return cust
  
  def _get_init_states(self):
    """
    Get the init states of the cell (which will become the init_states of the
    EvolutionSequence)
    
    The init_states are returned as a list indexed by islot.
    """
    init1 = self.ext_builder.addInput(self.state_dims[:1], iclass=NormalInputNode)
    init2 = self.ext_builder.addInput(self.state_dims[1:], iclass=NormalInputNode)
    
    return [init1, init2]


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
    super(NormalTriLCell, self).__init__(builder,
                                         state_sizes,
                                         num_inputs=num_inputs,
                                         num_outputs=3,
                                         **dirs)
    self.cname = 'NormalTriLCell' if name is None else name
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    self.directives = {'out_0_name' : 'main0',
                       'out_1_name' : 'mean',
                       'out_2_name' : 'scale'}
    
    self.directives.update(dirs)
    
  def declare_cell_encoder(self):
    """
    Declare the cell inner encoder
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
  
  def _get_init_states(self):
    """
    Get the init states of the cell (which will become the init_states of the EvolutionSequence)
    
    The init_states are returned as a list indexed by islot.
    """
    return [self.ext_builder.addInput(self.state_dims,
                                      iclass=NormalInputNode,
                                      **self.directives)]
    
  @property
  def state_size(self):
    """
    Get self.state_size
    """
    return self.state_dims
  
  @property
  def output_size(self):
    """
    Set self.output_size
    """
    return (TensorShape(self.state_dims[0]),
            TensorShape(self.state_dims[0]),
            TensorShape(self.state_dims[0]*2))
  
  def __call__(self, inputs, state, scope=None):
    """
    Evaluate the cell encoder on a set of inputs
    """
    output, _ = super(NormalTriLCell, self).__call__(inputs, state, scope=scope)

    return (output[:], output[0])
    
  def on_linked_as_node_2(self, islot):
    """
    """
    if islot == 0:
      if self._islot_to_shape[islot][-1] != self.num_features:
        raise ValueError("Input dimension of BasicCustomCell in islot 0"
                         "must equal self.num_features")
        

class LDSCell(CustomCell):
  """
  A CustomCell with 1 inner state, evolved by means of a stochastic Linear
  Dynamical System (LDS).
  """
  def __init__(self,
               state_sizes,
               builder,
               num_inputs=1,
               name=None,
               **dirs):
    """
    Initialize the LDSCell.
    """
    super(LDSCell, self).__init__(builder,
                                  state_sizes,
                                  num_inputs=num_inputs,
                                  num_outputs=4,
                                  **dirs)
    self.cname = 'LDSCell' if name is None else name

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    self.directives = {'out_0_name' : 'main',
                       'out_1_name' : 'loc',
                       'out_2_name' : 'invQ',
                       'out_3_name' : 'A'}
    
    self.directives.update(dirs)
    
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
                             node_class=LDSNode,
                             name='LDSCell')
    
    cust.declareIslot(islot=0, innernode_name=enc_name, inode_islot=0)
    cust.declareOslot(oslot=0, innernode_name=enc_name, inode_oslot=0)
    cust.declareOslot(oslot=1, innernode_name=enc_name, inode_oslot=1)
    cust.declareOslot(oslot=2, innernode_name=enc_name, inode_oslot=2)
    cust.declareOslot(oslot=3, innernode_name=enc_name, inode_oslot=3)
    cust.commit()

    return cust
    
  def _get_init_states(self):
    """
    Get the init states of the cell (which will become the init_states of the
    EvolutionSequence)
    
    The init_states are returned as a list indexed by islot.
    """
    return [self.ext_builder.addInput(self.state_dims,
                                      iclass=NormalInputNode,
                                      **self.directives)]
    
  @property
  def state_size(self):
    """
    Get self.state_size
    """
    return self.state_dims
  
  @property
  def output_size(self):
    """
    Set self.output_size
    """
    return (TensorShape(self.state_dims[0]),
            TensorShape(self.state_dims[0]),
            TensorShape(self.state_dims[0]*2),
            TensorShape(self.state_dims[0]*2))

  def __call__(self, inputs, state, scope=None):
    """
    Evaluate the cell encoder on a set of inputs
    """
    output, _ = super(LDSCell, self).__call__(inputs, state, scope=scope)
    
    return (output[:], output[0])
  