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
from neurolib.encoder.normal import NormalTriLNode
from tensorflow.python.framework.tensor_shape import TensorShape  #pylint: disable=no-name-in-module

# pylint: disable=bad-indentation, no-member, protected-access

class CustomCell(tf.nn.rnn_cell.RNNCell):
  """
  An abstract class for custom cells to be used with sequential models.
  
  A CustomCell is defined in the neurolib as any node that takes its own output
  as an input. Subclasses of CustomCell are expected to define the methods
  
  _update_directives
  declare_cell_encoder
  _get_init_states
  __call__
  
  and the property
  output_size
  """
  def __init__(self,
               builder,
               state_sizes,
               num_inputs,
#                num_outputs,
               **dirs):

    """
    Initialize the CustomCell
    """
    super(CustomCell, self).__init__()

    # builders
    self.ext_builder = builder
    self.batch_size = self.ext_builder.batch_size

    # inputs/outputs
    self.num_expected_inputs = num_inputs
#     self.num_expected_outputs = num_outputs
    
    # state shapes
    self.state_sizes = self.state_sizes_to_list(state_sizes)
    self.state_dims = tuple(state_sizes)
    self.num_states = len(state_sizes)
    self.ranks = self.get_state_size_ranks()
    
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
    self.directives = {}
    self.directives.update(dirs)
  
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
    return self.state_dims

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
   
  def __call__(self, inputs, state):  #pylint: disable=signature-differs
    """
    Evaluate the cell encoder on a set of inputs    
    """
    def make_tuple(t):
      try:
        return tuple(t)
      except TypeError:
        return (t,)
    
    inputs = make_tuple(inputs)
    state = make_tuple(state)
    print("\ncell, inputs", inputs)
    print("cell, state", state)
    print("cell, inputs+state", inputs+state)
    
    output = self.encoder(*inputs, *state)
    print("customcell, output", output)
    if len(output) == 1:
      out = output[0]
      return (out, out)
    print("customcell; (output, output)", (output, output))
    return (output, output)
  
  def compute_output_shape(self, input_shape):
    """
    Compute output shape.
    """
    tf.nn.rnn_cell.RNNCell.compute_output_shape(self, input_shape)


class BasicEncoderCell(CustomCell):
  """
  A CustomCell with 1 inner state.
  """
  num_expected_outputs = 1
  def __init__(self,
               builder,
               state_sizes,
               num_inputs=2,
#                num_outputs=1,
               name=None,
               **dirs):
    """
    Initialize the BasicEncoderCell
    
    Args:
        state_sizes (list of list of ints):
        
        builder :
               
        num_inputs :

        num_outputs :
  
        name :
        
        dirs :
    """
    self.cname = 'BasicCell' if name is None else name

    super(BasicEncoderCell, self).__init__(builder,
                                           state_sizes,
                                           num_inputs=num_inputs,
#                                            num_outputs=num_outputs,
                                           **dirs)

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_cell_directives = {'outputname_0' : 'main'}
    this_cell_directives.update(dirs)
    super(BasicEncoderCell, self)._update_directives(**this_cell_directives)
    
  def declare_cell_encoder(self):
    """
    Declare the cell inner encoder
    """
    builder = self.ext_builder
    num_inputs = self.num_expected_inputs
    num_outputs = self.num_expected_outputs
    
    # Create the cell's Custom Encoder
    cust = builder.createCustomNode(num_inputs, num_outputs, name="CellEnc")
    
    cust_in1 = cust.addInner(self.state_size[0:1], num_inputs=2)
    cust.declareIslot(islot=0, innernode_name=cust_in1, inode_islot=0)
    cust.declareIslot(islot=1, innernode_name=cust_in1, inode_islot=1)

    cust.declareOslot(oslot='main', innernode_name=cust_in1, inode_oslot='main')
        
    return cust

  def _get_init_states(self):
    """
    Get the init states of the cell (which become the initial states of the
    EvolutionSequence)
    
    The init_states are returned as a list indexed by islot.
    """
    init1 = self.ext_builder.addInput(self.state_dims[:1], iclass=NormalInputNode)
    return [init1]

  @property
  def output_size(self):
    """
    """
    return self.state_dims


class TwoEncodersCell(CustomCell):
  """
  A CustomCell with 2 inner states, each evolved by means of a deterministic RNN.
  """
  num_expected_outputs = 2
  
  def __init__(self,
               ext_builder,
               state_sizes,
               num_inputs=3,
#                num_outputs=2,
               name=None,
               **dirs):
    """
    Initialize the TwoEncodersCell
    
    Args:
        state_sizes (list of list of ints):
        
        ext_builder :
               
        num_inputs :

        num_outputs :
  
        name :
        
        dirs :
    """
    self.cname = 'TwoEncCell' if name is None else name
    
    super(TwoEncodersCell, self).__init__(ext_builder,
                                          state_sizes,
                                          num_inputs=num_inputs,
#                                           num_outputs=num_outputs,
                                          **dirs)

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_cell_directives = {'outputname_0' : 'sec',
                            'outputname_1' : 'main'}
    this_cell_directives.update(dirs)
    super(TwoEncodersCell, self)._update_directives(**this_cell_directives)
    
  def declare_cell_encoder(self):
    """
    Declare the cell inner encoder
    """
    builder = self.ext_builder
    num_inputs = self.num_expected_inputs
    num_outputs = self.num_expected_outputs
    dirs = self.directives
    
    # Create the cell's Custom Encoder
    cust = builder.createCustomNode(num_inputs, num_outputs, name="Custom", **dirs)
    
    cust_in1 = cust.addInner(self.state_size[0:1], num_inputs=2)
    cust_in2 = cust.addInner(self.state_size[1:2], num_inputs=2)
    cust.addDirectedLink(cust_in1, cust_in2, islot=1)
    cust.declareIslot(islot=0, innernode_name=cust_in1, inode_islot=0)
    cust.declareIslot(islot=1, innernode_name=cust_in2, inode_islot=0)
    cust.declareIslot(islot=2, innernode_name=cust_in1, inode_islot=1)

    cust.declareOslot(oslot='sec', innernode_name=cust_in1, inode_oslot='main')
    cust.declareOslot(oslot='main', innernode_name=cust_in2, inode_oslot='main')    
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
  def output_size(self):
    """
    """
    return self.state_dims
    

class TwoEncodersCell2(CustomCell):
  """
  A CustomCell with 2 inner states, each evolved by means of a deterministic RNN.
  """
  num_expected_outputs = 2
  
  def __init__(self,
               builder,
               state_sizes,
               num_inputs=3,
#                num_outputs=2,
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
    self.cname = 'TwoEncCell' if name is None else name

    super(TwoEncodersCell2, self).__init__(builder,
                                           state_sizes,
                                           num_inputs=num_inputs,
#                                            num_outputs=num_outputs,
                                           **dirs)

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_cell_directives = {'outputname_0' : 'sec',
                            'outputname_1' : 'main'}
    this_cell_directives.update(dirs)
    super(TwoEncodersCell2, self)._update_directives(**this_cell_directives)
    
  def declare_cell_encoder(self):
    """
    Declare the cell inner encoder
    """
    builder = self.ext_builder
    num_inputs = self.num_expected_inputs
    num_outputs = self.num_expected_outputs
    dirs = self.directives
    
    cust = builder.createCustomNode(num_inputs, num_outputs, name="Custom",
                                    **dirs)
    cust_in1 = cust.addInner(self.state_size[0:1], num_inputs=3) # 3 inputs
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

    cust.declareOslot(oslot='sec', innernode_name=cust_in1, inode_oslot='main')
    cust.declareOslot(oslot='main', innernode_name=cust_in2, inode_oslot='main')    
    
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
      
  @property
  def output_size(self):
    """
    The size of the outputs
    """
    return self.state_dims


class NormalTriLCell(CustomCell):
  """
  A CustomCell with 1 inner state, evolved by means of a stochastic normal RNN. 
  """
  num_expected_outputs = 3
  
  def __init__(self,
               builder,
               state_sizes,
               num_inputs=2,
               name=None,
               **dirs):
    """
    Initialize the NormalTriLCell.
    """
    self.cname = 'NormalTriLCell' if name is None else name

    super(NormalTriLCell, self).__init__(builder,
                                         state_sizes,
                                         num_inputs=num_inputs,
#                                          num_outputs=3,
                                         **dirs)
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_cell_directives = {'outputname_0' : 'main',
                            'outputname_1' : 'loc',
                            'outputname_2' : 'scale'}
    this_cell_directives.update(dirs)
    super(NormalTriLCell, self)._update_directives(**this_cell_directives)
        
  def declare_cell_encoder(self):
    """
    Declare the cell inner encoder
    """
    builder = self.ext_builder
    num_inputs = self.num_expected_inputs
    num_outputs = self.num_expected_outputs
    dirs = self.directives
    
    cust = builder.createCustomNode(num_inputs, num_outputs, name="Custom",
                                    **dirs)
    enc_name = cust.addInner(list(self.state_dims),
                             num_inputs=num_inputs,
                             node_class=NormalTriLNode,
                             name='NormalTriLCell')
    
    cust.declareIslot(islot=0, innernode_name=enc_name, inode_islot=0)
    cust.declareIslot(islot=1, innernode_name=enc_name, inode_islot=1)
    cust.declareOslot(oslot='main', innernode_name=enc_name, inode_oslot='main')
    cust.declareOslot(oslot='loc', innernode_name=enc_name, inode_oslot='loc')
    cust.declareOslot(oslot='scale', innernode_name=enc_name, inode_oslot='scale')

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
  def output_size(self):
    """
    Set self.output_size
    """
    return (TensorShape(self.state_dims[0]),
            TensorShape(self.state_dims[0]),
            TensorShape(self.state_dims[0]*2))
  
  def __call__(self, inputs, state):
    """
    Evaluate the cell encoder on a set of inputs
    """
    output, _ = super(NormalTriLCell, self).__call__(inputs, state)
    return (output[:], output[0])
    
    
# class LDSCell(CustomCell):
#   """
#   A CustomCell with 1 inner state, evolved by means of a stochastic Linear
#   Dynamical System (LDS).
#   """
#   num_expected_outputs = 4
#   
#   def __init__(self,
#                builder,
#                state_sizes,
#                num_inputs=1,
#                name=None,
#                **dirs):
#     """
#     Initialize the LDSCell.
#     """
#     self.cname = 'LDSCell' if name is None else name
# 
#     super(LDSCell, self).__init__(builder,
#                                   state_sizes,
#                                   num_inputs=num_inputs,
# #                                   num_outputs=4,
#                                   **dirs)
# 
#   def _update_directives(self, **dirs):
#     """
#     Update the node directives
#     """
#     this_cell_directives = {'outputname_0' : 'main',
#                             'outputname_1' : 'loc',
#                             'outputname_2' : 'A',
#                             'outputname_3' : 'prec'}
#     this_cell_directives.update(dirs)
#     super(LDSCell, self)._update_directives(**this_cell_directives)
#     
#   def declare_cell_encoder(self):
#     """
#     Declare the cell inner encoder.
#     """
#     builder = self.ext_builder
#     num_inputs = self.num_expected_inputs
#     num_outputs = self.num_expected_outputs
#     dirs = self.directives
#     
#     cust = builder.createCustomNode(num_inputs, num_outputs, name="Custom",
#                                     **dirs)
#     enc_name = cust.addInner(list(self.state_dims),
#                              num_inputs=num_inputs,
#                              node_class=LDSNode,
#                              name='LDSCell')
#     
#     cust.declareIslot(islot=0, innernode_name=enc_name, inode_islot=0)
#     cust.declareOslot(oslot='main', innernode_name=enc_name, inode_oslot='main')
#     cust.declareOslot(oslot='loc', innernode_name=enc_name, inode_oslot='loc')
#     cust.declareOslot(oslot='prec', innernode_name=enc_name, inode_oslot='prec')
#     cust.declareOslot(oslot='A', innernode_name=enc_name, inode_oslot='A')
# 
#     return cust
#     
#   def _get_init_states(self):
#     """
#     Get the init states of the cell (which are also the init states of the
#     EvolutionSequence)
#     
#     The init_states are returned as a list indexed by islot.
#     
#     The directives to the init states should are passed to the Evolution
#     Sequence with the prefix 'init_'. 
#     
#     TODO: Change this so that the prefix is 'initx_' where x is the islot or
#     think a better way
#     """
#     init_dirs = {key[5:] : self.directives[key] for key in self.directives
#                  if key.startswith('init_')}
#     return [self.ext_builder.addInput(self.state_dims,
#                                       iclass=NormalInputNode,
#                                       **init_dirs)]
#     
#   @property
#   def output_size(self):
#     """
#     Set self.output_size
#     """
#     return (TensorShape(self.state_dims[0]),
#             TensorShape(self.state_dims[0]))
# #             TensorShape(self.state_dims[0]*2),
# #             TensorShape(self.state_dims[0]*2))
# 
#   def __call__(self, inputs, state):  #pylint: disable=signature-differs
#     """
#     Evaluate the cell encoder on a set of inputs    
#     """
#     def make_tuple(t):
#       try:
#         return tuple(t)
#       except TypeError:
#         return (t,)
#     
#     state = make_tuple(state)
# #     output = self.encoder(*state)
#     output = self.encoder(*state)[:2] # !! COMMENT
# 
#     return (output[:], output[0])
# 
#   def build_output(self, oslot, *inputs):
#     """
#     Get an oslot output tensor
#     """
#     return self.encoder.build_output(oslot, *inputs)
#     
#   
# class LLDSCell(CustomCell):
#   """
#   A CustomCell with 1 inner state, evolved by means of a stochastic Linear
#   Dynamical System (LDS).
#   """
#   num_expected_outputs = 5
#   
#   def __init__(self,
#                builder,
#                state_sizes,
#                num_inputs=1,
#                name=None,
#                **dirs):
#     """
#     Initialize the LLDSCell.
#     """
#     self.cname = 'LLDSCell' if name is None else name
# 
#     super(LLDSCell, self).__init__(builder,
#                                   state_sizes,
#                                   num_inputs=num_inputs,
#                                   **dirs)
# 
#   def _update_directives(self, **dirs):
#     """
#     Update the node directives
#     """
#     this_cell_directives = {'outputname_0' : 'main',
#                             'outputname_1' : 'loc',
#                             'outputname_2' : 'A',
#                             'outputname_3' : 'prec',
#                             'outputname_4' : 'Alinear'}
#     this_cell_directives.update(dirs)
#     super(LLDSCell, self)._update_directives(**this_cell_directives)
#     
#   def declare_cell_encoder(self):
#     """
#     Declare the cell inner encoder.
#     """
#     builder = self.ext_builder
#     num_inputs = self.num_expected_inputs
# #     num_outputs = self.num_expected_outputs
#     dirs = self.directives
#     
#     enc_name = builder.addInner(list(self.state_dims),
#                                 num_inputs=num_inputs,
#                                 node_class=LDSNode,
#                                 name='LLDSCell')
#     enc = builder.nodes[enc_name]
#     
# #     cust = builder.createCustomNode(num_inputs, num_outputs, name="Custom",
# #                                     **dirs)
# #     enc_name = cust.addInner(list(self.state_dims),
# #                              num_inputs=num_inputs,
# #                              node_class=LDSNode,
# #                              name='LDSCell')
# #     
# #     cust.declareIslot(islot=0, innernode_name=enc_name, inode_islot=0)
# #     cust.declareOslot(oslot='main', innernode_name=enc_name, inode_oslot='main')
# #     cust.declareOslot(oslot='loc', innernode_name=enc_name, inode_oslot='loc')
# #     cust.declareOslot(oslot='prec', innernode_name=enc_name, inode_oslot='prec')
# #     cust.declareOslot(oslot='A', innernode_name=enc_name, inode_oslot='A')
# 
#     return enc
#     
#   def _get_init_states(self):
#     """
#     Get the init states of the cell (which are also the init states of the
#     EvolutionSequence)
#     
#     The init_states are returned as a list indexed by islot.
#     
#     The directives to the init states should are passed to the Evolution
#     Sequence with the prefix 'init_'. 
#     
#     TODO: Change this so that the prefix is 'initx_' where x is the islot or
#     think a better way
#     """
#     init_dirs = {key[5:] : self.directives[key] for key in self.directives
#                  if key.startswith('init_')}
#     return [self.ext_builder.addInput(self.state_dims,
#                                       iclass=NormalInputNode,
#                                       **init_dirs)]
#     
#   @property
#   def output_size(self):
#     """
#     Set self.output_size
#     """
#     return (TensorShape(self.state_dims[0]),
#             TensorShape(self.state_dims[0]))
# #             TensorShape(self.state_dims[0]*2),
# #             TensorShape(self.state_dims[0]*2))
# 
#   def __call__(self, inputs, state):  #pylint: disable=signature-differs
#     """
#     Evaluate the cell encoder on a set of inputs    
#     """
#     def make_tuple(t):
#       try:
#         return tuple(t)
#       except TypeError:
#         return (t,)
#     
#     state = make_tuple(state)
# #     output = self.encoder(*state)
#     output = self.encoder(*state)[:2] # !! COMMENT
# 
#     return (output[:], output[0])
# 
#   def build_output(self, oslot, *inputs):
#     """
#     Get an oslot output tensor
#     """
#     return self.encoder.build_output(oslot, *inputs)
#     