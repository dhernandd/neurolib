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

# from neurolib.encoder.custom import CustomNode
# from neurolib.encoder.deterministic import DeterministicNNNode
from neurolib.encoder.input import NormalInputNode
from neurolib.encoder.anode import ANode
from neurolib.builders.static_builder import StaticBuilder
from neurolib.encoder.normal import NormalTriLNode


# pylint: disable=bad-indentation, no-member, protected-access

class CustomCell(ANode, tf.nn.rnn_cell.RNNCell):
  """
  """
  def __init__(self,
               num_inputs,
               num_outputs,
               builder=None):

    """
    """
    super(CustomCell, self).__init__()
    
    self.builder = builder
    self.num_expected_inputs = num_inputs
    self.num_expected_outputs = num_outputs
    
  def get_init_states(self, ext_builder):
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

class BasicNormalTriLCell(CustomCell):
  """
  """
  def __init__(self,
               state_size,
               num_inputs=2,
               builder=None,
               **dirs):
    """
    """
    self.state_dim = state_size
    super(BasicNormalTriLCell, self).__init__(num_inputs=num_inputs,
                                              num_outputs=1,
                                              builder=builder)
    
    self._update_directives(**dirs)
    self.encoder = self.declare_cell_encoder()

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
    return self.state_dim
  
  @property
  def output_size(self):
    """
    """
    return self.state_dim
  
  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    Evaluate the cell encoder on a set of inputs
    """
    output = self.encoder(inputs, islot_to_itensor)
    return (output, output)
  
  def declare_cell_encoder(self):
    """
    """
    if self.builder is None:
      builder = StaticBuilder(scope='Main')
    else:
      builder = self.builder
    
    enc_name = builder.addInner(self.state_dim,
                                num_inputs=2,
                                node_class=NormalTriLNode,
                                name='NormalTrilCell')
    return builder.nodes[enc_name]
    
  def on_linked_as_node_2(self, islot):
    """
    """
    if islot == 0:
      if self._islot_to_shape[islot][-1] != self.num_features:
        raise ValueError("Input dimension of BasicCustomCell in islot 0"
                         "must equal self.num_featuresx")
        
  def get_init_states(self, ext_builder):
    """
    """
    return [ext_builder.addInput(self.state_dim,
                                 iclass=NormalInputNode,
                                 **self.directives)]