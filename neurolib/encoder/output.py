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

from neurolib.encoder.anode import ANode

# pylint: disable=bad-indentation, no-member, protected-access

class OutputNode(ANode):
  """
  An ANode representing a sink of information in the Model Graph (MG).
  
  OutputNodes are mainly useful for model bending (training). A cost function
  for instance, should depends only on the tensors flowing into OutputNodes (inputs).
    
  OutputNodes have no outputs, that is, information is "destroyed" at the
  OutputNode. Assignment to self.num_declared_outputs is therefore forbidden.
  OutputNodes have a single input assigned to islot = 0. 
  
  Class attributes:
    num_expected_inputs = 1
    num_expected_outputs = 0
  """
  num_expected_inputs = 1
  num_expected_outputs = 0
  
  def __init__(self,
               builder,
               name=None):
    """
    Initialize the OutputNode
    
    Args:
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes.

      name (str): A unique name for this node.
    """
    super(OutputNode, self).__init__()

    self.builder = builder
    self.label = builder.num_nodes
    builder.num_nodes += 1
    
    self.name = "Out_" + str(self.label) if name is None else name
    
    self.free_islots = list(range(self.num_expected_inputs))
    
  def _build(self):
    """
    Build the OutputNode.
    
    Renames the input tensor to the name of the node so that it can be easily
    accessed.
    
    NOTE: The _islot_to_itensor attribute of this node has been updated by the
    Builder object during processing of this OutputNode's parent by the
    Builder build algorithm
    """
    self._islot_to_itensor[0] = tf.identity(self._islot_to_itensor[0],
                                            name=self.name)
    
    self._is_built = True
    
class OutputSequence(OutputNode):
  """
  """  
  def __init__(self,
               builder,
               name=None):
    """
    Initialize the OutputSequence. This is the same as an OutputNode but for the name
    
    Args:
      label (int): A unique integer identifier for the node.
      
      name (str): A unique name for this node.
    """
    super(OutputSequence, self).__init__(builder, name)
    name = "OutSeq_" + str(self.label) if name is None else name