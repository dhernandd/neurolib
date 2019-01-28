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

import tensorflow as tf

from neurolib.encoder import _globals as dist_dict
from neurolib.encoder.anode import ANode
from neurolib.utils.utils import basic_concatenation
    
# pylint: disable=bad-indentation, abstract-method

class InnerNode(ANode):
  """
  Abstract class for interior nodes.
  
  An InnerNode is an Anode that resides in the interior of the model graph. It
  performs an operation on its inputs yielding its outputs. Alternatively, an
  InnerNode can be defined as any node that is not an OutputNode nor an
  InputNode. InnerNodes have num_inputs > 0 and num_outputs > 0. Its outputs can
  be deterministic, as in the DeterministicNNNode, or stochastic, as in the
  NormalTriLNode.
  
  The InnerNode should implement `__call__` and `_build`.
  """
  def __init__(self,
               builder,
               is_sequence=False,
               name_prefix=None,
               **dirs):
    """
    Initialize the InnerNode
    
    Args:
      builder (Builder): The builder object for the node
      
      is_sequence (bool): Do the node outputs have a sequential dimension?
    """
    super(InnerNode, self).__init__(builder,
                                    is_sequence,
                                    name_prefix=name_prefix,
                                    **dirs)
    
  def _update_directives(self, **dirs):
    """
    Update the node directives
    
    Add the directives for specific of this class and propagate up the class
    hierarchy
    """
    this_node_dirs = {'output_0_name' : 'main'}
    this_node_dirs.update(dirs)
    super(InnerNode, self)._update_directives(**this_node_dirs)
                  
  def __call__(self, inputs=None):
    """
    Call the node transformation on inputs, return the outputs.
    """
    raise NotImplementedError("Please implement me")
  
  @abstractmethod
  def _build(self):
    """
    Build the node
    """
    raise NotImplementedError("Please implement me.")

 
class CopyNode(InnerNode):
  """
  Utility node that copies its input to its output.
  """
  num_expected_outputs = 1
  num_expected_inputs = 1

  def __init__(self,
               builder,
               name=None):
    """
    Initialize the CopyNode
    """
    super(CopyNode, self).__init__(builder)
    self.name = "Copy_" + str(self.label) if name is None else name
    
  def __call__(self, inputs=None, islot_to_itensor=None):
    """
    Call the CopyNode
    """
    if inputs is not None:
      _input = basic_concatenation(inputs)
    else:
      _input = basic_concatenation(islot_to_itensor)
    
    return _input
  
  def _build(self):
    """
    Build the CopyNode
    """
    output = _input = self._islot_to_itensor[0] # make sure the inputs are ordered
    output_name = self.name + '_out'
    
    self._oslot_to_otensor[0] = tf.identity(output, output_name)
    
    self._is_built = True
    
    
if __name__ == '__main__':
  print(dist_dict)
  