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
import abc

from neurolib.encoder.deterministic import DeterministicNNNode #@UnusedImport
from neurolib.utils.utils import check_name

# pylint: disable=bad-indentation, no-member, protected-access

innernode_dict = {'deterministic' : DeterministicNNNode}

class Builder(abc.ABC):
  """
  An abstract class representing the Builder type. A Builder object builds
  a single Model by taking the following steps: 
  
  i) adds encoder nodes to the Model graph (MG)
  ii) defines directed links between them representing tensors
  iii) builds a tensorflow graph from the Model graph 
  
  A Builder object MUST implement the method build()
  """
  def __init__(self, scope, batch_size=None):
    """
    Initialize the builder
    
    Args:
      scope (str): The tensorflow scope of the Model to be built
      
      batch_size (int): The batch size. Defaults to None (unspecified)
    """
    self.scope = scope
    self.batch_size = batch_size
    
    self.num_nodes = 0
    
    # Dictionaries that map name/label to node for the three node types.
    self.nodes = {}
    self.input_nodes = {}
    self.output_nodes = {}
    self._label_to_node = {}

  @check_name
  def addInner(self,
               state_sizes,
               num_inputs=1,
               node_class=DeterministicNNNode,
               is_sequence=False,
               name=None,
               **dirs):
    """
    Add an InnerNode to the Encoder Graph
    
    Args:
      *main_params (list): List of mandatory params for the InnerNode
      node_class (InnerNode): class of the node
      name (str): A unique string identifier for the node being added to the MG
      dirs (dict): A dictionary of directives for the node
    """
    if isinstance(node_class, str):
      node_class = innernode_dict[node_class]
    enc_node = node_class(self,
                          state_sizes,
                          num_inputs=num_inputs,
                          is_sequence=is_sequence,
                          name=name,
                          **dirs)
      
    self.nodes[enc_node.name] = self._label_to_node[enc_node.label] = enc_node
      
    return enc_node.name
  
  @abc.abstractmethod
  def build(self): 
    """
    Build the Model.
    
    Builders MUST implement this method
    """
    raise NotImplementedError("Builders must implement build")
  
  def visualize_model_graph(self, filename="model_graph"):
    """
    Generate a visual representation of the Model graph
    """
    self.model_graph.write_png(self.scope+filename)