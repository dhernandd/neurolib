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
from bidict import bidict

from neurolib.encoder.basic import InnerNode
from neurolib.encoder.deterministic import DeterministicNNNode  # @UnusedImport

# pylint: disable=bad-indentation, no-member, protected-access


class CustomNode(InnerNode):
  """
  A custom InnerNode that can be built using a builder attribute 
  
  A CustomNode is an InnerNode representing an arbitrary map. In particular, it
  may take an arbitrary number of inputs and return an arbitrary number of
  outputs. CustomNodes may be used to put together portions of the computational
  graph into an encoder that can be subsequently treated as a black box.
  
  A CustomNode defines two dictionaries to handle the mapping between the
  CustomNode islots and the islots of the InnerNodes that are built inside it.
  
  _islot_to_inner_node_islot ~ {islot : (inner_node, inner_node_islot)}
  _oslot_to_inner_node_oslot ~ {oslot : (inner_node, inner_node_oslot)}

  """
  def __init__(self,
               out_builder,
               in_builder,
               num_inputs,
               num_outputs,
               is_sequence=False,
               name=None):
    """
    Initialize a CustomNode
    
    
    Args:
        out_builder (Builder) : External builder used to create the CustomNode
        
        in_builder (Builder) : Internal builder, used by the CustomNode to build
            its InnerNodes
        
        num_inputs (int) :
        
        num_outputs (int) :
        
        is_sequence (bool) : 
        
        name (str) :
    """
    super(CustomNode, self).__init__(out_builder,
                                     is_sequence)
    
    self.num_expected_inputs = num_inputs
    self.num_expected_outputs = num_outputs

    self.in_builder = in_builder
    self.nodes = in_builder.nodes
    self.name = 'Cust_' + str(self.label) if name is None else name

    self._innernode_to_its_avlble_islots = {}
    self._innernode_to_its_avlble_oslots = {}
    self._islot_to_inner_node_islot = bidict()
    self._oslot_to_inner_node_oslot = bidict()
    
    self._is_committed = False
    self._is_built = False
    
    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))

  def declareIslot(self, islot=None, innernode_name=None, inode_islot=None):
    """
    Declare an inner islot as an input to the CustomNode
    """
    if any(elem is None for elem in [islot, innernode_name, inode_islot]):
      raise ValueError("Missing argument")
    node = self.in_builder.nodes[innernode_name]
    self.in_builder.input_nodes[innernode_name] = node 
    self._islot_to_inner_node_islot[islot] = (innernode_name, inode_islot) 
  
  def declareOslot(self, oslot=None, innernode_name=None, inode_oslot=None):
    """
    Declare an inner oslot as an output to the CustomNode 
    """
    if any(elem is None for elem in [oslot, innernode_name, inode_oslot]):
      raise ValueError("Missing argument")

    self._oslot_to_inner_node_oslot[oslot] = (innernode_name, inode_oslot)
    
    node = self.in_builder.nodes[innernode_name]
    self.in_builder.output_nodes[innernode_name] = node
     
    self._oslot_to_shape[oslot] = node.get_oslot_shape(inode_oslot)
    
  def addInner(self, 
#                *main_params,
               state_sizes,
               num_inputs=1,
               node_class=DeterministicNNNode,
               name=None,
               **dirs):
    """
    Add an InnerNode to the CustomNode
    """
    node_name = self.in_builder.addInner(state_sizes,
                                         num_inputs=num_inputs,
#                                          *main_params,
                                         node_class=node_class,
                                         is_sequence=False,
                                         name=name,
                                         **dirs)
    node = self.in_builder.nodes[node_name]
    
    # Assumes fixed number of expected_inputs
    self._innernode_to_its_avlble_islots[node_name] = node.free_islots
    self._innernode_to_its_avlble_oslots[node_name] = node.free_oslots
    
    return node.name
    
  def addDirectedLink(self, enc1, enc2, islot=0, oslot=0):
    """
    Add a DirectedLink to the CustomNode inner graph
    """
    if isinstance(enc1, str):
      enc1 = self.in_builder.nodes[enc1]
    if isinstance(enc2, str):
      enc2 = self.in_builder.nodes[enc2]
    
    self.in_builder.addDirectedLink(enc1, enc2, islot, oslot)

    # Remove the connected islot and oslot from the lists of available ones
#     self._innernode_to_its_avlble_oslots[enc1.name].remove(oslot)
#     self._innernode_to_its_avlble_islots[enc2.name].remove(islot)
    
  @property
  def dist(self):
    """
    If self has only one OutputNode `onode` and it is a distribution, make
    `self.dist = onode.dist`. Otherwise, raise AttributeError
    """
    if self.num_expected_outputs == 1:
      inner_onode_name = self._oslot_to_inner_node_oslot[0][0]
      inner_onode = self.in_builder.nodes[inner_onode_name]
      try:
        return inner_onode.dist
      except AttributeError:
        raise AttributeError("The output of this CustomNode is not random")
    else:
      raise AttributeError("`dist` attribute not defined for CustomNodes "
                           "with self.num_outputs > 1")
    
  def _get_output(self, inputs=None, islot_to_itensor=None, name=None):
    """
    Get a sample from the CustomNode
    """
    if not self._is_built:
      raise ValueError("Node is not built")
    
    num_outputs = self.num_expected_outputs
    rslt, rslt_dict = self.in_builder.get_output(inputs, islot_to_itensor, self, name)
    for i in range(num_outputs):
      onode_name, oslot = self._oslot_to_inner_node_oslot[i]
      onode = self.in_builder.nodes[onode_name]
      self._oslot_to_otensor[i] = onode.get_outputs()[oslot]

    return rslt, rslt_dict
  
  def commit(self):
    """
    Prepare the CustomNode for building.
    
    In particular fill, the dictionaries 
      _islot_to_inner_node_islot ~ {inode_islot : (inner_node, inner_node_islot)}
      _oslot_to_inner_node_oslot ~ {inode_oslot : (inner_node, inner_node_oslot)}
  
    Stage A: If an InnerNode of the CustomNode has an available inode_islot, then this
    is an input to the CustomNode.
    """
    print('BEGIN COMMIT')
    # Stage A
    assigned_islots = 0
    for innernode_name, islot_list in self._innernode_to_its_avlble_islots.items():
      if islot_list:
#         inode = self.in_builder.nodes[innernode_name]
#         self.in_builder.input_nodes[inode.name] = inode
        for inode_islot in islot_list:
          if (innernode_name, inode_islot) not in self._islot_to_inner_node_islot.values():
#             print("(innernode_name, inode_islot)", (innernode_name, inode_islot))
#             print("self._islot_to_inner_node_islot.items()",
#                   self._islot_to_inner_node_islot.values())
            raise ValueError("")
          assigned_islots += 1
    if assigned_islots != self.num_expected_inputs:
      raise ValueError("Mismatch between num_expected_inputs and "
                       "assigned_islots")
        
#     Stage B
    assigned_oslots = 0
#     print("self._innernode_to_its_avlble_oslots.items()", 
#           self._innernode_to_its_avlble_oslots.items())
    for innernode_name, oslot_list in self._innernode_to_its_avlble_oslots.items():
      if oslot_list:
#         inode = self.in_builder.nodes[innernode_name]
#         self.in_builder.output_nodes[inode.name] = inode
        for inode_oslot in oslot_list:
#           print("self._oslot_to_inner_node_oslot.values()",
#                 self._oslot_to_inner_node_oslot.items())
#           print("(innernode_name, inode_islot)", (innernode_name, inode_oslot))
#           print("self._islot_to_inner_node_islot.items()",
#                 self._oslot_to_inner_node_oslot.items())
#           print("(innernode_name, inode_oslot)", 
#                 [(innernode_name, inode_oslot) in self._oslot_to_inner_node_oslot.values()])
          if (innernode_name, inode_oslot) not in self._oslot_to_inner_node_oslot.values():
            raise ValueError("")
          assigned_oslots += 1
    if assigned_oslots != self.num_expected_outputs:
      raise ValueError("Mismatch between num_expected_inputs and "
                       "assigned_islots")
    
    self._is_committed = True
    print('END COMMIT')

  def __call__(self, inputs=None, itensor_to_islot=None):
    """
    """
    return self._get_output(inputs, itensor_to_islot)
  
  def _build(self):
    """
    Build the CustomNode
    """
    print("\nBEGIN CUSTOM BUILD")

    num_outputs = self.num_expected_outputs
    if self.num_declared_outputs != num_outputs:
      raise ValueError("`self.num_declared_outputs != self.num_expected_outputs`",
                       (self.num_declared_outputs, self.num_expected_outputs))
    
    self.in_builder.build()
    for i in range(num_outputs):
      onode_name, oslot = self._oslot_to_inner_node_oslot[i]
      onode = self.in_builder.nodes[onode_name]
      self._oslot_to_otensor[i] = onode.get_outputs()[oslot]
    
    output = self._oslot_to_otensor
    self._is_built = True
    print("END CUSTOM BUILD\n")
#     else:
#       print("\nBEGIN CUSTOM BUILD")
#       temp = self._islot_to_itensor # horrible hack to allow calling for a CustomNode
#       self._islot_to_itensor = islot_to_itensor
#       
#       self.in_builder.build()
#       output = {}
#       for i in range(num_outputs):
#         onode_name, oslot = self._oslot_to_inner_node_oslot[i]
#         onode = self.in_builder.nodes[onode_name]
#         output[i] = onode.get_outputs()[oslot]
#       self._islot_to_itensor = temp

    return list(zip(*sorted(output.items())))[1]
    
  def get_node(self, label):
    """
    """
    return self.in_builder.nodes[label]