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
from neurolib.encoder.sequence import EvolutionSequence
from neurolib.encoder.deterministic import DeterministicNNNode  # @UnusedImport

# pylint: disable=bad-indentation, no-member, protected-access

class CustomEvolutionSeq(EvolutionSequence):
  """
  """
  def __init__(self,
               label,
               num_islots,
               num_oslots,
               builder,
               num_features,
               max_steps=30,
               batch_size=1,
               name=None,
               mode='forward'):
    """
    Initialize a CustomNode
    """
    if num_islots is None:
      raise NotImplementedError("CustomNodes with unspecified number of inputs"
                                "not implemented")
    if num_oslots is None:
      raise NotImplementedError("CustomNodes with unspecified number of outputs"
                                "not implemented")
    name = 'CustEvSeq_' + str(label) if name is None else name
    self.num_expected_inputs = num_islots
    self.num_expected_outputs = num_oslots

    super(CustomEvolutionSeq, self).__init__(label,
                                             num_features,
                                             init_states=None,
                                             num_islots=num_islots,
                                             max_steps=max_steps,
                                             batch_size=batch_size,
                                             name=name,
                                             builder=builder,
                                             mode=mode)
    
    self._builder = builder
    
    self._innernode_to_its_avlble_islots = {}
    self._innernode_to_its_avlble_oslots = {}
    self._islot_to_inner_node_islot = {}
    self._oslot_to_inner_node_oslot = {}
            
    self._is_committed = False
    self._is_built = False
    
    self.free_oslots = list(range(self.num_expected_outputs))
    print("self.free_oslots", self.free_oslots)
    
  def addInnerSequence(self, 
               *main_params,
               name=None,
               node_class=DeterministicNNNode,
               **dirs):
    """
    Add an InnerNode to the CustomNode
    """
    node_name = self._builder.addInner(*main_params,
                                        name=name,
                                        node_class=node_class,
                                        **dirs)
    node = self._builder.nodes[node_name]
    
    # Assumes fixed number of expected_inputs
    self._innernode_to_its_avlble_islots[node_name] = list(
                                    range(node.num_expected_inputs))
    self._innernode_to_its_avlble_oslots[node_name] = list(
                                    range(node.num_expected_outputs))
    return node.name
  
  def addEvolutionSequence(self):
    """
    """
    pass
    
  def addDirectedLink(self, enc1, enc2, islot=0, oslot=0):
    """
    Add a DirectedLink to the CustomNode inner graph
    """
    if isinstance(enc1, str):
      enc1 = self._builder.nodes[enc1]
    if isinstance(enc2, str):
      enc2 = self._builder.nodes[enc2]
    self._builder.addDirectedLink(enc1, enc2, islot, oslot)
    
    # Remove the connected islot and oslot from the lists of available ones
    self._innernode_to_its_avlble_oslots[enc1.name].remove(oslot)
    self._innernode_to_its_avlble_islots[enc2.name].remove(islot)
    
  def commit(self):
    """
    Prepare the CustomNode for building.
    
    In particular fill, the dictionaries 
      _islot_to_inner_node_islot ~ {islot : (inner_node, inner_node_islot)}
      _oslot_to_inner_node_oslot ~ {oslot : (inner_node, inner_node_oslot)}
    that allow the build algorithm to connect nodes outside the CustomNode to
    its islots and oslots of 
    """
    print('BEGIN COMMIT')
    # Stage A
    assigned_islots = 0
    for node_name, islot_list in self._innernode_to_its_avlble_islots.items():
      if islot_list:
        node = self._builder.nodes[node_name]
        self._builder.input_nodes[node.name] = node
        for islot in islot_list:
          self._islot_to_inner_node_islot[assigned_islots] = (node, islot)
          assigned_islots += 1
    if assigned_islots != self.num_expected_inputs:
      raise ValueError("Mismatch between num_expected_inputs and "
                       "assigned_islots")
        
    # Stage B
    assigned_oslots = 0
    for node_name, oslot_list in self._innernode_to_its_avlble_oslots.items():
      if oslot_list:
        node = self._builder.nodes[node_name]
        self._builder.output_nodes[node.name] = node
        for oslot in oslot_list:
          self._oslot_to_shape[assigned_oslots] = node.get_oslot_shape(oslot)
          self._oslot_to_inner_node_oslot[assigned_oslots] = (node, oslot)
          assigned_oslots += 1
    if assigned_oslots != self.num_expected_outputs:
      raise ValueError("Mismatch between num_expected_inputs and "
                       "assigned_islots")
    
    self._is_committed = True
    print('END COMMIT')
    
    
  def __call__(self):
    """
    """
    pass
  
  def _build(self):
    """
    """
    pass