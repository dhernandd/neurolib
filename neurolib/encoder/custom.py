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
from neurolib.encoder.inner import InnerNode
from neurolib.encoder.deterministic import DeterministicNNNode  # @UnusedImport
from neurolib.utils.directives import NodeDirectives

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
               name=None,
               name_prefix='Cust',
               **dirs):
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
    # set name
    name_prefix = name_prefix or 'Cust'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    super(CustomNode, self).__init__(out_builder,
                                     is_sequence,
                                     name_prefix=name_prefix,
                                     **dirs)
    
    # number of inputs/outputs
    self.num_expected_inputs = num_inputs
    self.num_expected_outputs = num_outputs
    
    # directives object
    self.directives = NodeDirectives(self.directives)
    self.oslot_names = self.directives.output_names

    # inner builder
    self.in_builder = in_builder
    self.nodes = in_builder.nodes

    self._is_built = False
    
    # init list of free i/o slots
    self._islot_to_itensor = [{} for _ in range(self.num_expected_inputs)]
    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))

  def _update_directives(self, **directives):
    """
    """
    this_node_dirs = {}
    this_node_dirs.update(directives)
    super(CustomNode, self)._update_directives(**this_node_dirs)
    
  def _get_all_oshapes(self):
    """
    TODO!
    """
    pass
    
  def addInner(self, 
               state_sizes,
               num_inputs=1,
               node_class=DeterministicNNNode,
               name=None,
               **dirs):
    """
    Add an InnerNode to the CustomNode
    """
    in_builder = self.in_builder
    node_name = in_builder.addInner(state_sizes,
                                         num_inputs=num_inputs,
                                         node_class=node_class,
                                         is_sequence=False,
                                         name=name,
                                         **dirs)
    node = in_builder.nodes[node_name]
    
    # Assumes fixed number of expected_inputs
    in_builder._innernode_to_its_avlble_islots[node_name] = node.free_islots
    in_builder._innernode_to_its_avlble_oslots[node_name] = node.free_oslots
    
    return node.name
  
  def declareIslot(self, islot, innernode_name, inode_islot):
    """
    Declare an inner islot as an input to the CustomNode
    """
    in_builder = self.in_builder
    in_builder._islot_to_inner_node_islot[islot].append((innernode_name, inode_islot)) 
  
  def declareOslot(self, oslot, innernode_name, inode_oslot):
    """
    Declare an inner oslot as an output to the CustomNode 
    """
    in_builder = self.in_builder
    
    node = in_builder.nodes[innernode_name]
    in_builder.output_nodes[innernode_name] = node 
    in_builder._oslot_to_inner_node_oslot[oslot] = (innernode_name, inode_oslot)
    
  def addDirectedLink(self, enc1, enc2, islot):
    """
    Add a DirectedLink to the CustomNode inner graph
    """
    if isinstance(enc1, str):
      enc1 = self.in_builder.nodes[enc1]
    if isinstance(enc2, str):
      enc2 = self.in_builder.nodes[enc2]
    
    self.in_builder.addDirectedLink(enc1, enc2, islot=islot)
    
  @property
  def dist(self):
    """
    If self has only one OutputNode `onode` and it is a distribution, make
    `self.dist = onode.dist`. Otherwise, raise AttributeError
    """
    in_builder = self.in_builder
    if self.num_expected_outputs == 1:
      print("self._oslot_to_inner_node_oslot", in_builder._oslot_to_inner_node_oslot)
      inner_onode_name = in_builder._oslot_to_inner_node_oslot['main'][0]
      inner_onode = self.in_builder.nodes[inner_onode_name]
      try:
        return inner_onode.dist
      except AttributeError:
        raise AttributeError("The output of this CustomNode is not random")
    else:
      raise AttributeError("`dist` attribute not defined for CustomNodes "
                           "with self.num_outputs > 1")
    
  def __call__(self, *inputs):
    """
    Call the CustomNode
    """
    if not inputs:
      raise ValueError("Inputs are mandatory for the CustomNode")
    print("cust, inputs", inputs)
    islot_to_itensor = [{'main' : ipt} for ipt in inputs]
    return self.get_outputs(islot_to_itensor)

  def get_outputs(self, islot_to_itensor=None):
    """
    Get a sample from the CustomNode
    """
    if islot_to_itensor is not None:
      _input = islot_to_itensor
    else:
      _input = self._islot_to_itensor
    
    print("cust; self.oslot_names", self.oslot_names)
    result = self.in_builder.build_outputs(_input,
                                           self.oslot_names)
    return result

  def _build(self):
    """
    Build the Custom Node
    """    
    # Fill Custom Node oslots
    rslt = self.get_outputs()
    
    for oslot, tensor in enumerate(rslt):
      self.fill_oslot_with_tensor(oslot, tensor)
    self._is_built = True 

  def entropy(self):
    """
    Get the CustomNode entropy
    """
    print("self.dist", self.dist)
    try:
      return self.dist.entropy()
    except AttributeError:
      raise AttributeError("No distribution (`dist` attribute) has been "
                           "associated to this CustomNode")

  def log_prob(self, Y):
    """
    Get the CustomNode loglikelihood
    """
    try:
      return self.dist.log_prob(Y)
    except AttributeError:
      raise AttributeError("No distribution (`dist` attribute) has been "
                           "associated to this CustomNode")
