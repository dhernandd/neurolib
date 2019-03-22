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
from neurolib.encoder.deterministic import DeterministicNN  # @UnusedImport
from neurolib.utils.directives import NodeDirectives

# pylint: disable=bad-indentation, no-member, protected-access


class CustomNode(InnerNode):
  """
  A CustomNode is an InnerNode defined by the user.
  
  Abstract class for CustomNodes.
  """
  def __init__(self,
               out_builder,
               in_builder,
               is_sequence=False,
               name_prefix=None,
               **dirs):
    """
    CustomNode Initializer
    """
    super(CustomNode, self).__init__(out_builder,
                                     is_sequence,
                                     name_prefix=name_prefix,
                                     **dirs)

    # inner builder
    self.in_builder = in_builder
    self.nodes = in_builder.nodes
    
  def declareOslot(self, oslot, innernode_name, inode_oslot_name):
    """
    Declare an inner oslot as an output to the CustomNode 
    """
    in_builder = self.in_builder
    
    node = in_builder.nodes[innernode_name]
    in_builder.output_nodes[innernode_name] = node 
    in_builder._oslot_to_inner_node_oslot[oslot] = (innernode_name, inode_oslot_name)

  def _build(self):
    """
    """
    raise NotImplementedError("")


class CustomInnerNode(CustomNode):
  """
  """
  def __init__(self,
               out_builder,
               in_builder,
               inputs,
               num_outputs,
               is_sequence=False,
               name=None,
               name_prefix='Cust',
               **dirs):
    """
    """
    # set name
    name_prefix = name_prefix or 'Cust'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)

    # outputs
    self.num_expected_outputs = num_outputs
        
    super(CustomInnerNode, self).__init__(out_builder,
                                          in_builder,
                                          is_sequence,
                                          name_prefix=name_prefix,
                                          **dirs)

    # inputs, TODO: MAKE THIS RIGHT
    self.inputs = inputs
    self.num_expected_inputs = len(self.inputs)
    
    # directives object
    self.directives = NodeDirectives(self.directives)
    self.oslot_names = self.directives.output_names

    # init list of free i/o slots
    self._islot_to_itensor = [{} for _ in range(self.num_expected_inputs)]
    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))

    self._is_built = False
    
  def _update_directives(self, **directives):
    """
    Update the CustomNode directives
    
    TODO: Define a CustomNode directives object
    """
#     this_node_dirs = {'outputname_'+str(i) : 'output'+str(i) for i 
#                       in range(self.num_expected_outputs)}
    this_node_dirs = {'outputname_0' : 'main'}
    this_node_dirs.update({'outputname_' + str(i) : 'sec' + str(i) for i
                           in range(1, self.num_expected_outputs)})
    
    this_node_dirs.update(directives)
    super(CustomInnerNode, self)._update_directives(**this_node_dirs)
    
  def addTransformInner(self, 
                        state_sizes,
                        main_inputs,
                        node_class=DeterministicNN,
                        lmbda=None,
                        name=None,
                        name_prefix=None,
                        **dirs):
    """
    Add an InnerNode to the CustomNode
    
    An inner node is input for Custom if all its inputs are external
    """
    in_builder = self.in_builder
    
    node_name = in_builder.addTransformInner(state_sizes,
                                             main_inputs=main_inputs,
                                             node_class=node_class,
                                             lmbda=lmbda,
                                             is_sequence=False,
                                             name=name,
                                             name_prefix=name_prefix,
                                             **dirs)
    node = in_builder.nodes[node_name]
    if any([isinstance(islot, int) for islot in main_inputs]):
      in_builder.have_external_inputs[node_name] = node
      
    # An inner node is an input for the inner builder if all its inputs are ints
    if all([isinstance(islot, int) for islot in main_inputs]):
      in_builder.input_nodes[node_name] = node
      
    return node_name
  
  def __call__(self, *inputs):
    """
    """
    raise NotImplementedError
        
  def _build(self):
    """
    Build the Custom Node
    
    TODO: Prepare inputs
    TODO: Build output by output
    
    """    
    self.build_outputs()
    
    self._is_built = True 

  def build_outputs(self, **inputs):
    """
    Build the CustomNode's outputs
    """
    self._fill_inner_islots()
      
    rslt = self.in_builder.build_outputs(**inputs)
    
    if not self._is_built:
      for oslot in rslt:
        self.fill_oslot_with_tensor(oslot, rslt[oslot])
        
    return tuple([rslt[oslot] for oslot in range(self.num_expected_outputs)]) 
  
  def prepare_inputs(self, **inputs):
    """
    Prepare the inputs
    
    TODO: Use the islots directive to define main_inputs
    """
    islot_to_itensor = self._islot_to_itensor
    main_inputs = {'imain' + str(i) : islot_to_itensor[i]['main'] for i in 
                  range(self.num_expected_inputs)}
    if inputs:
      print("\tUpdating defaults,", self.name, "with", list(inputs.keys()))
      main_inputs.update(inputs)

    return main_inputs    

  def build_output(self, oslot_name, **inputs):
    """
    """
    inputs = self.prepare_inputs(**inputs)
    
    return self.in_builder.build_output(oslot_name, **inputs)
  
  def _fill_inner_islots(self):
    """
    """
    for islot, elem in enumerate(self._islot_to_itensor):
      for node_name in self.in_builder.islot_to_innernode_names[islot]:
        node = self.in_builder.nodes[node_name]
        inner_islot = node.islot_to_input.index(islot)
        node._islot_to_itensor[inner_islot] = elem
        
  def entropy(self):
    """
    """
    if self.num_expected_outputs > 1:
      raise NotImplementedError("")
    
    try:
      onode = list(self.in_builder.output_nodes.values())[0]
      return onode.entropy()
    except:
      raise AttributeError("Could not define entropy")
      
  def log_prob(self, Y):
    """
    """
    if self.num_expected_outputs > 1:
      raise NotImplementedError("")
    
    try:
      onode = list(self.in_builder.output_nodes.values())[0]
      return onode.log_prob(Y)
    except:
      raise AttributeError("Could not define logprob")
      
        

class CustomCellNode(CustomNode):
  """
  A CustomNode to be used as the transformation of an RNN cell.
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
    """
    # set name
    name_prefix = name_prefix or 'CustCellNode'
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)

    # outputs
    self.num_expected_outputs = num_outputs
        
    super(CustomCellNode, self).__init__(out_builder,
                                         in_builder,
                                         is_sequence,
                                         name_prefix=name_prefix,
                                         **dirs)
    
    # inputs, TODO: MAKE THIS RIGHT
    self.num_expected_inputs = num_inputs
    
    # directives object
    self.directives = NodeDirectives(self.directives)
    self.oslot_names = self.directives.output_names

    # init list of free i/o slots
    self._islot_to_itensor = [{} for _ in range(self.num_expected_inputs)]
    self.free_oslots = list(range(self.num_expected_outputs))
    self.free_islots = list(range(self.num_expected_inputs))

    self._is_built = False

  def addTransformInner(self, 
                        state_sizes,
                        main_inputs,
                        node_class=DeterministicNN,
                        lmbda=None,
                        name=None,
                        name_prefix=None,
                        **dirs):
    """
    Add an InnerNode to the CustomNode
    
    An inner node is input for Custom if all its inputs are external
    """
    in_builder = self.in_builder
    
    node_name = in_builder.addTransformInner(state_sizes,
                                             main_inputs=main_inputs,
                                             node_class=node_class,
                                             lmbda=lmbda,
                                             is_sequence=False,
                                             name=name,
                                             name_prefix=name_prefix,
                                             **dirs)
    node = in_builder.nodes[node_name]
    if any([isinstance(islot, int) for islot in main_inputs]):
      in_builder.have_external_inputs[node_name] = node
      
    # An inner node is an input for the inner builder if all its inputs are ints
    if all([isinstance(islot, int) for islot in main_inputs]):
      in_builder.input_nodes[node_name] = node
      
    return node_name
        
  def __call__(self, *inputs):
    """
    """
    raise NotImplementedError("")
        
  def _build(self):
    """
    """
    raise AttributeError("CustomCellNodes cannot be built")
  
  def build_outputs(self, **inputs):
    """
    Build the CustomNode's outputs
    """
    self._fill_inner_islots()
      
    rslt = self.in_builder.build_outputs(**inputs)
    
    print("CCellNode, rslt", rslt)
    rslt = tuple([rslt[oslot] for oslot in range(self.num_expected_outputs)])
    
    return rslt
  
  def prepare_inputs(self, **inputs):
    """
    Prepare the inputs
    
    TODO: Use the islots directive to define main_inputs
    """
    islot_to_itensor = self._islot_to_itensor
    main_inputs = {'imain' + str(i) : islot_to_itensor[i]['main'] for i in 
                  range(self.num_expected_inputs)}
    if inputs:
      print("\tUpdating defaults,", self.name, "with", list(inputs.keys()))
      main_inputs.update(inputs)

    return main_inputs    

  def build_output(self, oslot_name, **inputs):
    """
    """
    inputs = self.prepare_inputs(**inputs)
    return self.in_builder.build_output(oslot_name, **inputs)
  
  def _fill_inner_islots(self):
    """
    """
    for islot, elem in enumerate(self._islot_to_itensor):
      for node_name in self.in_builder.islot_to_innernode_names[islot]:
        node = self.in_builder.nodes[node_name]
        inner_islot = node.islot_to_input.index(islot)
        node._islot_to_itensor[inner_islot] = elem
            