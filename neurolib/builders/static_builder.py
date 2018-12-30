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
import numpy as np
import tensorflow as tf

from neurolib.builders.builder import Builder
from neurolib.encoder.deterministic import DeterministicNNNode
from neurolib.encoder.anode import ANode
from neurolib.encoder.custom import CustomNode
from neurolib.encoder.input import PlaceholderInputNode  # @UnusedImport
from neurolib.encoder.output import OutputNode
from neurolib.utils.utils import check_name
from neurolib.encoder.basic import CopyNode
from neurolib.utils.graphs import get_session

# pylint: disable=bad-indentation, no-member, protected-access

class StaticBuilder(Builder):
  """
  A StaticBuilder is a Builder for statistical models or nodes that do not
  involve sequential data. In particular, models of time series cannot be built
  using a StaticBuilder.
  
  Building of a static Model through a StaticBuilder is done in two stages:
  Declaration and Construction. In the Declaration stage, the input, output and
  inner nodes of the Model are 'added' to the Model graph (MG), and directed
  links - representing the flow of tensors - are defined between them. In the
  Construction stage, a BFS-like algorithm is called that generates a tensorflow
  graph out of the MG specification
  
  A StaticBuilder defines the following key methods
  
    addOutput(): ...
    
    addInput(): ...
    
    addDirectedLink(): ...
    
    build()
    
  Ex: The following code builds a simple regression Model
      
      builder = StaticBuilder(scope='regression')
      in0 = builder.addInput(input_dim, name="features")
      enc1 = builder.addInner(output_dim, **dirs)
      out0 = builder.addOutput(name="prediction")
      builder.addDirectedLink(in0, enc1)
      builder.addDirectedLink(enc1, out0)

      in1 = builder.addInput(output_dim, name="input_response")
      out1 = builder.addOutput(name="response")
      builder.addDirectedLink(in1, out1)
      
      builder.build()
    
    The 2 input nodes define placeholders for the features and response data
  
  """
  def __init__(self,
               scope,
               batch_size=None):
    """
    Initialize the StaticBuilder
    
    Args:
      scope (str): The tensorflow scope of the Model to be built
      
      batch_size (int or None): The batch size. Defaults to None (unspecified)
    """
    self.custom_encoders = {}
    self.dummies = {}
    self.adj_matrix = None
    self.adj_list = None

    super(StaticBuilder, self).__init__(scope,
                                        batch_size=batch_size)
              
  @check_name
  def addInput(self,
               state_size,
               iclass=PlaceholderInputNode,
               is_sequence=False,
               name=None,
               **dirs):
    """
    Add an InputNode to the Encoder Graph
    
    Args:
      state_size (int): An integer specifying the dimension of the output
      
      iclass (InputNode or str): class of the node

      is_sequence (bool) : Is this node a sequence?
      
      name (str): Unique identifier for the Input Node
      
      dirs (dict): A dictionary of directives for the node
    """
    in_node = iclass(self,
                     state_size,
                     is_sequence=is_sequence,
                     name=name,
                     **dirs)
    name = in_node.name
    self.input_nodes[name] = self.nodes[name] = in_node 
    self._label_to_node[in_node.label] = in_node
    
    return name
    
  @check_name
  def addOutput(self, name=None):
    """
    Add an OutputNode to the Encoder Graph
    
    Args:
      name (str): Unique identifier for the Output Node
    """
    out_node = OutputNode(self,
                          name=name)
    name = out_node.name
    self.output_nodes[name] = self.nodes[name] = out_node 
    self._label_to_node[out_node.label] = out_node
    
    return name
  
  @check_name
  def addCopyNode(self, name=None):
    """
    Add a Copy Node
    """
    c_node = CopyNode(self,
                      name=name)
    name = c_node.name
    self.nodes[name] = c_node
    self._label_to_node[c_node.label] = c_node
    
    return name
    
  def addDirectedLink(self, node1, node2, oslot=0, islot=0, name=None):
    """
    Add directed links to the Encoder graph. 
    
    A) Deal with different item types. The client may provide as arguments,
    either EncoderNodes or integers. Get the EncoderNodes in the latter case
 
    B) Check that the provided oslot for node1 is free. Otherwise, raise an
    exception.
    
    C) Initialize/Add dimensions to the graph representations stored in the
    builder. Specifically, the first time a DirectedLink is added an adjacency
    matrix and an adjacency list are created. From then on, the appropriate
    number of dimensions are added to these representations.

    D) Update the representations to represent the new link. 
    
    E) Fill the all important dictionaries _child_to_oslot and _parent_to_islot.
    For node._child_to_oslot[key] = value, key represents the labels of the
    children of node, while the values are the indices of the oslot in node
    that outputs to that child. Analogously, in node._parent_to_islot[key] =
    value, the keys are the labels of the parents of node and the values are the
    input slots in node corresponding to each key.
    
    F) Possibly update the attributes of node2. In particular deal with nodes
    whose output shapes are dynamically inferred. This is important for nodes such
    as CloneNode and ConcatNode whose output shapes are not provided at
    creation. Once these nodes gather their inputs, they can infer their
    output_shape at this stage.
    
    Args:
      node1 (ANode or str): Node from which the edge emanates
      
      node2 (ANode or str): Node to which the edge arrives
       
      oslot (int): Output slot in node1
      
      islot (int): Input slot in node2
    """
    # A
    if isinstance(node1, str):
      node1 = self.nodes[node1]
    if isinstance(node2, str):
      node2 = self.nodes[node2]
    if not (isinstance(node1, ANode) and isinstance(node2, ANode)):
      raise TypeError("Args node1 and node2 must be either of type `str` "
                      "or type `ANode`")
    
    # B
    nnodes = self.num_nodes
    if not node1._oslot_to_shape:
      if isinstance(node1, OutputNode):
        raise ValueError("Outgoing directed links cannot be defined for "
                         "OutputNodes")
      else:
        raise ValueError("Node1 appears to have no outputs. This software has "
                         "no clue why that would be.\n Please report to my "
                         "master.")
    elif oslot not in node1._oslot_to_shape:
      print("oslot, node1._oslot_to_shape:", oslot, node1._oslot_to_shape)
      raise KeyError("The requested oslot has not been found. Inferring this "
                     "oslot shape may require knowledge of the shape of its "
                     "inputs. In that case, all the inputs for this node must "
                     "be declared")
    if islot in node2._islot_to_shape:
      raise AttributeError("That input slot is already occupied. Assign to "
                           "a different islot")

    # C
    print('Adding dlink:', node1.name, ' -> ', node2.name)
    if self.adj_matrix is None:
      self.adj_matrix = [[0]*nnodes for _ in range(nnodes)]
      self.adj_list = [[] for _ in range(nnodes)]
    else:
      if nnodes > len(self.adj_matrix):
        l = len(self.adj_matrix)
        for row in range(l):
          self.adj_matrix[row].extend([0]*(nnodes-l))
        for _ in range(nnodes-l):
          self.adj_matrix.append([0]*nnodes)
          self.adj_list.append([])
    
    # D
    self.adj_matrix[node1.label][node2.label] = 1
    self.adj_list[node1.label].append(node2.label)
      
    # E
    if node1.num_expected_outputs > 1:
      if oslot is None:
        raise ValueError("The in-node has more than one output slot, so pairing "
                         "to the out-node is ambiguous.\n You must specify the "
                         "output slot. The declared output slots for node 1 are: ",
                         node1._oslot_to_shape)
    if node2.num_expected_inputs > 1:
      if islot is None:
        raise ValueError("The out-node has more than one input slot, so pairing "
                         "from the in-node is ambiguous.\n You must specify the " 
                         "input slot")
    exchanged_shape = node1._oslot_to_shape[oslot]
    node1._child_label_to_oslot[node2.label] = oslot

    if oslot in node1.free_oslots:
      node1.num_declared_outputs += 1
      node1.free_oslots.remove(oslot)
    
    node2._islot_to_shape[islot] = exchanged_shape
    node2._parent_label_to_islot[node1.label] = islot
    node2.num_declared_inputs += 1
    node2.free_islots.remove(islot)
    node2._islot_to_name[islot] = (node1._oslot_to_name[oslot] if name is None
                                   else name)

    # Initialize _built_parents for the child node.
    node2._built_parents[node1.label] = False
    
    # F
    node1.update_when_linked_as_node1()
    node2.update_when_linked_as_node2()
      
  def addMergeNode(self,
                   state_size,
                   nodes,
                   merge_class=None,
                   **dirs):
    """
    Merge two or more nodes
    """
    print("st bld; nodes:", nodes)
    nodes = [self.nodes[node_name] for node_name in nodes]
    merger = merge_class(self,
                         state_size,
                         nodes,
                         **dirs)
    name = merger.name
    self.nodes[name] = merger    
    
    nnodes = self.num_nodes
    if nnodes > len(self.adj_matrix):
        l = len(self.adj_matrix)
        for row in range(l):
          self.adj_matrix[row].extend([0]*(nnodes-l))
        for _ in range(nnodes-l):
          self.adj_matrix.append([0]*nnodes)
          self.adj_list.append([])
    for node in nodes:
      self.adj_matrix[node.label][merger.label] = 1
      self.adj_list[node.label].append(merger.label)
      merger._built_parents[node.label] = False

    return name
  
  def check_graph_correctness(self):
    """
    Checks the graph declared so far. 
    
    TODO:
    """
    pass
        
  def createCustomNode(self,
                       num_inputs,
                       num_outputs,
                       is_sequence=False,
                       name=None):
    """
    Create a CustomNode
    """
    # Define here to avoid circular dependencies
    custom_builder = StaticBuilder(scope=name,
                                   batch_size=self.batch_size)
    cust = CustomNode(self,
                      custom_builder,
                      num_inputs,
                      num_outputs,
                      is_sequence=is_sequence,
                      name=name)
    name = cust.name
    self.custom_encoders[name] = self.nodes[name] = cust
    self._label_to_node[cust.label] = cust
    
    return cust
  
  def get_custom_encoder(self, name):
    """
    Get a CustomNode by name
    """
    return self.custom_encoders[name] 
  
  def add_to_custom(self,
                    custom_node,
                    output_shapes,
                    name=None,
                    node_class=DeterministicNNNode,
                    **dirs):
    """
    Add an InnerNode to a CustomNode
    """
    custom_node.builder.addInner(output_shapes,
                                 name=name,
                                 node_class=node_class,
                                 **dirs)

  def get_label_from_name(self, name):
    """
    Get the label of a node from name
    """
    return self.nodes[name].label
  
  def get_output(self,
                 inputs=None,
                 islot_to_itensor=None,
                 custom_node=None,
                 scope_suffix=None):
    """
    Get the output for this node from a set of inputs.
    
    This method follows the directed links in the Model Graph (MG) in BFS
    fashion to construct the tensorflow graph. The method behaves slightly
    different for the case in which self is the inside Builder of a CustomNode.
    In that case, the method must allows for sampling from the CustomNode. These
    two cases may be different enough that they may need to be implemented
    separately in a future version.
    
    The algorithm goes through the following stages:
    
    A. If custom_node is provided:
      A.1. Loop over the islots of inode and get the corresponding islots
          of the parent custom_node.
      A.2. Assign the inputs to custom_node._islot_to_itensor

    * For inode in self.input_nodes:
        B. Add inode to the queue
        
      * Start main BFS loop:
          B. Pop node from the queue. Set visited[node.label] to True

          D. Build the popped node 
            
          * For child_node in node's children:
              E. Set node in child_node._buils_parents to True
              F. Get the endslots of the parent and child
              G. Fill the inputs of the child node
                G.1. If child_node is a CustomNode, also match the input of the
                    islot with the input of the inner_islot
              H. Append to the queue if all parents of child_node have been built
        
          I. If custom_node is provided and node is an OutputNode
            I.1. Loop over oslots of node and get the corresponding oslots and
                otensors of custom_node. Build the dictionary of results
    """      
    result = None
    # I THINK THAT IN THIS METHOD ONLY INPUT DICTS SHOULD BE ALLOWED
    if custom_node is not None:
      try:
        _input = dict(list(enumerate(inputs)))
      except TypeError:
        _input = islot_to_itensor
#       if not isinstance(inputs, dict):
      result = {}

    scope_suffix = "" if scope_suffix is None else "_" + scope_suffix
    with tf.variable_scope(self.scope + scope_suffix, reuse=tf.AUTO_REUSE): 
      # A.
      if custom_node is not None:
#         for inner_node_data, islot in custom_node._islot_to_inner_node_islot.inv.items():
#           inode_name, inode_islot = inner_node_data
#           self.nodes[inode_name]._islot_to_itensor[inode_islot] = _input[islot]
        for islot, inner_node_data in custom_node._islot_to_inner_node_islot.items():
          for inode_name, inode_islot in inner_node_data:
            self.nodes[inode_name]._islot_to_itensor[inode_islot] = _input[islot]
          

      visited = [False for _ in range(self.num_nodes)]
      queue = []
      for cur_inode_name in self.input_nodes:
        # B.
        cur_inode_label = self.get_label_from_name(cur_inode_name)
        queue.append(cur_inode_label)
        while queue:
          # C.
          cur_node_label = queue.pop(0)
          visited[cur_node_label] = True
          cur_node = self._label_to_node[cur_node_label]

          # D.
          cur_node._build()
          
          # Update the oslots of the children of cur_node
          for child_label in self.adj_list[cur_node_label]:
            # E.
            child_node = self._label_to_node[child_label]
            child_node._built_parents[cur_node_label] = True
            
            # F.
            oslot = cur_node._child_label_to_oslot[child_label]
            islot = child_node._parent_label_to_islot[cur_node_label]
            
            # G.
#             print((child_node.name, islot), (cur_node.name, oslot))
#             print(cur_node.get_outputs()[oslot])
            child_node._islot_to_itensor[islot] = cur_node.get_outputs()[oslot]
            if isinstance(child_node, CustomNode):
#               enc_name, enc_islot = child_node._islot_to_inner_node_islot[islot]
#               enc = child_node.in_builder.nodes[enc_name]
#               enc._islot_to_itensor[enc_islot] = cur_node.get_outputs()[oslot]
              for enc_name, enc_islot in child_node._islot_to_inner_node_islot[islot]:
                enc = child_node.in_builder.nodes[enc_name]
                enc._islot_to_itensor[enc_islot] = cur_node.get_outputs()[oslot]
            
            # H.
            if isinstance(child_node, OutputNode):
              queue.append(child_node.label)
              continue
            if all(child_node._built_parents.values()):
              queue.append(child_node.label)
          
          # I.
          if custom_node is not None and cur_node.name in self.output_nodes:
            for onode_oslot in cur_node._oslot_to_otensor:
              custom_node_oslot = custom_node._oslot_to_inner_node_oslot.inv[(cur_node.name, onode_oslot)]
              result[custom_node_oslot] = cur_node._oslot_to_otensor[onode_oslot]
              
      if custom_node is not None:
#         print("result.items()", result.items())
        result_as_list = list(zip(*sorted(result.items())))[1]
      else:
        result_as_list = None
      
#     print("result_as_list", result_as_list)
#     print("result", result)
    return result_as_list, result

  def build(self):
    """
    """
    self.get_output()
    
  def make_dummy_fd(self, batch_size):
    """
    """
    print("st_b; self.dummies.items()", self.dummies.items())
    return {key : np.zeros([batch_size] + value[1:]) for key, value in self.dummies.items()}
    
  def eval(self, node, feed_dict, oslot=0, lmbda=None):
    """
    """
    otensor = self.nodes[node].get_outputs()[oslot]
    
    batch_size = list(feed_dict.values())[0].shape[0]
#     print("st_b; list(feed_dict.values())[0].shape", list(feed_dict.values())[0].shape)
    dummy_feed = self.make_dummy_fd(batch_size)
#     print("st_b; dummy_feed", dummy_feed)
    feed_dict.update(dummy_feed)
    
#     print("st_b; feed_dict", feed_dict)
    sess = get_session()
    sess.run(tf.global_variables_initializer())
    if lmbda is not None:
      rslt = sess.run(lmbda(otensor), feed_dict=feed_dict)
    else:
      rslt = sess.run(otensor, feed_dict=feed_dict)
    return rslt
    