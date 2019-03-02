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

from neurolib.encoder.anode import ANode
from neurolib.encoder.input import PlaceholderInputNode

from neurolib.encoder.deterministic import DeterministicNNNode #@UnusedImport
from neurolib.utils.utils import check_name
from neurolib.encoder.deterministic import DeterministicNN
from neurolib.encoder.merge import MergeSeqsNormalwNormalEv

# pylint: disable=bad-indentation, no-member, protected-access

innernode_dict = {'deterministic' : DeterministicNNNode}

class Builder():
  """
  Abstract class for model builders
  """
  innernode_dict = {'deterministic' : DeterministicNNNode}
  def __init__(self,
               scope,
               batch_size=None):
    """
    Initialize the StaticBuilder
    
    Args:
      scope (str): The tensorflow scope of the Model to be built
      
      batch_size (int or None): The batch size. Defaults to None (unspecified)
    """
    self.scope = scope
    self.batch_size = batch_size
    self.num_nodes = 0
    
    # graph representation
    self.custom_encoders = {}
    self.adj_matrix = []
    self.adj_list = []
    self.num_subbuilders = 0
    
    # name to node dicts
    self.nodes = {}
    self.input_nodes = {}
    self._label_to_node = {}

    # for restoring a model
    self.otensor_names = {}
        
  def add_node_to_model_graph(self):
    """
    Add node to the graph representations (adjacency list and adjacency matrix)
    """
    def add_node_adj_matrix():
      l = len(self.adj_matrix)
      for row in range(l):
        self.adj_matrix[row].extend([0])
      self.adj_matrix.append([0]*(l+1))
    def add_node_adj_list():
      self.adj_list.append([])  
    
    add_node_adj_matrix()
    add_node_adj_list()
  
  @check_name
  def addExternalNode(self,
                      node,
                      name=None):
    """
    """
    self.add_node_to_model_graph()
    
    self.ext_builder[name] = node
    self._label_to_node[self.num_labels] = node
    self.num_labels += 1
    
    return name
  
  @check_name
  def addInput(self,
               state_size,
               iclass=PlaceholderInputNode,
               is_sequence=False,
               name=None,
               **dirs):
    """
    TODO: Change iclass arg to node_class for consistency  
    
    Add an InputNode to the Encoder Graph
    
    Args:
      state_size (int): An integer specifying the dimension of the output
      
      iclass (InputNode or str): class of the node

      is_sequence (bool) : Is this node a sequence?
      
      name (str): Unique identifier for the Input Node
      
      dirs (dict): A dictionary of directives for the node
    """
    self.add_node_to_model_graph()
    
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
  def addTransformInner(self,
                        state_size,
                        main_inputs,
                        lmbda=None,
                        node_class=DeterministicNN,
                        name=None,
                        name_prefix=None,
                        **dirs):
    """
    Add a Transform InnerNode
    """
    if isinstance(main_inputs, str):
      main_inputs = [main_inputs]
    num_inputs = len(main_inputs)
    if len(main_inputs) < 1:
      raise ValueError("`InnerNodes must have at least one input "
                       "(`num_inputs = {}`".format(num_inputs))
    
    self.add_node_to_model_graph()
    
    if isinstance(node_class, str):
      node_class = self.innernode_dict[node_class]
    print("builder, main_inputs", main_inputs)
    enc_node = node_class(self,
                          state_size,
                          main_inputs=main_inputs,
                          lmbda=lmbda,
                          name=name,
                          name_prefix=name_prefix,
                          **dirs)
    self.nodes[enc_node.name] = self._label_to_node[enc_node.label] = enc_node
    
    self.add_directed_links(main_inputs, enc_node)
      
    return enc_node.name
    
  @check_name
  def addInner(self,
               state_sizes,
               *args,
               num_inputs=1,
               node_class=DeterministicNNNode,
               is_sequence=False,
               name=None,
               **dirs):
    """
    Add an InnerNode to the Encoder Graph
    
    Args:
      state_sizes (int or list of list of int) : For a single output, the
          dimension of the output. For more than one output, a list of list of
          ints where `state_sizes[ot]` are the dimensions of the output
          corresponding to the oslot `ot`.
      
      num_inputs (int) : The number of inputs to this node.
      
      node_class (InnerNode or str): class of the node
      
      is_sequence (bool) : Does this node represent a sequence?
      
      name (str): A unique string identifier for the node being added to the MG
      
      dirs (dict): A dictionary of directives for the node
    """
    if num_inputs < 1:
      raise ValueError("`InnerNodes must have at least one input "
                       "(`num_inputs = {}`".format(num_inputs))
    
    self.add_node_to_model_graph()
    
    if isinstance(node_class, str):
      node_class = self.innernode_dict[node_class]
    enc_node = node_class(self,
                          state_sizes,
                          *args,
                          num_inputs=num_inputs,
                          is_sequence=is_sequence,
                          name=name,
                          **dirs)
      
    self.nodes[enc_node.name] = self._label_to_node[enc_node.label] = enc_node
      
    return enc_node.name

  def addMergeSeqwDS(self,
                     seq_inputs,
                     ds_inputs,
                     prior_inputs,
                     merge_class=MergeSeqsNormalwNormalEv,
                     name=None,
                     name_prefix=None,
                     **dirs):
    """
    """
    if isinstance(seq_inputs, str):
      seq_inputs = [seq_inputs]
    if isinstance(ds_inputs, str):
      ds_inputs = [ds_inputs]
    if isinstance(prior_inputs, str):
      prior_inputs = [prior_inputs]
    
    self.add_node_to_model_graph()
    
    merger = merge_class(self,
                         seq_inputs=seq_inputs,
                         ds_inputs=ds_inputs,
                         prior_inputs=prior_inputs,
                         name=name,
                         name_prefix=name_prefix,
                         **dirs)
    name = merger.name
    self.nodes[name] = merger
    self._label_to_node[merger.label] = merger
    
    self.add_directed_links(seq_inputs, merger)
    nsofar = len(seq_inputs)
    self.add_directed_links(ds_inputs, merger, nsofar)
    nsofar += len(ds_inputs)
    self.add_directed_links(prior_inputs, merger, nsofar)
    
    return name
    
  def addMergeNode(self,
                   node_list=None,
                   node_dict=None,
                   parents_to_oslot_tuples=None,
                   merge_class=None,
                   name=None,
                   **dirs):
    """
    Merge two or more nodes
    """
    self.add_node_to_model_graph()
    
    merger = merge_class(self,
                         node_list=node_list,
                         node_dict=node_dict,
                         parents_to_oslot_tuples=parents_to_oslot_tuples,
                         name=name,
                         **dirs)
    name = merger.name
    self.nodes[name] = merger
    self._label_to_node[merger.label] = merger
    
    return name
  
  def addDirectedLink(self, node1, node2, islot):
    """
    Add directed links to the Model Graph
    """
    if isinstance(node1, str):
      node1 = self.nodes[node1]
    if isinstance(node2, str):
      node2 = self.nodes[node2]
    if not (isinstance(node1, ANode) and isinstance(node2, ANode)):
      raise TypeError("Args node1 and node2 must be either of type `str` "
                      "or type `ANode`")
      
    if islot > node2.num_expected_inputs - 1:
      raise ValueError("`islot` %d out of range (`num_expected_inputs = %d)"
                       "" %(islot, node2.num_expected_inputs))
    if islot not in node2.free_islots:
      raise ValueError("`islot` %d has already been assigned." %islot)
      
    self.adj_matrix[node1.label][node2.label] = 1
    if node2.label not in self.adj_list[node1.label]: 
      self.adj_list[node1.label].append(node2.label)
      
    for oname in node1.oslot_names:
      node1._child_label_to_slot_pairs[node2.label].append((oname, islot))
    
    node2.free_islots.remove(islot)

    # Initialize _built_parents for the child node.
    node2._built_parents[node1.label] = False
    
    # cleanup
    node1.update_when_linked_as_node1()
    node2.update_when_linked_as_node2()
    
  def add_directed_links(self, inputs, node, start_islot=0):
    """
    """
    for i, inode in enumerate(inputs, start_islot):
      self.addDirectedLink(inode, node, i)
  
  def make_dummy_fd(self, batch_size):
    """
    Make the feed dict for the dummies (batch size, for instance) of the Model 
    """
    return {self.dummies[key] : np.array([batch_size], dtype=np.int32) for key 
            in self.dummies}
  
  def check_graph_correctness(self):
    """
    Checks the graph declared so far. 
    
    TODO:
    """
    pass
        
  def get_custom_encoder(self, name):
    """
    Get a CustomNode by name
    """
    return self.custom_encoders[name] 

  def get_label_from_name(self, name):
    """
    Get the label of a node from name
    """
    return self.nodes[name].label
  
  def add_to_output_names(self, name, tensor):
    """
    Add a tensor to the list of names available on restore
    """
    self.otensor_names[name] = tensor.name
    
  