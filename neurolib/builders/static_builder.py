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
from collections import defaultdict

from bidict import bidict

import tensorflow as tf

from neurolib.builders.builder import Builder
from neurolib.encoder.anode import ANode
from neurolib.encoder.deterministic import DeterministicNNNode, DeterministicNN
from neurolib.encoder.custom import CustomNode, CustomNode2
from neurolib.utils.utils import check_name

# pylint: disable=bad-indentation, no-member, protected-access

class StaticBuilder(Builder):
  """
  A Builder for statistical models that do not involve sequential data.
  
  A static Model is built through a StaticBuilder in two stages:
  Declaration and Construction. In the Declaration stage, the input, output and
  inner nodes of the Model are "added" to the Model graph (MG), and directed
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
  def __init__(self, scope, batch_size=None):
    """
    Initialize the StaticBuilder
    
    # TODO: The proper way to define a batch_size is to declare it a TensorShape
    """
    super(StaticBuilder, self).__init__(scope, batch_size)

    # dummies
    self.dummies = {}
    if self.batch_size is None:
      self.dummy_bsz = tf.placeholder(tf.int32, [None], 'dummy_bsz')
      self.dummies['dummy_bsz'] = self.dummy_bsz.name
  
  def createCustomNode(self,
                       num_inputs,
                       num_outputs,
                       is_sequence=False,
                       name=None,
                       **dirs):
    """
    Create a CustomNode
    """
    self.add_node_to_model_graph()
    
    # Define here to avoid circular imports
    custom_builder = CustomNodeBuilder(scope=name,
                                       batch_size=self.batch_size,
                                       dummy_bsz=self.dummy_bsz)
    cust = CustomNode(self,
                      custom_builder,
                      num_inputs,
                      num_outputs,
                      is_sequence=is_sequence,
                      name=name,
                      **dirs)
    name = cust.name
    self.custom_encoders[name] = self.nodes[name] = cust
    self._label_to_node[cust.label] = cust
    
    return cust
  
  def createCustomNode2(self,
                        inputs,
                        num_outputs,
                        is_sequence=False,
                        name=None,
                        **dirs):
    """
    Create a CustomNode
    """
    if isinstance(inputs, str):
      inputs = [inputs]
    num_inputs = len(inputs)
    
    self.add_node_to_model_graph()
    
    # Define here to avoid circular imports
    custom_builder = CustomNodeBuilder2(scope=name,
                                        ext_builder=self,
                                        num_inputs=num_inputs,
                                        batch_size=self.batch_size,
                                        dummy_bsz=self.dummy_bsz)
    cust = CustomNode2(self,
                       custom_builder,
                       inputs,
                       num_outputs,
                       is_sequence=is_sequence,
                       name=name,
                       **dirs)
    name = cust.name
    self.custom_encoders[name] = self.nodes[name] = cust
    self._label_to_node[cust.label] = cust
    
    self.add_directed_links(inputs, cust)
    
    return cust

  
  def build(self,
             scope_suffix=None):
    """
    Build the Model
    
    TODO: Check that every node's input have been provided. The code compiles
    even if they haven't!
    """
    scope_suffix = "" if scope_suffix is None else "_" + scope_suffix
    with tf.variable_scope(self.scope + scope_suffix, reuse=tf.AUTO_REUSE):
      # init BFS
      visited = [False for _ in range(self.num_nodes)]
      queue = []
      
      # start at every input node
      for cur_inode_name in self.input_nodes:
        cur_inode_label = self.get_label_from_name(cur_inode_name)
        queue.append(cur_inode_label)
        while queue:
          cur_node_label = queue.pop(0)
          cur_node = self._label_to_node[cur_node_label]
          
          # Build the current node
          cur_node._build()
          visited[cur_node_label] = True

          # Update the oslots of the children of cur_node
          for child_label in self.adj_list[cur_node_label]:
            child_node = self._label_to_node[child_label]
            
            # A parent of this child has been built
            child_node._built_parents[cur_node_label] = True
            
            # assign cur_node output tensor to a child islot...
            for oname, islot in cur_node._child_label_to_slot_pairs[child_label]:
              child_node._islot_to_itensor[islot][oname] = cur_node.get_output_tensor(oname)
            
            # once all parents of a child have been built, add to bfs queue
            if all(child_node._built_parents.values()):
              queue.append(child_node.label)
          
  def get_node_output(self, node_name, oslot='main'):
    """
    Get output tensor
    """
    return self.nodes[node_name].get_output_tensor(oslot)
  
  def eval(self,
           sess,
           otensor,
           feed_dict=None,
           lmbda=None):
    """
    Evaluate a tensor
    """
    if feed_dict is None:
      feed_dict = {}
      batch_size = 1
    else:
      batch_size = list(feed_dict.values())[0].shape[0]
    dummy_feed = self.make_dummy_fd(batch_size)
    feed_dict.update(dummy_feed)
#     print("feed_dict", feed_dict)

    if lmbda is not None:
      rslt = sess.run(lmbda(otensor), feed_dict=feed_dict)
    else:
      rslt = sess.run(otensor, feed_dict=feed_dict)
    return rslt
  
  def make_shapes_compatible_or_puke(self, feed_dict):
    """
    Make shapes compatible or puke! 
    """
    def are_shapes_pseudocompatible(tfshape, arshape):
      """
      """
      raise NotImplementedError
    
    def make_array_shape_compatible(tfshape, array):
      """
      """
      raise NotImplementedError
      
    for tname in feed_dict:
      tensor = tf.get_default_graph().get_tensor_by_name(tname)
      tfshape, arshape = tensor.shape, feed_dict[tname].shape
#       print("tfshape, arshape", tfshape, arshape, tfshape.is_compatible_with(arshape))
      if tfshape.is_compatible_with(arshape):
        continue
      elif are_shapes_pseudocompatible(tfshape, arshape):
        feed_dict[tname] = make_array_shape_compatible(tfshape, feed_dict[tname])
      else:
        raise ValueError("")
      
  def eval_node_oslot(self,
                      sess,
                      node,
                      oslot='main',
                      feed_dict=None,
                      inputs=None,
                      lmbda=None):
    """
    Evaluate the output tensor of a node
    
    TODO:
      - Automatic handling of feed_dict;
      - Call to node.get_outputs if inputs is not None? (adds to the graph after building, ugh)
      - Smart handling of shapes. 
    """
    if feed_dict is not None:
      feed_dict = dict(feed_dict) # make a copy of user provided fd
      self.make_shapes_compatible_or_puke(feed_dict)
    otensor = self.nodes[node].get_output_tensor(oslot)
    
    
    return self.eval(sess, otensor,
                     feed_dict=feed_dict,
                     lmbda=lmbda)
  
  
class CustomNodeBuilder2(StaticBuilder):
  """
  A builder for building CustomNodes
  """
  def __init__(self,
               scope,
               ext_builder,
               num_inputs,
               batch_size=None,
               dummy_bsz=None):
    """
    Initialize the CustomNodeBuilder
    """
    super(CustomNodeBuilder2, self).__init__(scope,
                                             batch_size)
    
    # dummies
    self.dummies = {}
    if dummy_bsz is None:
      if self.batch_size is None:
        self.dummy_bsz = tf.placeholder(tf.int32, [None], 'dummy_bsz')
    else:
      self.dummy_bsz = dummy_bsz
    self.dummies['dummy_bsz'] = self.dummy_bsz.name
    
    # dicts for access and talking to the outer nodes
    self.output_nodes = {}
    self._innernode_to_its_avlble_islots = {}
    self._innernode_to_its_avlble_oslots = {}
    self._islot_to_inner_node_islot = defaultdict(list)
    self._oslot_to_inner_node_oslot = bidict()
    
    # new
    self.num_inputs = num_inputs
    self.islot_to_innernode_names = [[] for _ in range(num_inputs)]
    
    ##
    self.ext_builder = ext_builder
    
#   @check_name
#   def addTransformInner(self, 
#                         state_size,
#                         main_inputs, 
#                         lmbda=None,
#                         node_class=DeterministicNN, 
#                         name=None, 
#                         name_prefix=None,
#                         **dirs):
#     """
#     """
#     return StaticBuilder.addTransformInner(self,
#                                            state_size,
#                                            main_inputs,
#                                            lmbda=lmbda,
#                                            node_class=node_class,
#                                            name=name,
#                                            name_prefix=name_prefix,
#                                            **dirs)

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
                          num_inputs=num_inputs,
                          is_sequence=is_sequence,
                          name=name,
                          **dirs)
    self.input_nodes[enc_node.name] = enc_node
    self.nodes[enc_node.name] = self._label_to_node[enc_node.label] = enc_node
      
    return enc_node.name

  def addDirectedLink(self, node1, node2, islot):
    """
    Add directed links to the Model Graph
    
    TODO: Is this different from the parent?
    """
    if isinstance(node1, str):
      node1 = self.nodes[node1]
    if isinstance(node2, str):
      node2 = self.nodes[node2]
    if not (isinstance(node1, ANode) and isinstance(node2, ANode)):
      raise TypeError("Args node1 and node2 must be either of type `str` "
                      "or type `ANode`")
      
    if islot > node2.num_expected_inputs - 1:
      raise ValueError("`islot` {} out of range (`num_expected_inputs = {})"
                       "".format(islot, node2.num_expected_inputs))
    if islot not in node2.free_islots:
      raise ValueError("`islot` {} has already been assigned.".format(islot))
      
    self.adj_matrix[node1.label][node2.label] = 1
    if node2.label not in self.adj_list[node1.label]: 
      self.adj_list[node1.label].append(node2.label)
      
    for oname in node1.oslot_names:
      node1._child_label_to_slot_pairs[node2.label].append((oname, islot))
    
    node2.free_islots.remove(islot)

    # Initialize _built_parents for the child node.
    node2._built_parents[node1.label] = False
    
    # Remove node2 from input_nodes
#     self.input_nodes.pop(node2.name)
    
    # cleanup
    node1.update_when_linked_as_node1()
    node2.update_when_linked_as_node2()
    
  def build_outputs2(self,
                     **inputs):
    """
    Once this method is called, the CustomNode is in possession of its final
    _islot_to_itensor AND each InnerNode of the CustomNode is in possession of a
    partial _islot_to_itensor. In particular, the inner InputNodes have a
    complete _islot_to_itensor
    """
    result = {}
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      # init bfs
      visited = [False for _ in range(self.num_nodes)]
      queue = []
      for cur_inode_name in self.input_nodes:
        cur_inode_label = self.get_label_from_name(cur_inode_name)
        queue.append(cur_inode_label)
        while queue:
          cur_node_label = queue.pop(0)
          visited[cur_node_label] = True
          cur_node = self._label_to_node[cur_node_label]
            
          # fill `cur_node._oslot_to_otensor
#           cur_node._build()
          if cur_inode_name in self.input_nodes:
            cur_node.build_outputs(**inputs)
          else:
            cur_node._build()
          
          # fetch children 
          for child_label in self.adj_list[cur_node_label]:
            child_node = self._label_to_node[child_label]
            
            # A parent of this child has been built
            child_node._built_parents[cur_node_label] = True

            # assign cur_node output tensor to a child islot...
            for oname, islot in cur_node._child_label_to_slot_pairs[child_label]:
              child_node._islot_to_itensor[islot][oname] = cur_node.get_output_tensor(oname)
            
            # once all parents of a child have been built, add to bfs queue
            if all(child_node._built_parents.values()):
              queue.append(child_node.label)

          # fill the output of Custom fetching from inner nodes `_oslot_to_otensor` attributes 
          if cur_node.name in self.output_nodes:
            for onode_oslot in cur_node._oslot_to_otensor:
              if (cur_node.name, onode_oslot) in self._oslot_to_inner_node_oslot.inv: 
                custom_node_oslot = self._oslot_to_inner_node_oslot.inv[(cur_node.name, onode_oslot)]
                result[custom_node_oslot] = cur_node._oslot_to_otensor[onode_oslot]
                    
    # TODO: Add attribute output_names to CustomBuilder                    
#     return tuple([result[oslot] for oslot in self.output_names])
    return result
          
                
  def build_outputs(self,
                    islot_to_itensor,
                    output_names):
    """
    Build outputs for the associated Custom Node
    
    TODO: This function has many problems. The most glaring one is that it does
    not behave as the regular `build_outputs` function from InnerNodes. As is is written, it cannot bu
    """
    print("\ncust_builder; self.input_nodes", self.input_nodes)
    
    result = {}
    _input = islot_to_itensor
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      # fill the inner nodes islot_to_itensors with the input 
      for islot in self._islot_to_inner_node_islot:
        inner_node_data = self._islot_to_inner_node_islot[islot]
        for inode_name, inode_islot in inner_node_data:
          print("inode_name, inode_islot", inode_name, inode_islot)
          print("_input", _input)
          self.nodes[inode_name]._islot_to_itensor[inode_islot] = _input[islot]

      # init bfs
      visited = [False for _ in range(self.num_nodes)]
      queue = []
      for cur_inode_name in self.input_nodes:
        cur_inode_label = self.get_label_from_name(cur_inode_name)
        queue.append(cur_inode_label)
        while queue:
          cur_node_label = queue.pop(0)
          visited[cur_node_label] = True
          cur_node = self._label_to_node[cur_node_label]
          
          # build current inner node
          print("sb; cur_node._islot_to_itensor", cur_node.name, cur_node._islot_to_itensor)
          cur_node._build()
          print("sb; cur_node._oslot_to_otensor", cur_node.name, cur_node._oslot_to_otensor)
          
          # fetch children 
          for child_label in self.adj_list[cur_node_label]:
            # mark current node as visited for its children
            child_node = self._label_to_node[child_label]
            child_node._built_parents[cur_node_label] = True
            
            # fill child `_islot_to_itensor` attribute
            for oslot, islot in cur_node._child_label_to_slot_pairs[child_label]:
              child_node._islot_to_itensor[islot][oslot] = cur_node.get_output_tensor(oslot)

            # if all its parents are built, add to bfs queue
            if all(child_node._built_parents.values()):
              queue.append(child_node.label)
          
          # fill result with the inner node `_oslot_to_otensor` attribute 
          if cur_node.name in self.output_nodes:
            for onode_oslot in cur_node._oslot_to_otensor:
              if (cur_node.name, onode_oslot) in self._oslot_to_inner_node_oslot.inv: 
                custom_node_oslot = self._oslot_to_inner_node_oslot.inv[(cur_node.name, onode_oslot)]
                result[custom_node_oslot] = cur_node._oslot_to_otensor[onode_oslot]
                    
    return tuple([result[oslot] for oslot in output_names])
  
  def add_directed_links(self, inputs, node, start_islot=0):
    """
    """
    for i, _input in enumerate(inputs, start_islot):
      if isinstance(_input, int):
#         inode = self.ext_builder[inode_name]
#         self.declareIslot(_input, node)
        self.declareIslot(_input, node.name)
      else:
        self.addDirectedLink(_input, node, i)

#   def declareIslot(self, islot, innernode_name, inode_islot):
  def declareIslot(self, islot, innernode_name):
    """
    Declare an inner islot as an input to the CustomNode
    """
#     in_builder = self.in_builder
#     self._islot_to_inner_node_islot[islot].append((innernode_name, inode_islot)) 
    self.islot_to_innernode_names[islot].append(innernode_name) 
    
    
class CustomNodeBuilder(StaticBuilder):
  """
  A builder for building CustomNodes
  """
  def __init__(self,
               scope,
               batch_size=None,
               dummy_bsz=None):
    """
    Initialize the CustomNodeBuilder
    """
    super(CustomNodeBuilder, self).__init__(scope,
                                            batch_size)
    
    # dummies
    self.dummies = {}
    if dummy_bsz is None:
      if self.batch_size is None:
        self.dummy_bsz = tf.placeholder(tf.int32, [None], 'dummy_bsz')
    else:
      self.dummy_bsz = dummy_bsz
    self.dummies['dummy_bsz'] = self.dummy_bsz.name
    
    # dicts for access and talking to the outer nodes
    self.output_nodes = {}
    self._innernode_to_its_avlble_islots = {}
    self._innernode_to_its_avlble_oslots = {}
    self._islot_to_inner_node_islot = defaultdict(list)
    self._oslot_to_inner_node_oslot = bidict()
    
#   @check_name
#   def addTransformInner(self, 
#                         state_size,
#                         main_inputs, 
#                         lmbda=None,
#                         node_class=DeterministicNN, 
#                         name=None, 
#                         name_prefix=None,
#                         **dirs):
#     """
#     """
#     return StaticBuilder.addTransformInner(self,
#                                            state_size,
#                                            main_inputs,
#                                            lmbda=lmbda,
#                                            node_class=node_class,
#                                            name=name,
#                                            name_prefix=name_prefix,
#                                            **dirs)

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
                          num_inputs=num_inputs,
                          is_sequence=is_sequence,
                          name=name,
                          **dirs)
    self.input_nodes[enc_node.name] = enc_node
    self.nodes[enc_node.name] = self._label_to_node[enc_node.label] = enc_node
      
    return enc_node.name

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
      raise ValueError("`islot` {} out of range (`num_expected_inputs = {})"
                       "".format(islot, node2.num_expected_inputs))
    if islot not in node2.free_islots:
      raise ValueError("`islot` {} has already been assigned.".format(islot))
      
    self.adj_matrix[node1.label][node2.label] = 1
    if node2.label not in self.adj_list[node1.label]: 
      self.adj_list[node1.label].append(node2.label)
      
    for oname in node1.oslot_names:
      node1._child_label_to_slot_pairs[node2.label].append((oname, islot))
    
    node2.free_islots.remove(islot)

    # Initialize _built_parents for the child node.
    node2._built_parents[node1.label] = False
    
    # Remove node2 from input_nodes
    self.input_nodes.pop(node2.name)
    
    # cleanup
    node1.update_when_linked_as_node1()
    node2.update_when_linked_as_node2()
    
  def build_outputs2(self,
                     **inputs):
    """
    """
    result = {}
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      # init bfs
      visited = [False for _ in range(self.num_nodes)]
      queue = []
      for cur_inode_name in self.input_nodes:
        cur_inode_label = self.get_label_from_name(cur_inode_name)
        queue.append(cur_inode_label)
        while queue:
          cur_node_label = queue.pop(0)
          visited[cur_node_label] = True
          cur_node = self._label_to_node[cur_node_label]
            
          # fill `cur_node._oslot_to_otensor
#           cur_node._build()
          if cur_inode_name in self.input_nodes:
            cur_node._build_outputs(**inputs)
          else:
            cur_node._build()
          
          # fetch children 
          for child_label in self.adj_list[cur_node_label]:
            child_node = self._label_to_node[child_label]
            
            # A parent of this child has been built
            child_node._built_parents[cur_node_label] = True

            # assign cur_node output tensor to a child islot...
            for oname, islot in cur_node._child_label_to_slot_pairs[child_label]:
              child_node._islot_to_itensor[islot][oname] = cur_node.get_output_tensor(oname)
            
            # once all parents of a child have been built, add to bfs queue
            if all(child_node._built_parents.values()):
              queue.append(child_node.label)

          # fill the output of Custom fetching from inner nodes `_oslot_to_otensor` attributes 
          if cur_node.name in self.output_nodes:
            for onode_oslot in cur_node._oslot_to_otensor:
              if (cur_node.name, onode_oslot) in self._oslot_to_inner_node_oslot.inv: 
                custom_node_oslot = self._oslot_to_inner_node_oslot.inv[(cur_node.name, onode_oslot)]
                result[custom_node_oslot] = cur_node._oslot_to_otensor[onode_oslot]
                    
    # TODO: Add attribute output_names to CustomBuilder                    
    return tuple([result[oslot] for oslot in self.output_names])
          
                
  def build_outputs(self,
                    islot_to_itensor,
                    output_names):
    """
    Build outputs for the associated Custom Node
    
    TODO: This function has many problems. The most glaring one is that it does
    not behave as the regular `build_outputs` function from InnerNodes. As is is written, it cannot bu
    """
    print("\ncust_builder; self.input_nodes", self.input_nodes)
    
    result = {}
    _input = islot_to_itensor
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      # fill the inner nodes islot_to_itensors with the input 
      for islot in self._islot_to_inner_node_islot:
        inner_node_data = self._islot_to_inner_node_islot[islot]
        for inode_name, inode_islot in inner_node_data:
          print("inode_name, inode_islot", inode_name, inode_islot)
          print("_input", _input)
          self.nodes[inode_name]._islot_to_itensor[inode_islot] = _input[islot]

      # init bfs
      visited = [False for _ in range(self.num_nodes)]
      queue = []
      for cur_inode_name in self.input_nodes:
        cur_inode_label = self.get_label_from_name(cur_inode_name)
        queue.append(cur_inode_label)
        while queue:
          cur_node_label = queue.pop(0)
          visited[cur_node_label] = True
          cur_node = self._label_to_node[cur_node_label]
          
          # build current inner node
          print("sb; cur_node._islot_to_itensor", cur_node.name, cur_node._islot_to_itensor)
          cur_node._build()
          print("sb; cur_node._oslot_to_otensor", cur_node.name, cur_node._oslot_to_otensor)
          
          # fetch children 
          for child_label in self.adj_list[cur_node_label]:
            # mark current node as visited for its children
            child_node = self._label_to_node[child_label]
            child_node._built_parents[cur_node_label] = True
            
            # fill child `_islot_to_itensor` attribute
            for oslot, islot in cur_node._child_label_to_slot_pairs[child_label]:
              child_node._islot_to_itensor[islot][oslot] = cur_node.get_output_tensor(oslot)

            # if all its parents are built, add to bfs queue
            if all(child_node._built_parents.values()):
              queue.append(child_node.label)
          
          # fill result with the inner node `_oslot_to_otensor` attribute 
          if cur_node.name in self.output_nodes:
            for onode_oslot in cur_node._oslot_to_otensor:
              if (cur_node.name, onode_oslot) in self._oslot_to_inner_node_oslot.inv: 
                custom_node_oslot = self._oslot_to_inner_node_oslot.inv[(cur_node.name, onode_oslot)]
                result[custom_node_oslot] = cur_node._oslot_to_otensor[onode_oslot]
                    
    return tuple([result[oslot] for oslot in output_names])

