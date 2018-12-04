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
# from tensorflow.python.ops import rnn_cell_impl

from neurolib.encoder.anode import ANode
from neurolib.encoder.evolution_sequence import (BasicRNNEvolutionSequence, 
                LSTMEvolutionSequence, EvolutionSequence, CustomEvolutionSequence)
from neurolib.encoder.custom import CustomNode
# from neurolib.encoder.rnn import CustomRNN
from neurolib.encoder.output import OutputNode
from neurolib.utils.utils import check_name
from neurolib.builders.static_builder import StaticBuilder
# from neurolib.encoder.output_seq import OutputSequence
from neurolib.encoder.deterministic import DeterministicNNNode
from neurolib.encoder.input import PlaceholderInputNode

# pylint: disable=bad-indentation, no-member, protected-access

sequence_dict = {'basic' : BasicRNNEvolutionSequence,
                 'lstm' : LSTMEvolutionSequence,
                 'custom' : CustomEvolutionSequence}

class SequentialBuilder(StaticBuilder):
  """
  A SequentialBuilder is a Builder for Sequential Models, Models that involve
  sequential data (such as a time series).
  
  Building of a static Model through a SequentialBuilder is done in two stages:
  Declaration and Construction. In the Declaration stage, the input, output and
  inner nodes of the Model are 'added' to the Model graph (MG) and directed
  links - representing the flow of tensors - are defined between them. In the
  Construction stage, a BFS-like algorithm is called that generates a tensorflow
  graph out of the MG specification
  
  A SequentialBuilder defines the following key methods
  
    addOutput(): ...
    
    addInput(): ...
    
    addDirectedLink(): ...
    
    build()
    
  Ex: The following builds a regression Model
      
      builder = SequentialBuilder(scope='regression')
      in0 = builder.addInput(input_dim, name="features")
      enc1 = builder.addInner(output_dim, **dirs)
      out0 = builder.addOutput(name="prediction")
      builder.addDirectedLink(in0, enc1)
      builder.addDirectedLink(enc1, out0)
      
      builder.build()
    
    The 2 input nodes define placeholders for the features and response data
  
  """
  def __init__(self,
               max_steps,
               scope=None,
               batch_size=1):
    """
    Initialize the SequentialBuilder
    
    Args:
      max_steps (int): The maximum number of steps in the sequence
      scope (str): The tensorflow scope of the Model to be built
      batch_size (int): The batch size. Defaults to None (unspecified)
    """
    self.max_steps = max_steps
    super(SequentialBuilder, self).__init__(scope, batch_size=batch_size)
    
    self.input_sequences = {}
    self.sequences = {}
    self.output_sequences = {}

  @check_name
  def addInputSequence(self,
                       state_size,
                       name=None,
                       node_class=PlaceholderInputNode,
                       **dirs):
    """
    Add an InputSequence
    """
    node_name = StaticBuilder.addInput(self,
                                       state_size=state_size,
                                       iclass=node_class,
                                       name=name,
                                       is_sequence=True,
                                       **dirs)
    print("seq_builder: node_name", node_name)
    self.input_sequences[node_name] = self.input_nodes[node_name]
    
    return node_name
  
  @check_name
  def addOutputSequence(self, name=None):
    """
    Add OutputSequence
    """
    return self.addOutput(name=name)
  
  @check_name
  def addInnerSequence(self, 
                       state_size, 
                       num_inputs=1,
                       node_class='deterministic', 
                       name=None,
                       **dirs):
    """
    """
    return self.addInner(state_size,
                         num_inputs=num_inputs,
                         node_class=node_class,
                         is_sequence=True,
                         name=name,
                         **dirs)
  
  def addEvolutionSequence(self,
                           state_sizes, 
                           num_inputs,
#                            init_states=None,
                           mode='forward',
                           ev_seq_class='basic',
                           cell_class='basic', 
                           name=None,
                           **dirs):
    """
    """
    if isinstance(ev_seq_class, str):
      ev_seq_class = sequence_dict[ev_seq_class]
#     init_states = [self.nodes[node_name] for node_name in init_states]
    node = ev_seq_class(self,
                        state_sizes,
      #                   init_states=init_states,
                        num_inputs=num_inputs,
                        name=name,
                        mode=mode,
                        cell_class=cell_class,
                        **dirs)
    name = node.name
    self.nodes[name] = node
    self._label_to_node[node.label] = node
    
    return name

  def addDirectedLink(self, node1, node2, oslot=0, islot=0):
    """
    Add directed links to the Encoder graph. 
    
    A) Deal with different item types. The client may provide as arguments,
    either EncoderNodes or integers. Get the EncoderNodes in the latter case
 
    B) Check that the provided oslot for node1 is free. Otherwise, raise an
    exception.
    
    C) Initialize/Add dimensions the graph representations stored in the
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
      node1 (ANode): Node from which the edge emanates
      node2 (ANode): Node to which the edge arrives
      oslot (int): Output slot in node1
      islot (int): Input slot in node2
    """
    # Stage A
    if isinstance(node1, str):
      node1 = self.nodes[node1]
    if isinstance(node2, str):
      node2 = self.nodes[node2]
    
    # Stage B
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
      raise KeyError("The requested oslot has not been found. Inferring this "
                     "oslot shape may require knowledge of the shape of its "
                     "inputs. In that case, all the inputs for this node must "
                     "be declared")
    if islot in node2._islot_to_shape:
#       print("islot, node2._islot_to_shape", islot, node2._islot_to_shape)
      raise AttributeError("Input slot {} is already occupied. Assign to "
                           "a different islot".format(islot))

    # Stage C
    print('\nAdding dlink', node1.name, ' -> ', node2.name)
#     if not hasattr(self, "adj_matrix"):
    if self.adj_matrix is None:
      self.adj_matrix = [[0]*nnodes for _ in range(nnodes)]
      self.adj_list = [[] for _ in range(nnodes)]
    else:
#       print(self.adj_matrix)
#       print('Before:', self.adj_list)
      if nnodes > len(self.adj_matrix):
        l = len(self.adj_matrix)
        for row in range(l):
          self.adj_matrix[row].extend([0]*(nnodes-l))
        for _ in range(nnodes-l):
          self.adj_matrix.append([0]*nnodes)
          self.adj_list.append([])
    
    # Stage D
    if isinstance(node1, ANode) and isinstance(node2, ANode):
      self._check_items_do_exist()
      self.adj_matrix[node1.label][node2.label] = 1
      self.adj_list[node1.label].append(node2.label)
#       print('After:', self.adj_list)
      
#       self.model_graph.add_edge(pydot.Edge(node1.vis, node2.vis))
    else:
      raise ValueError("The endpoints of the links must be either Encoders or "
                       "integers labeling Encoders")
      
    # Stage E
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


    print('Exchanged shape:', exchanged_shape)
    node2._islot_to_shape[islot] = exchanged_shape
    node2._parent_label_to_islot[node1.label] = islot    
    node2.num_declared_inputs += 1
    
    # Stage F
    update = getattr(node2, '_update_when_linked_as_node2', None)
    if callable(update):
      node2._update_when_linked_as_node2()

    # Initialize _built_parents for the child node. This is used in the build
    # algorithm below.
    node2._built_parents[node1.label] = False
      
  def _check_items_do_exist(self):
    """
    TODO:
    """
    pass
      
  def check_graph_correctness(self):
    """
    Checks the coding graph outlined so far. 
    
    TODO:
    """
    pass
    
  def build(self):
    """
    Build the model for this builder.
    
    # put all nodes in a waiting list of nodes
    # for node in input_nodes:
      # start BFS from node. Add node to queue.
      # (*)Dequeue, mark as visited
      # build the tensorflow graph with the new added node
      # Look at all its children nodes.
      # For child in children of node
          Add node to the list of inputs of child
      #   have we visited all the parents of child?
          Yes
            Add to the queue
      # Go back to (*)
      * If the queue is empty, exit, start over from the next input node until all 
      # have been exhausted
      
      # TODO: deal with back links.
    """       
    self.check_graph_correctness()
    
    print('\nBEGIN MAIN BUILD')
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      visited = [False for _ in range(self.num_nodes)]
      queue = []
      for cur_inode_name in self.input_nodes:
        cur_inode_label = self.get_label_from_name(cur_inode_name)
        
        # start BFS from this input
        queue.append(cur_inode_label)
        while queue:
          # A node is visited by definition once it is popped from the queue
          cur_node_label = queue.pop(0)
          visited[cur_node_label] = True
          cur_node = self._label_to_node[cur_node_label]
          
          
          if isinstance(cur_node, EvolutionSequence):
            pass
          else:
            pass
    
          print("Building node: ", cur_node.label, cur_node.name)
          # Build the tensorflow graph for this Encoder
#           print("cur_node, _islot_to_itensor", cur_node.label, cur_node._islot_to_itensor)
          cur_node._build()
#           print("cur_node, _oslot_to_otensor", cur_node.label, cur_node._oslot_to_otensor)
                      
          # Go over the current node's children
          for child_label in self.adj_list[cur_node_label]:
            child_node = self._label_to_node[child_label]
            child_node._built_parents[cur_node_label] = True
            
            # Get islot and oslot
            oslot = cur_node._child_label_to_oslot[child_label]
            islot = child_node._parent_label_to_islot[cur_node_label]
            
            # Fill the inputs of the child node
#             print('cur_node', cur_node_label, cur_node.name)
#             print('cur_node.get_outputs()', cur_node.get_outputs() )
            child_node._islot_to_itensor[islot] = cur_node._oslot_to_otensor[oslot]
            if isinstance(child_node, CustomNode):
              enc, enc_islot = child_node._islot_to_enc_islot[islot]
              enc._islot_to_itensor[enc_islot] = cur_node._oslot_to_otensor[oslot]
            
            # If the child is an OutputNode, we can append to the queue right away
            # (OutputNodes have only one input)
            if isinstance(child_node, OutputNode):
              queue.append(child_node.label)
              continue
            
            # A child only gets added to the queue, i.e. ready to be built, once
            # all its parents have been built ( and hence, produced the
            # necessary inputs )
#             print("child_node._built_parents.values()", child_node.label, child_node._built_parents.values())
            if all(child_node._built_parents.values()):
              queue.append(child_node.label)
  

    print('Finished building')
    print('END MAIN BUILD')