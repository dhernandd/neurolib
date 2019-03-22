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
from collections import defaultdict
from bidict import bidict

import tensorflow as tf
# pylint: disable=bad-indentation, protected-access, no-member

class ANode(abc.ABC):
  """
  Abstract class for Nodes, the basic building block of the neurolib 
  
  An ANode is an abstraction of an operation, much like a tensorflow Op. Tensors
  enter and exit the node. However, compared to tensorflow nodes, ANodes are
  meant to represent higher level abstractions. Basically, they are mappings
  that encode some input data into their output. In the flow of information
  through a graphical statistical mode, ANodes represent the most relevant
  stops.
  
  Ex: The Variational Autoencoder Model graph is a sequence of ANodes given by: 
  
  [0->d_X] => [d_X->d_Z] => [d_Z->d_X] => [d_X->0]
  
  where the => arrows represent flowing tensors and the brackets [...] represent
  ANodes. Specifically, [d_X->d_Z] represents an ANode whose input is of
  dimension d_X and whose output is of dimension d_Z, d_Z < d_X. At the ends of
  the chain there is always an InputNode and an OutputNode, special types of
  ANode that represent respectively the Model's sources and sinks of
  information.
  
  Subclasses of ANode are not meant to be accessed (built and linked) directly,
  but rather through a Builder object.
    
  Upon initialization, an ANode holds specifications of its role in the full
  tensorflow graph of a Model to which the ANode belongs. The ANode tensorflow
  ops are not built at initialization, but only when an ANode `_build` method
  is called by the Builder object (see Builder docs).
  
  The algorithm to build the tensorflow graph of a Model depends on 3 ANode
  dictionaries that work together:
    self._built_parents : Keeps track of which among the parents of this node
        have been built. A node can only be built once all of its parents have
        been visited
    self._child_label_to_oslot : The keys are the labels of self's children. For each
        key, the only value value is an integer, the oslot in self that maps to
        that child.
    self._parent_label_to_islot : The keys are the labels of self's parents, the only
        value is an integer, the islot in self that maps to that child. 
  """
  def __init__(self,
               builder,
               is_sequence,
               name_prefix=None,
               **dirs):
    """
    Initialize an ANode
        
    TODO: Should the client be able to pass a tensorflow op directly? In
    that case, ANode could act as a simple wrapper that returns the input and
    the output.
    
    A child of ANode should implement __call__, build_outputs, _build.
    """
    # set builder attr and node label
    self.builder = builder
    self.label = builder.num_nodes
    builder.num_nodes += 1
    self._is_built = False

    # set name
    if not hasattr(self, 'name'):
      self.name = name_prefix + '_' + str(self.label)

    self._num_declared_inputs = 0
    self._num_declared_outputs = 0

    # batch size, max_steps
    self.batch_size = builder.batch_size
    self.is_sequence = is_sequence
    self.max_steps = builder.max_steps if is_sequence else None
    
    # Dictionaries for islot/oslot access    
    self._islot_to_shape = {}
    self._oslot_to_shape = {}
    self._oslot_to_otensor = {}
    self.islot_to_name = bidict({})
    self.oslot_to_name = bidict({})
    
    # Dictionaries for building the MG
    self._built_parents = {}
    self._child_label_to_slot_pairs = defaultdict(list)
    self._parent_label_to_islot = defaultdict(list)

    # Update directives
    self._update_directives(**dirs)
  
  def _set_name_or_get_name_prefix(self, name=None, name_prefix=None):
    """
    Set the node name if `name` is provided, else return `name_prefix`.
    
    NOTE: In case `name` is not provided, the `name` attribute is set during
    the call to ANode.__init__()
    """
    if name is not None:
      self.name = name
      name_prefix = None
    else:
      assert name_prefix is not None, "Cannot set `name` attribute"
    
    return name_prefix

  @property
  def num_declared_inputs(self):
    """
    Return the number of declared inputs.
    
    Useful for checks and debugging. This number should never change after a node
    is built.
    """
    return self._num_declared_inputs
  
  @num_declared_inputs.setter
  def num_declared_inputs(self, value):
    """
    Setter for self.num_declared_inputs
    """
    if value > self.num_expected_inputs:
      raise ValueError("Attribute num_inputs of {} must not be greater "
                       "than {}".format(self.__class__.__name__,
                                        self.num_expected_inputs))
    self._num_declared_inputs = value
    
  @property
  def num_declared_outputs(self):
    """
    Return the number of declared outputs.
    
    Useful for checks and debugging. This number should never change after a node
    is built.
    """
    return self._num_declared_outputs
  
  @num_declared_outputs.setter
  def num_declared_outputs(self, value):
    """
    Setter for self.num_declared_outputs
    """
    if value > self.num_expected_outputs:
      raise ValueError("Attribute num_outputs of {} must "
                       "not be greater than {}".format(self.__class__.__name__,
                                                           self.num_expected_outputs))
    self._num_declared_outputs = value
  
  def _update_directives(self, **directives):
    """
    Update the node directives
    """
    self.directives = {}
    self.directives.update(directives)
  
  @staticmethod
  def state_sizes_to_list(state_sizes):
    """
    Deal with different types of user provided state_sizes.
    
    Returns a list of lists of state sizes.
    
    NOTE: Call this method upon node initialization before any call to super to
    homogenize the user input
    """
    if isinstance(state_sizes, int):
      state_sizes = [[state_sizes]]
    elif isinstance(state_sizes, list):
      if not all([isinstance(size, list) for size in state_sizes]):
        raise TypeError("`state_sizes` argument must be an int of a list of lists "
                        "of int")
    return state_sizes

  def get_state_full_shapes(self):
    """
    Deprecated! (STILL USED BY RNN)
    
    Get the state size shapes for this ANode    
    """
    bsz = self.batch_size
    mx_stps = self.max_steps
    if self.is_sequence:
      main_oshapes = [[bsz, mx_stps] + sz for sz in self.state_sizes]
    else:
      main_oshapes = [[bsz] + sz for sz in self.state_sizes]
    
    return main_oshapes
         
  def get_state_size_ranks(self): 
    """
    Get the ranks of the states for this ANode
    """
    return [len(sz) for sz in self.state_sizes]
    
  def get_islot_shape(self, islot, iname):
    """
    Return the incoming shape corresponding to this islot.
    
    Args:
        islot (int) : The islot whose
    """
    return self._islot_to_shape[islot][iname]

#   def get_oslot_shape(self, oname):
  def get_oshape(self, oname):
    """
    Return the outgoing shape corresponding to this oname.
    """
#     return self._oslot_to_shape[oname]
    return self.oshapes[oname]

  def fill_oslot_with_tensor(self, oslot, tensor, name=None):
    """
    Assign otensor to its oslot.
    
    NOTE: A user-friendly key with the structure 'Node:key' is added to the dict
    `builder.otensor_names`, pointing to the tensorflow name of the tensor. This
    dictionary is pickled when a model is saved and loaded upon restore to
    provide access to the names in the tensorflow graph.
    """
    name = self.oslot_names[oslot] if name is None else name

    oslot_name = self.name + '_' + name
    o = tf.identity(tensor, name=oslot_name)
    self._oslot_to_otensor[name] = o
    oname = self.name + ':' + name
    self.builder.otensor_names[oname] = o.name
    
  def add_oname(self, oname, tensor):
    """
    """
    oname = self.name + ':' + oname
    self.builder.otensor_names[oname] = tensor.name
          
  def get_input_tensor(self, islot, iname):
    """
    Return a dictionary whose keys are the islots of the ANode and whose values
    are the incoming tensorflow Tensors.
    
    Requires the node to be built.
    """
    return self._islot_to_itensor[islot][iname]

  def get_output_tensor(self, oname):
    """
    Return a dictionary whose keys are the oslots of the ANode and whose values
    are the outgoing tensorflow Tensors.
    
    Requires the node to be built.
    """    
    return self._oslot_to_otensor[oname]

  def update_when_linked_as_node1(self):
    """
    Execute after linking as parent node
    """
    return
  
  def update_when_linked_as_node2(self):
    """
    Execute after linking as child node
    """
    return
    
  def __call__(self, *inputs):
    """
    Evaluate the node on a list of inputs.
    
    NOTE: Delegate to build_outputs
    """
    raise NotImplementedError("Please implement me.")

  def build_outputs(self, **inputs):
    """
    Evaluate the node on a dict of inputs. 
    """
    raise NotImplementedError("Please implement me.")
  
  def _build(self):
    """
    Build the node
    """
    raise NotImplementedError("Please implement me.")
  
  def fetch_tensor_by_oname(self, oname):
    """
    """
    oname = self.name + ':' + oname
    tname = self.builder.otensor_names[oname]
    return tf.get_default_graph().get_tensor_by_name(tname)
  
#   def get_inode_islot(self, inode):
#     """
#     """
#     if isinstance(inode, ANode):
#       inode = inode.name
#     return self.inputs_list.index(inode)
#   
  