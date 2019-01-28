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
    """
    self._num_declared_inputs = 0
    self._num_declared_outputs = 0
    
    # Dictionaries for access    
    self._islot_to_shape = {}
    self._oslot_to_shape = {}
    self._islot_to_itensor = {}
    self._oslot_to_otensor = {}
    self.islot_to_name = bidict({})
    self.oslot_to_name = bidict({})
    
    self._built_parents = {}
    self._child_label_to_slot_pairs = defaultdict(list)
    self._parent_label_to_islot = defaultdict(list)
    
    self._is_built = False

    # Set the node label and name
    self.builder = builder
    self.label = builder.num_nodes
    builder.num_nodes += 1
    if not hasattr(self, 'name'):
      self.name = name_prefix + '_' + str(self.label)
    
    # Set the batch size, max_steps
    self.batch_size = builder.batch_size
    self.is_sequence = is_sequence
    self.max_steps = builder.max_steps if is_sequence else None
    
    # Update directives
    self._update_directives(**dirs)
  
  def _set_name_or_get_name_prefix(self, name=None, name_prefix=None):
    """
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
  
  def _update_directives(self, **dirs):
    """
    Update the node directive
    
    Here, every ANode dictionary of directives is defined.
    """
    self.directives = {}
    self.directives.update(dirs)
  
  @staticmethod
  def state_sizes_to_list(state_sizes):
    """
    Get a list of output sizes corresponding to each oslot
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
    
  def get_islot_shape(self, islot=0):
    """
    Return the incoming shape corresponding to this islot.
    
    Args:
        islot (int) : The islot whose
    """
    return self._islot_to_shape[islot]

  def get_oslot_shape(self, oslot=0):
    """
    Return the outgoing shape corresponding to this oslot.
    """
    return self._oslot_to_shape[oslot]

  def get_input(self, islot):
    """
    Return a dictionary whose keys are the islots of the ANode and whose values
    are the incoming tensorflow Tensors.
    
    Requires the node to be built.
    """
    return self._islot_to_itensor[islot]
    
  def get_output(self, oslot):
    """
    Return a dictionary whose keys are the oslots of the ANode and whose values
    are the outgoing tensorflow Tensors.
    
    Requires the node to be built.
    """
    return self._oslot_to_otensor[oslot]
  
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
    
  def __call__(self, inputs, state):
    """
    """
    raise NotImplementedError("")
  