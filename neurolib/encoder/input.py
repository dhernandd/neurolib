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

from neurolib.encoder import _globals as dist_dict
from neurolib.encoder.anode import ANode
from abc import abstractmethod
from neurolib.utils.directives import NodeDirectives

# pylint: disable=bad-indentation, no-member

class InputNode(ANode):
  """
  An abstract ANode representing inputs to the Model Graph (MG).
  
  An InputNode represents a source of information. InputNodes are typically
  stand for user-provided data, that is fed to the MG by means of a tensorflow
  Placeholder. InputNodes may also represent random inputs to the MG.
  
  InputNodes have no incoming links. Information is "created" at the InputNode.
  Directed links into the InputNode and assignment to self.num_inputs are
  therefore forbidden.
  
  InputNodes have one main output and possibly secondary ones. The latter are
  used most often to output the relevant statistics of a random input. In that
  case, the main output is a sample from the corresponding distribution. The
  main output of a stochastic InputNode MUST be assigned to oslot = 0
  """
  num_expected_inputs = 0
  
  type_dict = {'float32' : tf.float32,
               'float64' : tf.float64,
               'int32' : tf.int32}

  def __init__(self,
               builder,
               state_sizes,
               is_sequence=False,
               name_prefix=None,
               **dirs):
    """
    Initialize the InputNode
    
    Args:
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes.
      
      state_sizes (int or list of ints): The shape of the main output
          code. This excludes the 0th dimension - batch size - and the 1st
          dimension when the data is a sequence - number of steps.
      
      is_sequence (bool): Is the input a sequence?
    """
    self.state_sizes = self.state_sizes_to_list(state_sizes)
    
    super(InputNode, self).__init__(builder,
                                    is_sequence,
                                    name_prefix=name_prefix,
                                    **dirs)
    # directives object
    self.directives = NodeDirectives(self.directives)
    self.oslot_names = self.directives.output_names
        
    # shapes
    self.oshapes = self._get_all_oshapes()
    self.D = self.get_state_size_ranks()
    self.xdim = self.state_sizes[0][0]
      
  def _update_directives(self, **directives):
    """
    Update this node directives
    """
    this_node_dirs = {'outputname_0' : 'main'}
    this_node_dirs.update(directives)
    super(InputNode, self)._update_directives(**this_node_dirs)
    
  @abstractmethod
  def _build(self):
    """
    Build the InputNode.
    """
    raise NotImplementedError("Please implement me")
    

class PlaceholderInputNode(InputNode):
  """
  An InputNode to represent data to be fed to the Model Graph (MG).
  
  Data fed to the MG, for instance for training or sampling purposes is
  represented by a PlaceholdeInputNode. On build, a tensorflow placeholder is
  created and added to the MG. PlaceholderInputNodes have a single output slot
  that maps to a tensorflow Placeholder.
 
  Class attributes:
    num_expected_outputs = 1
    num_expected_inputs = 0
  """
  num_expected_outputs = 1
  
  def __init__(self,
               builder,
               state_sizes,
               is_sequence=False,
               name=None,
               name_prefix='PhIn',
               **dirs):
    """
    Initialize the PlaceholderInputNode
    
    Args:
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes. 

      state_sizes (int or list of ints): The shape of the main output
          code. This excludes the 0th dimension - batch size - and the 1st
          dimension when the data is a sequence - number of steps.

      is_sequence (bool): Is the input a sequence?
      
      name (str): A unique name for this node.

      dirs (dict): A set of user specified directives for constructing this
          node.
    """
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    super(PlaceholderInputNode, self).__init__(builder,
                                               state_sizes,
                                               is_sequence=is_sequence,
                                               name_prefix=name_prefix,
                                               **dirs)
    
    # init list of free i/o slots
    self.free_oslots = list(range(self.num_expected_outputs))

  def _update_directives(self, **dirs):
    """
    Update default directives
    """
    this_node_dirs = {'dtype' : 'float64'}
    this_node_dirs.update(dirs)
    super(PlaceholderInputNode, self)._update_directives(**this_node_dirs)

  def _get_all_oshapes(self):
    """
    Declare the shapes for every output
    """
    bsz = self.batch_size
    mx_stps = self.max_steps
    const_sh = [bsz, mx_stps] if self.is_sequence else [bsz]
    
    xdim = self.state_sizes[0][0]
    return {self.oslot_names[0] : const_sh + [xdim]}
    
  def __call__(self, *inputs):
    """
    Call the InputNode
    """
    if inputs:
      raise ValueError("A call an InputNode must have no arguments")
    return self.get_outputs()
    
  def get_outputs(self, islot_to_itensor=None):
    """
    Evaluate the node on a dict of inputs. 
    """
    if islot_to_itensor is not None:
      raise ValueError("`InputNode.get_outputs` must have no arguments")

    # directives
    dirs = self.directives
    
    oname = self.oslot_names[0]
    dtype = self.type_dict[dirs.dtype]
    oshape = self.oshapes[oname]
    return tf.placeholder(dtype, shape=oshape)
  
  def _build(self):
    """
    Build a PlaceholderInputNode.
    
    Assigns a new tensorflow placeholder to _oslot_to_otensor[0]
    """
    output = self.get_outputs()
    self.fill_oslot_with_tensor(0, output)
    self._is_built = True


class NormalInputNode(InputNode):
  """
  An InputNode representing a random input to the Model Graph (MG) drawn from a
  normal distribution.
  
  Initial values for the statistics of a NormalInputNode (mean and stddev) can
  be passed to the node as directives. They may be specified as trainable or
  not. The main output (oslot=0) of a NormalInputNode is a sample from the.
  oslots 1 and 2 are reserved for the mean and stddev respectively.
  
  Instances of NormalInputNode have a `dist` attribute which is a tensorflow
  distribution. Sampling from the node is delegated to `self.dist.sample(..)`.
  
  Class attributes:
    num_expected_outputs = 3
    num_expected_inputs = 0
  """
  num_expected_outputs = 3

  def __init__(self,
               builder,
               state_sizes,
               is_sequence=False,
               name=None,
               name_prefix='NormalIn',
               **dirs):
    """
    Initialize the NormalInputNode
        
    Args:
      builder (Builder): An instance of Builder necessary to declare the
          secondary output nodes. 

      state_sizes (int or list of ints): The shape of the main output
          code. This excludes the 0th dimension - batch size - and the 1st
          dimension when the data is a sequence - number of steps.
          
      is_sequence (bool): Is the input a sequence?

      name (str): A unique name for this node.

      dirs (dict): A set of user specified directives for constructing this
          node.
    """
    name_prefix = self._set_name_or_get_name_prefix(name, name_prefix=name_prefix)
    
    super(NormalInputNode, self).__init__(builder,
                                          state_sizes,
                                          is_sequence=is_sequence,
                                          dtype='float64',
                                          name_prefix=name_prefix,
                                          **dirs)

    # Initialize list of free i/o slots
    self.free_oslots = list(range(self.num_expected_outputs))

    # Get the dummy batch size
    if self.batch_size is None:
      self.dummy_bsz = self.builder.dummy_bsz
    
    self.dist = None

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'outputname_1' : 'loc',
                      'outputname_2' : 'scale'}
    this_node_dirs.update(dirs)
    super(NormalInputNode, self)._update_directives(**this_node_dirs)
  
  def _get_all_oshapes(self):
    """
    Declare the shapes for every output
    """
    bsz = self.batch_size
    mx_stps = self.max_steps
    const_sh = [bsz, mx_stps] if self.is_sequence else [bsz]
    
    xdim = self.state_sizes[0][0]
    return {self.oslot_names[0] : const_sh + [xdim],
            self.oslot_names[1] : const_sh + [xdim],
            self.oslot_names[2] : const_sh + [xdim, xdim]}
  
  def __call__(self, *inputs):
    """
    Evaluate the node on a list of inputs.    
    """
    if inputs:
      raise ValueError("A call an InputNode must have no arguments")
    return self.get_outputs()
  
  def get_outputs(self, islot_to_itensor=None):
    """
    Evaluate the node on a dict of inputs.
    """
    if islot_to_itensor is not None:
      raise ValueError("`InputNode.get_outputs` must have no arguments")

    # directives
    dirs = self.directives
    dtype = self.type_dict[dirs.dtype]
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      if self.is_sequence and self.max_steps is None:
        raise NotImplementedError("")
      else:
        li = tf.zeros([self.xdim], dtype=dtype)
        loc_dist = tf.get_variable('loc',
                                   dtype=dtype,
                                   initializer=li)
        loc = tf.tile(loc_dist, self.dummy_bsz)
        loc = tf.reshape(loc, [-1, self.xdim])
        
        si = tf.eye(self.xdim, dtype=dtype)
        scale = tf.get_variable('scale',
                                dtype=dtype,
                                initializer=si)
        scale_dist = tf.linalg.LinearOperatorFullMatrix(scale)
        scale = tf.reshape(scale, [-1])
        scale = tf.tile(scale, self.dummy_bsz)
        scale = tf.reshape(scale, [-1, self.xdim, self.xdim])
    
    self.dist = dist = dist_dict['MultivariateNormalLinearOperator'](loc=loc_dist,
                                                                     scale=scale_dist)
    samp = dist.sample(sample_shape=self.dummy_bsz)
    if not self.is_sequence:
      samp.set_shape([None, self.xdim]) # set the known shape
      
    return samp, loc, scale

  def _build(self):
    """
    Build the NormalInputNode
    """
    samp, loc, scale = self.get_outputs()
    
    # Fill the oslots
    self.fill_oslot_with_tensor(0, samp)
    self.fill_oslot_with_tensor(1, loc)
    self.fill_oslot_with_tensor(2, scale)

    self._is_built = True
  