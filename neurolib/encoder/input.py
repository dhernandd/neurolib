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

# pylint: disable=bad-indentation, no-member

class InputNode(ANode):
  """
  An abstract ANode representing inputs to the Model Graph (MG).
  
  An InputNode represents a source of information. InputNodes are used to
  represent user-provided data to be fed to the MG by means of a tensorflow
  Placeholder. InputNodes may represent as well random inputs to the MG.
  
  InputNodes have no incoming links. Information is "created" at the InputNode.
  Directed links into the InputNode and assignment to self.num_inputs are
  therefore forbidden.
  
  InputNodes have one main output and possibly secondary ones. The latter are
  used most often to output the relevant statistics of a random input. In that
  case, the main output is a sample from the corresponding distribution. The
  main output of a stochastic InputNode MUST be assigned to oslot = 0
  """
  num_expected_inputs = 0
  
  dtype_dict = {'float32' : tf.float32,
                'float64' : tf.float64,
                'int32' : tf.int32}

  def __init__(self,
               builder,
               state_sizes,
               is_sequence=False,
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
    super(InputNode, self).__init__()

    self.builder = builder
    self.label = builder.num_nodes
    builder.num_nodes += 1
  
    self.is_sequence = is_sequence
    self.state_sizes = self.state_sizes_to_list(state_sizes)
    self.batch_size = builder.batch_size
    self.max_steps = builder.max_steps if hasattr(builder, 'max_steps') else None

    # Slot names
    self.oslot_to_name[0] = 'main_' + str(self.label) + '_0'
    
    # Deal with sequences
    self.main_oshapes = self.get_state_full_shapes()
    self.D = self.get_state_size_ranks()
    self._oslot_to_shape[0] = self.main_oshapes[0]
    
    # InputNode directives
    self.dtype = self.dtype_dict[dirs.pop('dtype')]
      
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
               dtype='float64',
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
    super(PlaceholderInputNode, self).__init__(builder,
                                               state_sizes,
                                               is_sequence=is_sequence,
                                               dtype=dtype,
                                               **dirs)

    self.name = name or "In_" + str(self.label)

    self.free_oslots = list(range(self.num_expected_outputs))

    self._update_default_directives(**dirs)

  def _update_default_directives(self, **dirs):
    """
    Update default directives
    """
    self.directives = {'dtype' : 'float64'}
    self.directives.update(dirs)
    
  def _build(self):
    """
    Build a PlaceholderInputNode.
    
    Assigns a new tensorflow placeholder to _oslot_to_otensor[0]
    """
#     dirs = self.directives
    
    name = self.name
    out_shape = self.main_oshapes
    for oslot, out_shape in enumerate(self.main_oshapes):
      self._oslot_to_otensor[oslot] = tf.placeholder(self.dtype,
                                                     shape=out_shape,
                                                     name=name)
    self._is_built = True

  def __call__(self, inputs, state):
    """
    """
    InputNode.__call__(self, inputs, state)


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
    super(NormalInputNode, self).__init__(builder,
                                          state_sizes,
                                          is_sequence=is_sequence,
                                          dtype='float64',
                                          **dirs)
    self.name = "Normal_" + str(self.label) if name is None else name
    self.dist = None
    self.xdim = self.state_sizes[0][0]
    
    if self.batch_size is None:
      self.dummy_bsz = tf.placeholder(tf.int32, [None], self.name + '_dummy_bsz')
#       dz = tf.expand_dims(self.dummy_bsz, axis=1)
#       dz = tf.tile(tf.zeros([self.xdim], tf.float64), self.dummy_bsz)
#       dz = tf.reshape(dz, [-1, self.xdim])
      #       self.dummy_zeros = tf.matmul(dz, )
      self.builder.dummies[self.dummy_bsz.name] = [None]

    self.free_oslots = list(range(self.num_expected_outputs))
    
    # Slot names
    self.oslot_to_name[1] = 'loc_'  + str(self.label) + '_1'
    self.oslot_to_name[2] = 'scale_' + str(self.label) + '_2'

    print("self.D", self.D)
    self._update_default_directives(**dirs)
    
    self._declare_secondary_outputs()

  def _update_default_directives(self, **dirs):
    """
    Update the node directives
    """
    if self.D[0] == 1:
#       oshape = self.main_oshapes[0]
      oshape = self.main_oshapes[0][-1:]
#       sc_oshape = oshape + oshape[-1:]
    else:
      raise NotImplementedError("main output with rank > 1 is not implemented "
                                "for the Normal Input Node. ")
    
    # Define dummy Placeholders (always in root scope)
    if self.batch_size is None:
#       dummy_loc = tf.placeholder(self.dtype, oshape, self.name + '_dummy_loc')
#       dummy_sc = tf.placeholder(self.dtype, sc_oshape, self.name + '_dummy_sc')
#       self.builder.dummies[dummy_loc.name] = oshape
#       self.builder.dummies[dummy_sc.name] = sc_oshape
      print("oshape", oshape)
#       dummy = tf.expand_dims(tf.cast(self.dummy_bsz, self.dtype), axis=1)
#       li = tf.zeros([1]+oshape, dtype=self.dtype)
#       dz = tf.reshape(dz, [-1, self.xdim])
      dz = tf.tile(tf.zeros([self.xdim], tf.float64), self.dummy_bsz)
#       self.loc_init = tf.reshape(dz, [-1, self.xdim])
#       print("self.loc_init", self.loc_init)
#       self.loc_init = tf.zeros(oshape, dtype=self.dtype)

#       self.sc_init = (tf.zeros_like(dummy_sc, dtype=self.dtype) 
#                       + tf.eye(oshape[-1], dtype=tf.float64))
#       si = tf.expand_dims(tf.eye(oshape[0], dtype=self.dtype), axis=0)
      sz = tf.reshape(tf.eye(self.xdim, dtype=tf.float64), [-1])
      sz = tf.tile(sz, self.dummy_bsz)
#       self.sc_init = tf.tensordot(dummy, si, axes=[1, 0])
#       self.sc_init = tf.reshape(sz, [-1, self.xdim, self.xdim])
#       print("self.sc_init", self.sc_init)
#       self.sc_init = tf.eye(oshape[0], dtype=self.dtype)
    else:
      dummy = tf.zeros(oshape, self.dtype)

    self.directives = {'output_0_name' : 'main',
                       'output_1_name' : 'loc',
                       'output_2_name' : 'scale'}
    self.directives.update(dirs)
        
  def _declare_secondary_outputs(self):
    """
    Declare the statistics of the normal as secondary outputs. 
    """
    oshape = self.main_oshapes[0]
    
    add_name = lambda x : self.name + '_' + x
    self._oslot_to_shape[1] = oshape[1:] # mean oslot
    o1 = self.builder.addOutput(name=add_name(self.directives['output_1_name']))
    self.builder.addDirectedLink(self, o1, oslot=1)
    
    self._oslot_to_shape[2] = oshape[1:] + [oshape[-1]] # stddev oslot  
    o2 = self.builder.addOutput(name=add_name(self.directives['output_2_name']))
    self.builder.addDirectedLink(self, o2, oslot=2)
  
  def _get_sample(self):
    """
    Get a sample from the distribution.
    """
#     sample = self.dist.sample(sample_shape=self.dummy_bsz)
    sample = self.dist.sample(sample_shape=self.dummy_bsz)
    sample.set_shape([None, self.xdim])
    return sample
  
  def __call__(self):
    """
    Return a sample from the distribution
    """
    return self._get_sample()
  
  def _build(self):
    """
    Build a NormalInputNode.
    
    Assign a sample from self.dist to _oslot_to_otensor[0]
     
    Assign the mean from self.dist to _oslot_to_otensor[1]
    
    Assign the Cholesky decomposition of the covariance from self.dist to
    _oslot_to_otensor[2]
    """
#     oshape = self.main_oshapes[0]
#     sc_oshape = oshape + oshape[-1:]
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      if self.is_sequence and self.max_steps is None:
        loc = tf.get_variable('loc',
                              dtype=self.dtype,
                              initializer=self.loc_init,
                              validate_shape=True)
#                               validate_shape=False)
        scale = tf.get_variable('scale',
                                dtype=self.dtype,
                                initializer=self.sc_init,
                                validate_shape=True)
#                                 validate_shape=False)
      else:
        li = tf.zeros([self.xdim], dtype=self.dtype)
#         dz = tf.tile(tf.zeros([self.xdim], tf.float64), self.dummy_bsz)
        loc_dist = tf.get_variable('loc',
                              dtype=self.dtype,
                              initializer=li)
#                               validate_shape=True)
#                               validate_shape=False)
        loc = tf.tile(loc_dist, self.dummy_bsz)
        loc = tf.reshape(loc, [-1, self.xdim])
#         self.loc_init = tf.reshape(loc, [-1, self.xdim])
#         loc.set_shape([None, self.xdim])
        
        si = tf.eye(self.xdim, dtype=self.dtype)
        scale = tf.get_variable('scale',
                                dtype=self.dtype,
                                initializer=si)
#                                 validate_shape=True)
#                                 validate_shape=False)
        scale_dist = tf.linalg.LinearOperatorFullMatrix(scale)
        scale = tf.reshape(scale, [-1])
        scale = tf.tile(scale, self.dummy_bsz)
        scale = tf.reshape(scale, [-1, self.xdim, self.xdim])
    
    self.dist = dist = dist_dict['MultivariateNormalLinearOperator'](loc=loc_dist,
                                                                     scale=scale_dist)
    
    samp = dist.sample(sample_shape=self.dummy_bsz)
    if not self.is_sequence:
      samp.set_shape([None, self.xdim])
    o0_name = self.name + '_' + self.directives['output_0_name']
    self._oslot_to_otensor[0] = tf.identity(samp, name=o0_name)
    
    o1_name = self.name + '_' + self.directives['output_1_name']
    self._oslot_to_otensor[1] = tf.identity(loc, name=o1_name)
    print("self._oslot_to_otensor[1]", self._oslot_to_otensor[1])
#     assert self._oslot_to_otensor[1].shape.as_list() == self._oslot_to_shape[1]
    
    o2_name = self.name + '_' + self.directives['output_2_name']
    self._oslot_to_otensor[2] = tf.identity(scale, name=o2_name)
#     assert self._oslot_to_otensor[2].shape.as_list() == self._oslot_to_shape[2]

    self._is_built = True


if __name__ == '__main__':
  print(dist_dict.keys())  # @UndefinedVariable
  