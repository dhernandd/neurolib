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
  
  dtype_dict = {'float32' : tf.float32,
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

    # Slot names
    self.oslot_to_name[0] = 'main_' + str(self.label) + '_0'
    
    # Deal with sequences
    self.main_oshapes = self.get_state_full_shapes()
    self.D = self.get_state_size_ranks()
    self._oslot_to_shape[0] = self.main_oshapes[0]
    
    # InputNode directives
    self.dtype = self.dtype_dict[dirs.pop('dtype')]
      
  def _update_directives(self, **dirs):
    """
    Update this node directives
    """
    self.directives = {'output_0_name' : 'main',}
    self.directives.update(dirs)
    
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
               dtype='float64',
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
                                               dtype=dtype,
                                               name_prefix=name_prefix,
                                               **dirs)

    self.free_oslots = list(range(self.num_expected_outputs))

    self._update_directives(**dirs)

  def _update_directives(self, **dirs):
    """
    Update default directives
    """
    this_node_dirs = {'dtype' : 'float64'}
    this_node_dirs.update(dirs)
    
    super(PlaceholderInputNode, self)._update_directives(**this_node_dirs)
    
  def _build(self):
    """
    Build a PlaceholderInputNode.
    
    Assigns a new tensorflow placeholder to _oslot_to_otensor[0]
    """
    name = self.name + '_' + self.directives['output_0_name']
    out_shape, oslot = self.main_oshapes[0], 0
    o0 = tf.placeholder(self.dtype, shape=out_shape, name=name)
    o0_rname = self.name + ':' + self.directives['output_0_name']
    self._oslot_to_otensor[oslot] = o0
    self.builder.otensor_names[o0_rname] = o0.name
    
    self._is_built = True

  def __call__(self, inputs, state):
    """
    Call the node with some inputs
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
    self.dist = None
    self.xdim = self.state_sizes[0][0]
    
    if self.batch_size is None:
      self.dummy_bsz = tf.placeholder(tf.int32, [None], self.name + '_dummy_bsz')
      self.builder.dummies[self.dummy_bsz.name] = [None]

    self.free_oslots = list(range(self.num_expected_outputs))
    
    # Slot names
    self.oslot_to_name[1] = 'loc_'  + str(self.label) + '_1'
    self.oslot_to_name[2] = 'scale_' + str(self.label) + '_2'

    self._update_directives(**dirs)
    
    self._declare_secondary_outputs()

  def _update_directives(self, **dirs):
    """
    Update the node directives
    """
    this_node_dirs = {'output_1_name' : 'loc',
                      'output_2_name' : 'scale'}
    this_node_dirs.update(dirs)
    
    super(NormalInputNode, self)._update_directives(**this_node_dirs)
        
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
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      if self.is_sequence and self.max_steps is None:
        loc = tf.get_variable('loc',
                              dtype=self.dtype,
                              initializer=self.loc_init,
                              validate_shape=True)
        scale = tf.get_variable('scale',
                                dtype=self.dtype,
                                initializer=self.sc_init,
                                validate_shape=True)
      else:
        li = tf.zeros([self.xdim], dtype=self.dtype)
        loc_dist = tf.get_variable('loc',
                                   dtype=self.dtype,
                                   initializer=li)
        loc = tf.tile(loc_dist, self.dummy_bsz)
        loc = tf.reshape(loc, [-1, self.xdim])
        
        si = tf.eye(self.xdim, dtype=self.dtype)
        scale = tf.get_variable('scale',
                                dtype=self.dtype,
                                initializer=si)
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
    
    o2_name = self.name + '_' + self.directives['output_2_name']
    self._oslot_to_otensor[2] = tf.identity(scale, name=o2_name)

    self._is_built = True


if __name__ == '__main__':
  print(dist_dict.keys())  # @UndefinedVariable
  