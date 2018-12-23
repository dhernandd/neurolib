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
  Placeholder. InputNodes represent as well random inputs to the MG.
  
  InputNodes have no inputs. In other words, InputNodes are sources, information
  is "created" at the InputNode. Incoming links to the InputNode and assignment
  to self.num_inputs are therefore forbidden.
  
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
  
    self.main_output_sizes = self.get_output_sizes(state_sizes)
    self.batch_size = builder.batch_size
    self.max_steps = builder.max_steps if hasattr(builder, 'max_steps') else None
    self.is_sequence = is_sequence

    # Deal with sequences
    self.main_oshapes, self.D = self.get_main_oshapes()
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
    dirs = self.directives
    
    name = self.name
    out_shape = self.main_oshapes
#     dtype = self.dtype_dict[dirs['dtype']]
    for oslot, out_shape in enumerate(self.main_oshapes):
      self._oslot_to_otensor[oslot] = tf.placeholder(self.dtype,
                                                     shape=out_shape,
                                                     name=name)
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

    self.free_oslots = list(range(self.num_expected_outputs))

    self._update_default_directives(**dirs)
    
    self._declare_secondary_outputs()
    self.dist = None

  def _update_default_directives(self, **dirs):
    """
    Update the node directives
    """
    if self.D[0] == 1:
      oshape = self.main_oshapes[0]
      osize = oshape[-1]
    else:
      raise NotImplementedError("main output with rank > 1 is not implemented "
                                "for the Normal Input Node. ")
    
    if self.batch_size is None:
      dummy = tf.placeholder(self.dtype, oshape, self.name + '_dummy')
#       self.builder.dummies.add(dummy.name)
      self.builder.dummies[dummy.name] = oshape
    else:
      dummy = tf.zeros(oshape, self.dtype)
      
    if self.is_sequence and self.max_steps is None:
      mean_init = tf.zeros_like(dummy, dtype=self.dtype)
      scale = tf.eye(osize, dtype=self.dtype)
      scale_init = tf.linalg.LinearOperatorFullMatrix(scale)
    else:
      mean_init = tf.zeros_like(dummy, dtype=self.dtype)
      scale = tf.eye(osize, dtype=self.dtype)
      scale_init = tf.linalg.LinearOperatorFullMatrix(scale)

    self.directives = {'output_mean_name' : self.name + '_mean',
                       'output_scale_name' : self.name + '_scale',
                       'mean_init' : mean_init,
                       'scale_init' : scale_init}
    self.directives.update(dirs)
        
  def _declare_secondary_outputs(self):
    """
    Declare the statistics of the normal as secondary outputs. 
    """
    oshape = self.main_oshapes[0]
    
    self._oslot_to_shape[1] = oshape # mean oslot
    o1 = self.builder.addOutput(name=self.directives['output_mean_name'])
    self.builder.addDirectedLink(self, o1, oslot=1)
    
    self._oslot_to_shape[2] = oshape[1:-1] + [oshape[-1]]*2 # stddev oslot  
    o2 = self.builder.addOutput(name=self.directives['output_scale_name'])
    self.builder.addDirectedLink(self, o2, oslot=2)
  
  def _get_sample(self):
    """
    Get a sample from the distribution.
    """
#     return self.dist.sample(sample_shape=self.batch_size)
    return self.dist.sample()
  
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
    mean = self.directives['mean_init']
    scale = self.directives['scale_init']
    self.dist = dist = dist_dict['MultivariateNormalLinearOperator'](loc=mean,
                                                                     scale=scale)
    
    self._oslot_to_otensor[0] = dist.sample(sample_shape=(), name=self.name)
    assert self._oslot_to_otensor[0].shape.as_list() == self._oslot_to_shape[0] 
    
    self._oslot_to_otensor[1] = dist.loc
    assert self._oslot_to_otensor[1].shape.as_list() == self._oslot_to_shape[1]
    
    self._oslot_to_otensor[2] = dist.scale.to_dense()
    assert self._oslot_to_otensor[2].shape.as_list() == self._oslot_to_shape[2]

    self._is_built = True


if __name__ == '__main__':
  print(dist_dict.keys())  # @UndefinedVariable
  