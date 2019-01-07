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

# pylint: disable=bad-indentation, no-member, protected-access

def mse(node_dict, node_names):
  """
  Define the Mean Squared Error between two Outputs of the Model Graph
  """
  try:
    nodeY = node_dict[node_names[0]]
    nodeX = node_dict[node_names[1]]
  except AttributeError:
    raise AttributeError("You must define two OutputNodes, named {} and {}, for "
                         "'mse' training".format(node_names[0], node_names[1]))

  Y = nodeY.get_input(0)
  X = nodeX.get_input(0)
  
  return tf.reduce_mean((Y - X)**2, name="mse")

def mabsdiff(node_dict, node_names):
  """
  Define the Mean Squared Error between two Outputs of the Model Graph
  """
  try:
    nodeY = node_dict[node_names[0]]
    nodeX = node_dict[node_names[1]]
  except AttributeError:
    raise AttributeError("You must define two OutputNodes, named {} and {}, for "
                         "'mse' training".format(node_names[0], node_names[1]))

  Y = nodeY.get_input(0)
  X = nodeX.get_input(0)
  
  return tf.reduce_mean(tf.abs(Y - X)**2, name="mabsdiff")

def cross_entropy_with_logits(node_dict, node_names):
  """
  """
  try:
    nodeY = node_dict[node_names[0]]
    nodeX = node_dict[node_names[1]]
  except AttributeError:
    raise AttributeError("You must define two OutputNodes, named 'prediction' and "
                         "'response', for 'cross_entropy' training")

  Y = nodeY.get_input(0)
  X = nodeX.get_input(0)
  
  Y = tf.squeeze(Y, axis=-1)
  ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,
                                                      logits=X,
                                                      name='cross_entropy')
  return tf.reduce_mean(ce)

def mse_reg(node_dict, node_names):
  """
  """
  pass

def elbo(node_dict, node_names):
  """
  """
  try:
    node_gen = node_dict[node_names[0]]
    node_rec = node_dict[node_names[1]]
  except AttributeError:
    raise AttributeError("You must define two InnerNodes, named 'Recognition' and "
                         "'Generative', for 'elbo' training")
    
  nodeY = node_dict[node_names[2]]
  Y = nodeY.get_output(0)
  
  return tf.reduce_sum(-node_rec.entropy()) - tf.reduce_sum(node_gen.log_prob(Y))
#   return tf.reduce_sum(- node_gen.log_prob(Y))

def entropy(node_dict, node_names):
  """
  """
  node = node_dict[node_names[0]]
  return tf.reduce_sum(node.entropy())

def logprob(node_dict, node_names):
  """
  """
  node = node_dict[node_names[0]]
  nodeY = node_dict[node_names[1]]
  Y = nodeY.get_output(0)
  
  return node.log_prob(Y)
