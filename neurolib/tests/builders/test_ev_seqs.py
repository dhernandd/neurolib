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
import unittest

import numpy as np
import tensorflow as tf

from neurolib.builders.sequential_builder import SequentialBuilder
from neurolib.encoder.evolution_sequence import NonlinearDynamicswGaussianNoise
from neurolib.utils.graphs import get_session

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 1
range_from = 0
range_to = 1
tests_to_run = list(range(range_from, range_to))
test_to_run = 10

def generate_echo_data(length, echo_step, max_steps, withY=True):
  """
  Generate some echo data
  """
  x = np.array(np.random.normal(size=length))
  y = np.roll(x, echo_step)
  y[0:echo_step] = 0
  
  X = np.reshape(x, [-1, max_steps, 1])
  Y = np.reshape(y, [-1, max_steps, 1])
  
  train_idx = 4*length//(5*max_steps)
  dataset = {'train_inputSeq' : X[:train_idx],
             'valid_inputSeq' : X[train_idx:]}
  if not withY:
    return dataset
  dataset.update({'train_outputSeq' : Y[:train_idx],
                  'valid_outputSeq' : Y[train_idx:]})
  return dataset


class EvolutionSequenceTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()

  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_evalLogProb(self):
    """
    """
    max_steps = 25
    dataset = generate_echo_data(1000, 3, max_steps)
    Y = dataset['train_outputSeq']
    Y2 = np.tile(Y, (1, 1, 2))
    
    idims = [[1]]
    builder = SequentialBuilder(scope='Test',
                                max_steps=max_steps)
    
    is1 = builder.addInputSequence(idims, name='inputSeq')
    evs1 = builder.addEvolutionSequence([[2]],
                                        num_inputs=2,
                                        ev_seq_class=NonlinearDynamicswGaussianNoise,
                                        name='EvSeq')
    os1 = builder.addOutputSequence()
    builder.addDirectedLink(is1, evs1, islot=1)
    builder.addDirectedLink(evs1, os1)
    
    Z = tf.placeholder(tf.float64, [None, 25, 2], name='Z')
    builder.build()
    print(builder.nodes)
    evnode = builder.nodes[evs1]
    entropy = evnode.entropy()
    log_prob = evnode.log_prob(Z)
    
    print(entropy)
    print("builder.dummies", builder.dummies)
    print("Y.shape", Y.shape)
    sess = get_session()
    sess.run(tf.global_variables_initializer())
    ent = sess.run(entropy, feed_dict={'Test/inputSeq:0' : Y,
                                       'Normal_2_dummy:0' : np.zeros((32, 2))})
    lp = sess.run(log_prob, feed_dict={'Test/inputSeq:0' : Y,
                                       'Normal_2_dummy:0' : np.zeros((32, 2)),
                                       'Z:0' : Y2})
#     print('entropy', ent)
    print('entropy, logprob', ent, lp)
    
if __name__ == "__main__":
  unittest.main(failfast=True)
