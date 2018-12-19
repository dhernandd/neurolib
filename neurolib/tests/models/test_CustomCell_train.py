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

from neurolib.encoder.seq_cells import TwoEncodersCell, NormalTriLCell
from neurolib.builders.sequential_builder import SequentialBuilder
from neurolib.trainer.gd_trainer import GDTrainer

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 2
range_from = 0
range_to = 3
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

def generate_echo_data_cat(num_labels, length, echo_step, max_steps, withY=True):
  """
  Generate some echo categorical data
  """
  p = num_labels*[1/num_labels]
  x = np.array(np.random.choice(num_labels, length, p=p))
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

class CustomCellTrainTest(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()
  
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_build_custom(self):
    """
    Test build Custom Cell
    """
    builder = SequentialBuilder(max_steps=25,
                                scope="Main")
    is1 = builder.addInputSequence([[1]])
    ev1 = builder.addEvolutionSequence([[2],[3]],
                                       num_inputs=3,
                                       num_outputs=2,
                                       cell_class=TwoEncodersCell)
    os1 = builder.addOutputSequence()
    os2 = builder.addOutputSequence()
    builder.addDirectedLink(is1, ev1, islot=2)
    builder.addDirectedLink(ev1, os1)
    builder.addDirectedLink(ev1, os2, oslot=1)
    builder.build()
  
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_train_custom(self):
    """
    Test train Custom cell.
    
    This test tests a configuration that exemplifies neurolib's capabilities. A
    CustomCell is defined consisting of two forward RNNs working in tandem. The
    second RNN also takes as inputs, the outputs of the first one (See
    TwoEncodersCell). The outputs of these two RNNS are passed to an MSE cost
    function.
    """
    scope = "Main"
    batch_size = 1
    max_steps = 25
    dataset = generate_echo_data(1000, 3, max_steps, withY=False)
    
    builder = SequentialBuilder(max_steps=max_steps,
                                scope=scope)
    is1 = builder.addInputSequence([[1]], name='inputSeq')
    ev1 = builder.addEvolutionSequence([[3],[3]],
                                       num_inputs=3,
                                       num_outputs=2,
                                       cell_class=TwoEncodersCell,
                                       o0_name='prediction',
                                       o1_name='response')
    os1 = builder.addOutputSequence(name='prediction')
    os2 = builder.addOutputSequence(name='response')
    builder.addDirectedLink(is1, ev1, islot=2)
    builder.addDirectedLink(ev1, os1)
    builder.addDirectedLink(ev1, os2, oslot=1)
    builder.build()
    
    cost = ('mse', ('prediction', 'response'))
    trainer = GDTrainer(builder,
                        cost,
                        name=scope,
                        batch_size=batch_size)
    dataset = trainer.prepare_datasets(dataset)
    trainer.train(dataset, num_epochs=10)

  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_train_custom1(self):
    """
    Test train Custom NormalTriLCell
    
    This test tests a configuration that exemplifies neurolib's capabilities. A
    CustomCell is defined that consists of a forward stochastic RNN.
    """
    scope = "Main"
    batch_size = 1
    max_steps = 25
    dataset = generate_echo_data(1000, 3, max_steps)
    
    builder = SequentialBuilder(max_steps=max_steps,
                                scope=scope)
    is1 = builder.addInputSequence([[1]], name='inputSeq')
    ev1 = builder.addEvolutionSequence([[3]],
                                       num_inputs=2,
                                       num_outputs=3,
                                       cell_class=NormalTriLCell,
                                       o0_name='prediction')
    inn1 = builder.addInnerSequence([[10]], 1)
    os1 = builder.addOutputSequence(name='prediction')
    builder.addDirectedLink(is1, ev1, islot=1)
    builder.addDirectedLink(ev1, inn1)
    builder.addDirectedLink(inn1, os1)
    
    is0 = builder.addInputSequence([[1]], name='outputSeq')
    os0 = builder.addOutputSequence(name='response')
    builder.addDirectedLink(is0, os0)
    builder.build()
    
    cost = ('mse', ('prediction', 'response'))
    trainer = GDTrainer(builder,
                        cost,
                        name=scope,
                        batch_size=batch_size)
    dataset = trainer.prepare_datasets(dataset)
    trainer.train(dataset, num_epochs=10)
    
if __name__ == '__main__':
  unittest.main(failfast=True) 