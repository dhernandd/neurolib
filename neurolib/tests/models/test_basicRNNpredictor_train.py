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

from neurolib.models.predictor_rnn import PredictorRNN

# pylint: disable=bad-indentation, no-member, protected-access

NUM_TESTS = 2
run_up_to_test = 2
tests_to_run = list(range(run_up_to_test))

def generate_echo_data(length, echo_step, max_steps):
  """
  Generate some echo data
  """
  x = np.array(np.random.normal(size=length))
  y = np.roll(x, echo_step)
  y[0:echo_step] = 0
  
  X = np.reshape(x, [-1, max_steps, 1])
  Y = np.reshape(y, [-1, max_steps, 1])

  dataset = {'train_inputSeq' : X[:300],
             'train_outputSeq' : Y[:300],
             'valid_inputSeq' : X[300:],
             'valid_outputSeq' : Y[300:]}

  return dataset

def generate_echo_data_cat(num_labels, length, echo_step, max_steps):
  """
  Generate some echo categorical data
  """
  p = num_labels*[1/num_labels]
  x = np.array(np.random.choice(num_labels, length, p=p))
  y = np.roll(x, echo_step)
  y[0:echo_step] = 0

  X = np.reshape(x, [-1, max_steps, 1])
  Y = np.reshape(y, [-1, max_steps, 1])

  dataset = {'train_inputSeq' : X[:300],
             'train_outputSeq' : Y[:300],
             'valid_inputSeq' : X[300:],
             'valid_outputSeq' : Y[300:]}

  return dataset


class RNNClassifierTrainTest(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()
  
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_train0(self):
    """
    Test train RNN : continuous outputs
    """
    max_steps = 25
    echo_step = 3
    dataset = generate_echo_data(10000, echo_step, max_steps)
    
    model = PredictorRNN(input_dims=1,
                         state_dims=20,
                         output_dims=1,
                         batch_size=1,
                         max_steps=max_steps)
    model.build()
    model.train(dataset, num_epochs=20)
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_train1(self):
    """
    Test train RNN : categorical data
    """
    num_labels = 4
    max_steps = 25
    echo_step = 3
    dataset = generate_echo_data_cat(num_labels, 10000, echo_step, max_steps)
    
    model = PredictorRNN(input_dims=1,
                         state_dims=20,
                         output_dims=1,
                         batch_size=1,
                         max_steps=max_steps,
                         num_labels=num_labels,
                         is_categorical=True)
    model.build()
    model.train(dataset, num_epochs=20)
    

    
if __name__ == '__main__':
  unittest.main(failfast=True) 