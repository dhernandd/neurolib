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
import os
path = os.path.dirname(os.path.realpath(__file__))
import unittest
import pickle

import tensorflow as tf

from neurolib.models.predictor_rnn import PredictorRNN

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS: 3
range_from = 2
range_to = 3
tests_to_run = list(range(range_from, range_to))

with open(path + '/datadict_seq01', 'rb') as f1:
  dataset = pickle.load(f1)
with open(path + '/datadict_seq01cat', 'rb') as f1:
  dataset_cat = pickle.load(f1)

class RNNPredictorTrainTest(tf.test.TestCase):
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
    Test train Basic RNN : continuous data
    """
    max_steps = dataset['train_Observation'].shape[1]
    
    model = PredictorRNN(input_dims=1,
                         state_dims=20,
                         output_dims=1,
                         max_steps=max_steps,
                         save_on_valid_improvement=False) # ok!

    print(dataset.keys())
    
    model.train(dataset, num_epochs=10)
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_train1(self):
    """
    Test train Basic RNN : categorical data
    """
    max_steps = dataset['train_Observation'].shape[1]
    num_labels = 4
    
    model = PredictorRNN(input_dims=1,
                         state_dims=20,
                         output_dims=1,
                         max_steps=max_steps,
                         num_labels=num_labels,
                         is_categorical=True,
                         save_on_valid_improvement=False) # OK!
    model.train(dataset_cat, num_epochs=10)
    
  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_train2(self):
    """
    Test train LSTM RNN : continuous outputs
    """
    max_steps = dataset['train_Observation'].shape[1]
    
    model = PredictorRNN(input_dims=1,
                         state_dims=20,
                         output_dims=1,
                         batch_size=1,
                         max_steps=max_steps,
                         cell_class='lstm',
                         save_on_valid_improvement=False) # OK!
    model.train(dataset, num_epochs=10)
    
    
if __name__ == '__main__':
  unittest.main(failfast=True)
  