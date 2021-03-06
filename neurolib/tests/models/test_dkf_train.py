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

from neurolib.models.dkf import DeepKalmanFilter

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 1
range_from = 0
range_to = 1
tests_to_run = list(range(range_from, range_to))

fname = '/datadict_gaussianobs2D'
with open(path + fname, 'rb') as f:
  datadict = pickle.load(f)

class DKFTestTrain(tf.test.TestCase):
  """
  """  
  def setUp(self):
    """
    """
    print()
    tf.reset_default_graph()
  
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_train(self):
    """
    """
    print("Test 0: DKF train")

    dataset = {}
    Ytrain = datadict['Ytrain']
    Yshape = Ytrain.shape
    print("Yshape", Yshape)
    dataset['train_Observation_0'] = Ytrain
    dataset['valid_Observation_0'] = datadict['Yvalid']
    max_steps, input_dims = Yshape[-2], Yshape[-1]
      
    dkf = DeepKalmanFilter(input_dims=[[input_dims]],
                           rnn_state_dims=[[10], [10]],
                           ds_state_dim=[[2]], # logs and save implemented
                           max_steps=max_steps,
                           batch_size=1,
                           save_on_valid_improvement=True) # OK!
    dkf.train(dataset, num_epochs=2)
    

if __name__ == '__main__':
  unittest.main(failfast=True)
  