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
import os.path

my_path = os.path.abspath(os.path.dirname(__file__))

import unittest
import pickle

import tensorflow as tf

from neurolib.models.dkf import DeepKalmanFilter

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 2
range_from = 0
range_to = 1
tests_to_run = list(range(range_from, range_to))

class DKFTestTrain(tf.test.TestCase):
  """
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()
  
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_train(self):
    """
    """
    print("\nTest 1: DKF train")

    dataset = {}
    fname = my_path + '/datadict_gaussianobs2D'
    with open(fname, 'rb') as f:
      datadict = pickle.load(f)
      Ytrain = datadict['Ytrain']
      Yshape = Ytrain.shape
      print("Yshape", Yshape)
      dataset['train_observation_0'] = Ytrain
      dataset['valid_observation_0'] = datadict['Yvalid']
      max_steps, input_dims = Yshape[-2], Yshape[-1]
      
    dkf = DeepKalmanFilter(input_dims=[[input_dims]],
                           max_steps=max_steps,
                           batch_size=1,
                           state_dims=[[40], [4]]) # logs and save implemented
#                            keep_logs=True,
#                            save_on_valid_improvement=True,
#                            root_rslts_dir='./rslts/')
    dkf.build()
    dkf.train(dataset, num_epochs=10)
    

if __name__ == '__main__':
  unittest.main(failfast=True)
  