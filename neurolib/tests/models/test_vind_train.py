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

from neurolib.models.vind import VIND

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 2
range_from = 1
range_to = 2
tests_to_run = list(range(range_from, range_to))


class VINDTestTrain(tf.test.TestCase):
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
    fname = '/datadict_gaussianobs2D'
    with open(path + fname, 'rb') as f:
      datadict = pickle.load(f)


    print("\nTest 1: VIND train")

    dataset = {}
    Ytrain = datadict['Ytrain']
    Yshape = Ytrain.shape
    print("Yshape", Yshape)
    dataset['train_Observation'] = Ytrain
    dataset['valid_Observation'] = datadict['Yvalid']
    max_steps, input_dims = Yshape[-2], Yshape[-1]
      
    vind = VIND(input_dims=input_dims,
                state_dim=[[2]],
                max_steps=max_steps,
                save_on_valid_improvement=False)
    vind.train(dataset, num_epochs=5)
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_train_wi(self):
    """
    """
    fname = '/datadict_pendulum_wi'
    with open(path + fname, 'rb') as f:
      datadict = pickle.load(f)

    print("\nTest 1: VIND train winputs")

    Ytrain = datadict['train_Observation']
    Yshape = Ytrain.shape
    print("Yshape", Yshape)
    max_steps, input_dims = Yshape[-2], Yshape[-1]
      
    vind = VIND(input_dims=[[input_dims], [1]],
                state_dim=[[2]],
                max_steps=max_steps,
                save_on_valid_improvement=False)
    vind.train(datadict, num_epochs=5)
    

if __name__ == '__main__':
  unittest.main(failfast=True)
  