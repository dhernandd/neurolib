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

from neurolib.models.flds import fLDS

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 1
range_from = 0
range_to = 1
tests_to_run = list(range(range_from, range_to))

class fLDSTestTrain(tf.test.TestCase):
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
    print("\nTest 1: fLDS build")

    dataset = {}
    fname = my_path + '/datadict_gaussianobs2D'
    with open(fname, 'rb') as f:
      datadict = pickle.load(f)
      Ytrain = datadict['Ytrain']
      Yshape = Ytrain.shape
      dataset['train_observation'] = Ytrain
      dataset['valid_observation'] = datadict['Yvalid']
      max_steps, input_dims = Yshape[-2], Yshape[-1]
      
    flds = fLDS(input_dims=[[input_dims]],
                max_steps=max_steps,
                state_dims=[[3]])
    flds.build()
    flds.train(dataset, num_epochs=10)
    

if __name__ == '__main__':
  unittest.main(failfast=True)
  