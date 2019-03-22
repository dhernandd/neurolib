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

from neurolib.models.regression import Regression

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 1
range_from = 0
range_to = 1
tests_to_run = list(range(range_from, range_to))

with open(path + '/datadict_regression', 'rb') as f1:
  dataset = pickle.load(f1)

class RegressionTestTrain(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """  
  def setUp(self):
    """
    """
    print()
    tf.reset_default_graph()
  
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_train(self):
    """
    Test train
    """
    print("Test 0: Regression train")
        
    dc = Regression(input_dim=2,
                    output_dim=1,
                    save_on_valid_improvement=False) # ok!
    dc.train(dataset,
             num_epochs=20) # train
    
    
if __name__ == '__main__':
  unittest.main(failfast=True)
