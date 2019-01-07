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

from neurolib.models.regression import Regression

# pylint: disable=bad-indentation, no-member, protected-access

NUM_TESTS = 1
run_test = 1

def generate_some_data():
  """
  """
  x = 10.0*np.random.randn(100, 2)
  y = x[:,0:1] + 1.5*x[:,1:]# + 3*x[:,1:]**2 + 0.5*np.random.randn(100,1)
  xtrain, xvalid, ytrain, yvalid = x[:80], x[80:], y[:80], y[80:]
  dataset = {'train_observation_in' : xtrain,
             'train_observation_out' : ytrain,
             'valid_observation_in' : xvalid,
             'valid_observation_out' : yvalid}
  return dataset

class RegressionTestTrain(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()
  
  @unittest.skipIf(run_test != 1, "Skipping")
  def test_train(self):
    """
    Test train
    """
    print("\nTest 0: Regression train")
    
    dataset = generate_some_data()
    
    dc = Regression(input_dim=2, output_dim=1)
    dc.build()
    dc.train(dataset, num_epochs=10) # train
    
    
if __name__ == '__main__':
  unittest.main(failfast=True)