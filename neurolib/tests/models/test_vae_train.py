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

from neurolib.models.vae import VariationalAutoEncoder 

# pylint: disable=bad-indentation, no-member, protected-access

NUM_TESTS = 1
test_to_run = 1 

class VAETestTrain(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()
  
  @unittest.skipIf(test_to_run != 1, "Skipping Test 0")
  def test_train(self):
    """
    """
    print("\nTest 0: VAE initialization")
    nsamps = 100
    idim = 3
    odim = 10
    x = 1.0*np.random.randn(nsamps, idim)
    W = np.random.randn(3, odim)
    y = np.tanh(np.dot(x, W) + 0.1*np.random.randn(nsamps, odim)) # + 3*x[:,1:]**2 + 0.5*np.random.randn(100,1)
    dataset = {'train_observation' : y[:80],
               'valid_observation' : y[80:]}
        
    vae = VariationalAutoEncoder(input_dim=3, output_dim=10)
    vae.build()
    vae.train(dataset, num_epochs=20)
    
if __name__ == '__main__':
  unittest.main(failfast=True)