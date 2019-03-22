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

import tensorflow as tf

from neurolib.models.dkf import DeepKalmanFilter

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 1
range_from = 0
range_to = 1
tests_to_run = list(range(range_from, range_to))

class DKFTestBuild(tf.test.TestCase):
  """
  """  
  def setUp(self):
    """
    """
    print()
    tf.reset_default_graph()
  
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_init(self):
    """
    """
    print("Test 0: DKF initialization")
    DeepKalmanFilter(input_dims=[[3]],
                     rnn_state_dims=[[5]],
                     ds_state_dim=[[4]])
  

if __name__ == '__main__':
  unittest.main(failfast=True)
