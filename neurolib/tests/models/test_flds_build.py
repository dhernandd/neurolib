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

from neurolib.models.flds import fLDS

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 2
range_from = 0
range_to = 2
tests_to_run = list(range(range_from, range_to))

class FLDSTestBuild(tf.test.TestCase):
  """
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()
  
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_init(self):
    """
    """
    print("\nTest 0: DKF initialization")
    fLDS(input_dims=[[10]],
         state_dims=[[3]])
  
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_build(self):
    """
    """
    print("\nTest 1: DKF build")
    flds = fLDS(input_dims=[[10]],
                state_dims=[[3]])
    flds.build()


if __name__ == '__main__':
  unittest.main(failfast=True)
  