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

from neurolib.models.regression import Regression

# pylint: disable=bad-indentation, no-member, protected-access

NUM_TESTS = 2
run_up_to_test = 2
tests_to_run = list(range(run_up_to_test))

class RegressionFullTest(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()

  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_init(self):
    """
    Test initialization
    """
    print("\nTest 0: Regression initialization")
    Regression(input_dim=10, output_dim=1)

  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_build(self):
    """
    Test build
    """
    print("\nTest 1: Regression build")
    model = Regression(input_dim=10, output_dim=1)
    model.build()


if __name__ == '__main__':
  unittest.main(failfast=True)