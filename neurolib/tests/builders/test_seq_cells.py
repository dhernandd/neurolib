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

from neurolib.builders.static_builder import StaticBuilder
from neurolib.encoder.seq_cells import BasicEncoderCell, NormalTriLCell

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 3
range_from = 2
range_to = 3
tests_to_run = list(range(range_from, range_to))


class SeqCellsBasicTest(tf.test.TestCase):
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
    Create a Custom Cell
    """
    print("Test 0: Custom cell init")
    builder = StaticBuilder(scope='BuildCell')
    
    cell = BasicEncoderCell(builder, state_sizes=[[5]])
    print("cell.encoder", cell.encoder)
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_call(self):
    """
    Call a Custom Cell Node
    """
    print("Test 1: Custom cell call")
    builder = StaticBuilder(scope='BuildCell')
    
    cell = BasicEncoderCell(builder, state_sizes=[[5]])
    print("cell.encoder", cell.encoder)
    
    X = tf.placeholder(tf.float64, [None, 5])
    Y = tf.placeholder(tf.float64, [None, 10])
    Z = cell(X, Y)
    print("Z", Z)
    
  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_NormalTriLCell(self):
    """
    Test the NormalTriLCell Node
    """
    print("Test 2: ")
    builder = StaticBuilder(scope='BuildCell')
    
    cell = NormalTriLCell(builder,
                          state_sizes=[[5]])
    print("cell.encoder", cell.encoder)
    
    X = tf.placeholder(tf.float64, [None, 5])
    Y = tf.placeholder(tf.float64, [None, 10])
    Z = cell(X, Y)
    print("Z", Z)
    
if __name__ == "__main__":
  unittest.main(failfast=True)
    