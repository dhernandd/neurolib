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
from neurolib.encoder.normal import LDSNode, NormalPrecisionNode
from neurolib.builders.sequential_builder import SequentialBuilder

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 2
range_from = 0
range_to = 2
tests_to_run = list(range(range_from, range_to))

class NormalNodeTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()

  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_LDSNode(self):
    """
    """
    builder = StaticBuilder(scope='Main')
    i1 = builder.addInput([[3]])
    in1 = builder.addInner([[3]], node_class=LDSNode)
    builder.addDirectedLink(i1, in1, islot=0)
    
    builder.build()
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_NormalPrecisionNode_init(self):
    """
    Test Merge Node initialization
    """
    builder = SequentialBuilder(max_steps=30,
                                scope='Main')
    is1 = builder.addInputSequence([[3]])
    ins2 = builder.addInnerSequence([[3]],
                                    num_inputs=1,
                                    node_class=NormalPrecisionNode)
    builder.addDirectedLink(is1, ins2, islot=0)
    builder.build()


if __name__ == '__main__':
  unittest.main(failfast=True)
