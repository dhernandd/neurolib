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

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 2
range_from = 0
range_to = 2
tests_to_run = list(range(range_from, range_to))

class CallEncoderTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()

  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_CallDeterministic(self):
    """
    Test Calling a DeterministicNode
    """
    print("\nTest 0: Calling a Deterministic Node")
    builder = StaticBuilder(scope="Basic")
    in_name = builder.addInput(10)
    enc_name = builder.addInner(3)
    builder.addDirectedLink(in_name, enc_name, islot=0)
    
    self.assertEqual(builder.num_nodes, 2, "The number of nodes has not been "
                     "assigned correctly")    
    builder.build()

    X = tf.placeholder(tf.float32, [1, 10], 'X')
    enc = builder.nodes[enc_name]
    Y = enc((X))
    print('Y', Y)
    self.assertEqual(Y.shape[-1], 3, "")
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_CallCustom(self):
    """
    Test, calling a CustomNode
    """
    print("\nTest 1: Build")
    builder = StaticBuilder("MyModel")

    cust = builder.createCustomNode(num_inputs=1,
                                    num_outputs=1,
                                    name="Custom")
    cust_in1 = cust.addInner(3)
    cust_in2 = cust.addInner(4)
    cust.addDirectedLink(cust_in1, cust_in2, islot=0)

    cust.declareIslot(islot=0, innernode_name=cust_in1, inode_islot=0)
    cust.declareOslot(oslot='main', innernode_name=cust_in2, inode_oslot='main')
     
    in1 = builder.addInput(10)
    builder.addDirectedLink(in1, cust, islot=0)
    builder.build()
    
    X = tf.placeholder(tf.float32, [1, 10], 'X')
    Y = cust(X)  # pylint: disable=unpacking-non-sequence
    print('Y', Y)
    self.assertEqual(Y[0].shape[-1], 4, "")
    
    
if __name__ == "__main__":
  unittest.main(failfast=True)
