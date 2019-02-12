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

# NUM_TESTS : 4
run_from = 0
run_to = 4
tests_to_run = list(range(run_from, run_to))

class CustomEncoderBuilderBasicTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()
    
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_init(self):
    """
    Create a CustomNode
    """
    print("\nTest 0: Initialization")
    builder = StaticBuilder(scope='BuildCust')
    builder.createCustomNode(1, 1, name="Custom")
  
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_add_encoder0(self):
    """
    Test commit
    """
    print("\nTest 1: Adding nodes to custom")
    builder = StaticBuilder("MyModel")

    builder.addInput(10)
    cust = builder.createCustomNode(num_inputs=1,
                                    num_outputs=1,
                                    name="Custom")
    cust_in1 = cust.addInner(3)
    cust_in2 = cust.addInner(4)
    cust.addDirectedLink(cust_in1, cust_in2, islot=0)

    cust.declareIslot(islot=0, innernode_name=cust_in1, inode_islot=0)
    cust.declareOslot(oslot='main', innernode_name=cust_in2, inode_oslot='main')

  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_build_outputs(self):
    """
    Test commit
    """
    print("\nTest 1: Building outputs")
    builder = StaticBuilder("MyModel")

    builder.addInput(10)
    cust = builder.createCustomNode(num_inputs=1,
                                    num_outputs=1,
                                    name="Custom")
    cust_in1 = cust.addInner(3)
    cust_in2 = cust.addInner(4)
    cust.addDirectedLink(cust_in1, cust_in2, islot=0)

    cust.declareIslot(islot=0, innernode_name=cust_in1, inode_islot=0)
    cust.declareOslot(oslot='main', innernode_name=cust_in2, inode_oslot='main')
    
    ipt = [{'main' : tf.placeholder(tf.float64, [1, 3])}]
    rslt = cust.get_outputs(ipt)
    print("rslt", rslt)
      
  @unittest.skipIf(3 not in tests_to_run, "Skipping")
  def test_add_encoder1(self):
    """
    Test build
    """
    print("\nTest 2: Build")
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
    
    
if __name__ == "__main__":
  unittest.main(failfast=True)
