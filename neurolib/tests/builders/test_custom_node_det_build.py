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
run_from = 4
run_to = 5
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
    i1 = builder.addInput([[3]], name='In1')
    cust = builder.createCustomNode2(i1, 3, name="Custom")
    
    print("cust.name", cust.name)
    print("cust.num_expected_inputs", cust.num_expected_inputs)
    print("cust.num_expected_outputs", cust.num_expected_outputs)
  
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_add_encoder0(self):
    """
    Test commit
    """
    print("\nTest 1: Adding nodes to custom")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    cust = builder.createCustomNode2(inputs=i1,
                                     num_outputs=1,
                                     name="Custom")
    cust.addTransformInner(3, main_inputs=[0])
    print("cust.in_builder.inode_to_innernode", cust.in_builder.inode_to_innernode_names)
#     ctin2 = cust.addInner(4, ctin1)
#     cust.addDirectedLink(cust_in1, cust_in2, islot=0)

#     cust.declareIslot(islot=0, innernode_name=ctin1, inode_islot=0)
#     cust.declareOslot(oslot='main', innernode_name=cust_in2, inode_oslot='main')

  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_add_encoder1(self):
    """
    Test commit
    """
    print("\nTest 1: Adding nodes to custom")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    cust = builder.createCustomNode2(inputs=i1,
                                     num_outputs=1,
                                     name="Custom")
    ctin1 = cust.addTransformInner(3, [0])
    cust.addTransformInner(4, [ctin1])
    print("cust.in_builder._inode_to_innernode", cust.in_builder.inode_to_innernode_names)
    print("cust.in_builder.input_nodes", cust.in_builder.input_nodes)
#     cust.addDirectedLink(cust_in1, cust_in2, islot=0)

#     cust.declareIslot(islot=0, innernode_name=ctin1, inode_islot=0)
#     cust.declareOslot(oslot='main', innernode_name=cust_in2, inode_oslot='main')

  @unittest.skipIf(3 not in tests_to_run, "Skipping")
  def test_add_encoder2(self):
    """
    Test commit
    """
    print("\nTest 1: Adding nodes to custom")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    i2 = builder.addInput(5)
    cust = builder.createCustomNode2(inputs=[i1,i2],
                                     num_outputs=1,
                                     name="Custom")
    ctin1 = cust.addTransformInner(3, [0])
    cust.addTransformInner(3, [1, ctin1])
    print("cust.in_builder._inode_to_innernode", cust.in_builder.inode_to_innernode_names)
    print("cust.in_builder.input_nodes", cust.in_builder.input_nodes)
#     cust.addDirectedLink(cust_in1, cust_in2, islot=0)

#     cust.declareIslot(islot=0, innernode_name=ctin1, inode_islot=0)
#     cust.declareOslot(oslot='main', innernode_name=cust_in2, inode_oslot='main')

  @unittest.skipIf(4 not in tests_to_run, "Skipping")
  def test_build0(self):
    """
    Test commit
    """
    print("\nTest 1: Adding nodes to custom")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    cust = builder.createCustomNode2(inputs=i1,
                                     num_outputs=1,
                                     name="Custom")
    ctin = cust.addTransformInner(3, main_inputs=[0])
    cust.declareOslot(0, ctin, 'main')
    
    builder.build()

  @unittest.skipIf(5 not in tests_to_run, "Skipping")
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
    rslt = cust.build_outputs(ipt)
    print("rslt", rslt)
      
  @unittest.skipIf(5 not in tests_to_run, "Skipping")
  def test_add_encoder4(self):
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
