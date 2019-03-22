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

# NUM_TESTS : 10
run_from = 0
run_to = 10
tests_to_run = list(range(run_from, run_to))

class CustomEncoderBuilderBasicTest(tf.test.TestCase):
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
    Create a CustomNode
    """
    print("Test 0: Initialization")
    builder = StaticBuilder(scope='BuildCust')
    i1 = builder.addInput([[3]], name='In1')
    cust = builder.createCustomNode(i1, 3, name="Custom")
    
    print("cust.name", cust.name)
    print("cust.num_expected_inputs", cust.num_expected_inputs)
    print("cust.num_expected_outputs", cust.num_expected_outputs)
  
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_add_encoder0(self):
    """
    Test commit
    """
    print("Test 1: Adding nodes to Custom")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    cust = builder.createCustomNode(inputs=i1,
                                     num_outputs=1,
                                     name="Custom")
    cust.addTransformInner(3, main_inputs=[0])
    print("cust.in_builder.inode_to_innernode", cust.in_builder.islot_to_innernode_names)

  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_add_encoder1(self):
    """
    Test commit
    """
    print("Test 2: Adding more nodes to Custom")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    cust = builder.createCustomNode(inputs=i1,
                                     num_outputs=1,
                                     name="Custom")
    ctin1 = cust.addTransformInner(3, [0])
    cust.addTransformInner(4, [ctin1])
    print("cust.in_builder._inode_to_innernode", cust.in_builder.islot_to_innernode_names)
    print("cust.in_builder.input_nodes", cust.in_builder.input_nodes)

  @unittest.skipIf(3 not in tests_to_run, "Skipping")
  def test_add_encoder2(self):
    """
    Test commit
    """
    print("Test 3: Declaring multiple inputs to the Custom Node")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    i2 = builder.addInput(5)
    cust = builder.createCustomNode(inputs=[i1,i2],
                                     num_outputs=1,
                                     name="Custom")
    ctin1 = cust.addTransformInner(3, [0])
    cust.addTransformInner(3, [1, ctin1])
    print("cust.in_builder._inode_to_innernode", cust.in_builder.islot_to_innernode_names)
    print("cust.in_builder.input_nodes", cust.in_builder.input_nodes)

  @unittest.skipIf(4 not in tests_to_run, "Skipping")
  def test_build0(self):
    """
    Test commit
    """
    print("Test 4: Basic Custom build")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    cust = builder.createCustomNode(inputs=i1,
                                     num_outputs=1,
                                     name="Custom")
    ctin = cust.addTransformInner(3, main_inputs=[0])
    cust.declareOslot(0, ctin, 'main')
    
    builder.build()
    
  @unittest.skipIf(5 not in tests_to_run, "Skipping")
  def test_build1(self):
    """
    Test build
    """
    print("Test 5: Slightly more complicated Custom build")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    cust = builder.createCustomNode(inputs=i1,
                                     num_outputs=1,
                                     name="Custom")
    ctin1 = cust.addTransformInner(3, main_inputs=[0])
    ctin2 = cust.addTransformInner(4, main_inputs=ctin1)
    
    cust.declareOslot(oslot=0, innernode_name=ctin2, inode_oslot_name='main')
    
    builder.build()
          
  @unittest.skipIf(6 not in tests_to_run, "Skipping")
  def test_build2(self):
    """
    Test build
    """
    print("Test 6: : Slightly more complicated Custom build")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    cust = builder.createCustomNode(inputs=i1,
                                     num_outputs=2,
                                     name="Custom")
    ctin1 = cust.addTransformInner(3, main_inputs=[0])
    ctin2 = cust.addTransformInner(4, main_inputs=ctin1)
    
    cust.declareOslot(oslot=0, innernode_name=ctin2, inode_oslot_name='main')
    cust.declareOslot(oslot=1, innernode_name=ctin1, inode_oslot_name='main')
    
    builder.build()
    
  @unittest.skipIf(7 not in tests_to_run, "Skipping")
  def test_build3(self):
    """
    Test build
    """
    print("Test 7: Slightly more complicated Custom build")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    i2 = builder.addInput(5)
    cust = builder.createCustomNode(inputs=[i1,i2],
                                     num_outputs=1,
                                     name="Custom")
    ctin1 = cust.addTransformInner(3, [0])
    ctin2 = cust.addTransformInner(3, [1, ctin1])
    cust.declareOslot(oslot=0, innernode_name=ctin2, inode_oslot_name='main')
    
    builder.build()
    
  @unittest.skipIf(8 not in tests_to_run, "Skipping")
  def test_build_outputs(self):
    """
    Test commit
    """
    print("Test 8: Building outputs")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    cust = builder.createCustomNode(inputs=i1,
                                     num_outputs=1,
                                     name="Custom")
    ctin = cust.addTransformInner(3, main_inputs=[0])
    cust.declareOslot(0, ctin, 'main')
    
    builder.build()
    
    X = tf.placeholder(tf.float64, [None, 10]) 
    cust.build_outputs(imain0=X)
    
  @unittest.skipIf(9 not in tests_to_run, "Skipping")
  def test_build_outputs1(self):
    """
    Test build
    """
    print("Test 9: Building outputs")
    builder = StaticBuilder("MyModel")

    i1 = builder.addInput(10)
    i2 = builder.addInput(5)
    cust = builder.createCustomNode(inputs=[i1,i2],
                                     num_outputs=1,
                                     name="Custom")
    ctin1 = cust.addTransformInner(3, [0])
    ctin2 = cust.addTransformInner(3, [0, ctin1, 1])
    cust.declareOslot(oslot=0, innernode_name=ctin2, inode_oslot_name='main')
    
    builder.build()
    
    X = tf.placeholder(tf.float64, [None, 10]) 
    Y = tf.placeholder(tf.float64, [None, 5]) 
    cust.build_outputs(imain0=X, imain1=Y)
    
if __name__ == "__main__":
  unittest.main(failfast=True)
