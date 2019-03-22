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
from neurolib.encoder.input import NormalInputNode

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 6
range_from = 0
range_to = 6
tests_to_run = list(range(range_from, range_to))

class StaticModelBuilderBasicTest(tf.test.TestCase):
  """
  Test the building of static Models (no sequences)
  """
  def setUp(self):
    """
    """
    print()
    tf.reset_default_graph()
    
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_init(self):
    """
    Test adding basic InputNode
    """
    print("Test 0: Initialization")
    builder = StaticBuilder(scope='Build0')
    in1_name = builder.addInput(state_size=10)
    in1 = builder.input_nodes[in1_name]
    
    print('Node keys in builder:', list(builder.input_nodes.keys()))
    self.assertEqual(in1.label, 0, "The label has not been assigned correctly")
    self.assertEqual(builder.num_nodes, 1, "The number of nodes has not been "
                     "assigned correctly")
    self.assertEqual(in1.num_declared_outputs, 0, "The number of outputs of "
                     "the InputNode has not been assigned correctly")
    self.assertEqual(in1.num_declared_inputs, 0, "The number of outputs of "
                     "the InputNode has not been assigned correctly")
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_init1(self):
    """
    Test adding basic InputNode
    """
    print("Test 1: Simple build")
    builder = StaticBuilder(scope='Build')
    in1_name = builder.addInput(state_size=10,
                                iclass=NormalInputNode)
    in1 = builder.input_nodes[in1_name]
    
    print('Node keys in builder:', list(builder.input_nodes.keys()))
    self.assertEqual(in1.label, 0, "The label has not been assigned correctly")
    self.assertEqual(builder.num_nodes, 1, "The number of nodes has not been "
                     "assigned correctly")
    self.assertEqual(in1.num_declared_outputs, 0, "The number of outputs of "
                     "the InputNode has not been assigned correctly")
    self.assertEqual(in1.num_declared_inputs, 0, "The number of outputs of "
                     "the InputNode has not been assigned correctly")
    builder.build()
    
    print('in1._oslot_to_otensor', in1._oslot_to_otensor)
    
  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_addInner(self):
    """
    Test adding basic Deterministic InnerNode. 
    
    NOTE: The DeterministicNode belongs to the class of nodes that starts with a
    known final number of outputs.
    """
    print("Test 2: Adding InnerNode")
    try:
      builder = StaticBuilder(scope='Build0')
      i1 = builder.addInput(state_size=10, name="In")
      enc_name = builder.addTransformInner(state_sizes=3,
                                           main_inputs=i1,
                                           name="In")
    except AttributeError:
      print("\nCAUGHT! (AttributeError exception) \n"
            "Trying to assign the same name to two nodes!")
      builder = StaticBuilder(scope='Build0')
      i1 = builder.addInput(state_size=10, name="In")
      enc_name = builder.addTransformInner(state_size=3,
                                           main_inputs=i1)

    enc1 = builder.nodes[enc_name]
    print('\nNode keys in builder:', list(builder.nodes.keys()))
    print("This node's key:", enc_name)
    print("Builder adjacency matrix", builder.adj_matrix)
    self.assertEqual(enc1.label, 1, "The label has not been assigned correctly")
    self.assertEqual(builder.num_nodes, 2, "The number of nodes has not been "
                     "assigned correctly")
    self.assertEqual(enc1.num_declared_outputs, 0, "The number of outputs of the "
                     "DeterministicNode has not been assigned correctly")
    self.assertEqual(enc1.num_declared_inputs, 0, "The number of inputs of the "
                     "DeterministicNode has not been assigned correctly")
  
  @unittest.skipIf(3 not in tests_to_run, "Skipping")
  def test_BuildModel0(self):
    """
    Test building the simplest model possible.
    """
    print("Test 3: Building a Basic Model")
    builder = StaticBuilder(scope="Basic")
    in_name = builder.addInput(10)
    enc_name = builder.addTransformInner(3,
                                         main_inputs=in_name)
        
    inn, enc = builder.nodes[in_name], builder.nodes[enc_name]
    builder.build()
    print("enc._islot_to_itensor", enc._islot_to_itensor)
    self.assertEqual(inn._oslot_to_otensor['main'].shape.as_list()[-1],
                     enc._islot_to_itensor[0]['main'].shape.as_list()[-1], 
                     "The input tensors have not been assigned correctly")

  @unittest.skipIf(4 not in tests_to_run, "Skipping")
  def test_BuildModel2(self):
    """
    Builds a model with 2 inputs. Test concatenation
    """
    print("Test 4: Building a Model with concat")
    builder = StaticBuilder("Concat")
    in1 = builder.addInput(10)
    in2 = builder.addInput(20)
    enc1 = builder.addTransformInner(3,
                                     main_inputs=[in1, in2])
    
    in1, in2, enc1 = builder.nodes[in1], builder.nodes[in2], builder.nodes[enc1]
    builder.build()
    print("enc1._islot_to_itensor", enc1._islot_to_itensor)
    print("enc1._oslot_to_otensor", enc1._oslot_to_otensor)
    
  @unittest.skipIf(6 not in tests_to_run, "Skipping")
  def test_BuildModel3(self):
    """
    Try to break it, the algorithm... !! Guess not mdrfkr.
    """
    print("Test 7: Building a more complicated Model")
    builder = StaticBuilder("BreakIt")
    in1 = builder.addInput(10)
    in2 = builder.addInput(20)
    enc1 = builder.addTransformInner(3, main_inputs=in1)
    enc2 = builder.addTransformInner(5, main_inputs=[in2, enc1])
    
    builder.build()
    enc1, enc2 = builder.nodes[enc1], builder.nodes[enc2]
    print("enc1._islot_to_itensor", enc1._islot_to_itensor)
    print("enc1._islot_to_itensor", enc2._islot_to_itensor)
    print("enc1._oslot_to_otensor", enc1._oslot_to_otensor)
    print("enc1._oslot_to_otensor", enc2._oslot_to_otensor)

    
if __name__ == "__main__":
  unittest.main(failfast=True)
