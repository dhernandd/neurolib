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

# NUM_TESTS : 8
range_from = 0
range_to = 8
tests_to_run = list(range(range_from, range_to))

class StaticModelBuilderBasicTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()
    
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_init(self):
    """
    Test adding basic InputNode
    """
    print("\nTest 0: Initialization")
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
  def test_addInner(self):
    """
    Test adding basic Deterministic InnerNode. 
    
    NOTE: The DeterministicNode belongs to the class of nodes that starts with a
    known final number of outputs.
    """
    print("\nTest 1: Adding InnerNode")
    try:
      builder = StaticBuilder(scope='Build0')
      builder.addInput(state_size=10, name="In")
      enc_name = builder.addInner(state_sizes=3, name="In")
    except AttributeError:
      print("\nCAUGHT! (AttributeError exception) \n"
            "Trying to assign the same name to two nodes!")
      builder = StaticBuilder(scope='Build0')
      builder.addInput(state_size=10, name="In")
      enc_name = builder.addInner(state_sizes=3, name="Det")

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
  
  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_addDirectedLinks(self):
    """
    Test adding DirectedLinks
    """
    print("\nTest 3: Adding DirectedLinks")
    builder = StaticBuilder(scope='Build0')
    in1 = builder.addInput(10, name="In")
    enc1 = builder.addInner(3, name="Det")
    
    builder.addDirectedLink(in1, enc1, islot=0)
    
    enc1 = builder.nodes[enc1]
    in1 = builder.nodes[in1]
    print('\nNode keys in builder:', list(builder.nodes.keys()))
    self.assertEqual(builder.num_nodes, 2, "The number of nodes has not been "
                     "assigned correctly")
    self.assertIn(1, builder.adj_list[0], "Node 1 has not been added to the "
                  "adjacency list of Node 0")
    
    print("builder.adj_matrix", builder.adj_matrix)
    print("in1._child_label_to_slot_pairs", in1._child_label_to_slot_pairs)

  @unittest.skipIf(3 not in tests_to_run, "Skipping")
  def test_BuildModel0(self):
    """
    Test building the simplest model possible.
    """
    print("\nTest 4: Building a Basic Model")
    builder = StaticBuilder(scope="Basic")
    in_name = builder.addInput(10)
    enc_name = builder.addInner(3)
    builder.addDirectedLink(in_name, enc_name, islot=0)
        
    inn, enc = builder.nodes[in_name], builder.nodes[enc_name]
    print("inn._child_label_to_slot_pairs", inn._child_label_to_slot_pairs)
    builder.build()
    print("enc._islot_to_itensor", enc._islot_to_itensor)
    self.assertEqual(inn._oslot_to_otensor['main'].shape.as_list()[-1],
                     enc._islot_to_itensor[0]['main'].shape.as_list()[-1], 
                     "The input tensors have not been assigned correctly")

  @unittest.skipIf(4 not in tests_to_run, "Skipping")
  def test_BuildModel2(self):
    """
    Builds a model with 2 inputs. Test ConcatNode
    """
    print("\nTest 6: Building a Model with Concat")
    builder = StaticBuilder("Concat")
    in1 = builder.addInput(10)
    in2 = builder.addInput(20)
    enc1 = builder.addInner(3, num_inputs=2)

    builder.addDirectedLink(in1, enc1, islot=0)
    builder.addDirectedLink(in2, enc1, islot=1)
    
    in1, in2, enc1 = builder.nodes[in1], builder.nodes[in2], builder.nodes[enc1]
    builder.build()
    print("enc1._islot_to_itensor", enc1._islot_to_itensor)
    
  @unittest.skipIf(5 not in tests_to_run, "Skipping")
  def test_BuildModel3(self):
    """
    Try to break it, the algorithm... !! Guess not mdrfkr.
    """
    print("\nTest 7: Building a more complicated Model")
    builder = StaticBuilder("BreakIt")
    in1 = builder.addInput(10)
    in2 = builder.addInput(20)
    enc1 = builder.addInner(3)
    enc2 = builder.addInner(5, num_inputs=2)
    
    builder.addDirectedLink(in1, enc1, islot=0)
    builder.addDirectedLink(in2, enc2, islot=0)
    builder.addDirectedLink(enc1, enc2, islot=1)
    
    builder.build()
    enc1, enc2 = builder.nodes[enc1], builder.nodes[enc2]
    print("enc1._islot_to_itensor", enc1._islot_to_itensor)
    print("enc1._islot_to_itensor", enc2._islot_to_itensor)


    
if __name__ == "__main__":
  unittest.main(failfast=True)
