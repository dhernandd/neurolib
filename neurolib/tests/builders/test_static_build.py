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
from neurolib.encoder.normal import NormalTriLNode

# pylint: disable=bad-indentation, no-member, protected-access

NUM_TESTS = 10
run_up_to_test = 10
tests_to_run = list(range(run_up_to_test))

class StaticModelBuilderBasicTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()
    
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test0_init(self):
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
    self.assertEqual(enc1.label, 1, "The label has not been assigned correctly")
    self.assertEqual(builder.num_nodes, 2, "The number of nodes has not been "
                     "assigned correctly")
    self.assertEqual(enc1.num_declared_outputs, 0, "The number of outputs of the "
                     "DeterministicNode has not been assigned correctly")
    self.assertEqual(enc1.num_declared_inputs, 0, "The number of inputs of the "
                     "DeterministicNode has not been assigned correctly")

  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_addOutput(self):
    """
    Test adding basic OutputNode
    """
    print("\nTest 2: Adding OutputNode")
    builder = StaticBuilder(scope='Build0')
    builder.addInput(10, name="In")
    builder.addInner(3, name="Det")
    o_name = builder.addOutput(name="Out")
    
    o1 = builder.nodes[o_name]
    print("\nNode keys in builder:", list(builder.nodes.keys()))
    print("This node's key:", o_name)
    self.assertEqual(o1.label, 2, "The label has not been assigned correctly")
    self.assertEqual(builder.num_nodes, 3, "The number of nodes has not been "
                     "assigned correctly")
    self.assertEqual(o1.num_declared_outputs, 0, "The number of outputs of the "
                     "OutputNode has not been assigned correctly")
    self.assertEqual(o1.num_declared_inputs, 0, "The number of inputs of the "
                     "OutputNode has not been assigned correctly")
  
  @unittest.skipIf(3 not in tests_to_run, "Skipping")
  def test_addDirectedLinks(self):
    """
    Test adding DirectedLinks
    """
    print("\nTest 3: Adding DirectedLinks")
    builder = StaticBuilder(scope='Build0')
    in1 = builder.addInput(10, name="In")
    enc1 = builder.addInner(3, name="Det")
    out1 = builder.addOutput(name="Out")
    
    builder.addDirectedLink(in1, enc1)
    builder.addDirectedLink(enc1, out1)
    
    enc1 = builder.nodes[enc1]
    print('\nNode keys in builder:', list(builder.nodes.keys()))
    self.assertEqual(builder.num_nodes, 3, "The number of nodes has not been "
                     "assigned correctly")
    self.assertEqual(enc1.num_declared_outputs, 1, "The number of outputs of the "
                     "DeterministicNode has not been assigned correctly")
    self.assertEqual(enc1.num_declared_inputs, 1, "The number of inputs of the "
                     "DeterministicNode has not been assigned correctly")
    self.assertIn(1, builder.adj_list[0], "Node 1 has not been added to the "
                  "adjacency list of Node 0")
    self.assertIn(2, builder.adj_list[1], "Node 2 has not been added to the "
                  "adjacency list of Node 1")

  @unittest.skipIf(4 not in tests_to_run, "Skipping")
  def test_BuildModel0(self):
    """
    Test building the simplest model possible.
    """
    print("\nTest 4: Building a Basic Model")
    builder = StaticBuilder(scope="Basic")
    in_name = builder.addInput(10)
    enc_name = builder.addInner(3)
    out_name = builder.addOutput()
    builder.addDirectedLink(in_name, enc_name)
    builder.addDirectedLink(enc_name, out_name)
    
    self.assertEqual(builder.num_nodes, 3, "The number of nodes has not been "
                     "assigned correctly")
    
    builder.build()
    inn, enc, out = ( builder.nodes[in_name], builder.nodes[enc_name],
                      builder.nodes[out_name] )
    self.assertEqual(inn._oslot_to_otensor[0].shape.as_list()[-1],
                     enc._islot_to_itensor[0].shape.as_list()[-1], 
                     "The input tensors have not been assigned correctly")
    self.assertEqual(enc._oslot_to_otensor[0].shape.as_list()[-1],
                     out._islot_to_itensor[0].shape.as_list()[-1], 
                     "The input tensors have not been assigned correctly")
    
  @unittest.skipIf(5 not in tests_to_run, "Skipping")
  def test_BuildModel1(self):
    """
    Test building a model with 2 outputs. Test Cloning an output
    """
    print("\nTest 5: Building a Model with cloning")
    builder = StaticBuilder("Clone")
    in1 = builder.addInput(10)
    enc1 = builder.addInner(3)
    out1 = builder.addOutput(name="Out1")
    out2 = builder.addOutput(name="Out2")
    
    builder.addDirectedLink(in1, enc1)
    builder.addDirectedLink(enc1, out1)
    builder.addDirectedLink(enc1, out2)
    
    builder.build()

  @unittest.skipIf(6 not in tests_to_run, "Skipping")
  def test_BuildModel2(self):
    """
    Builds a model with 2 inputs. Test ConcatNode
    """
    print("\nTest 6: Building a Model with Concat")
    builder = StaticBuilder("Concat")
    in1 = builder.addInput(10)
    in2 = builder.addInput(20)
    enc1 = builder.addInner(3,
                            num_inputs=2)
    out1 = builder.addOutput()

    builder.addDirectedLink(in1, enc1, islot=0)
    builder.addDirectedLink(in2, enc1, islot=1)
    builder.addDirectedLink(enc1, out1)
    
    builder.build()
    
  @unittest.skipIf(7 not in tests_to_run, "Skipping")
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
    out1 = builder.addOutput()
    out2 = builder.addOutput()
    
    builder.addDirectedLink(in1, enc1)
    builder.addDirectedLink(in2, enc2, islot=0)
    builder.addDirectedLink(enc1, enc2, islot=1)
    builder.addDirectedLink(enc1, out1)
    builder.addDirectedLink(enc2, out2)
    
    builder.build()

  @unittest.skipIf(8 not in tests_to_run, "Skipping")
  def test_BuildModel4(self):
    """
    Test building the simplest stochastic model possible.
    """
    print("\nTest 8: Building a Model with a Stochastic Node")
    builder = StaticBuilder(scope="BasicNormal")
    in_name = builder.addInput(10)
    enc_name = builder.addInner(3,
                                node_class=NormalTriLNode)
    out_name = builder.addOutput()
    builder.addDirectedLink(in_name, enc_name)
    builder.addDirectedLink(enc_name, out_name)
        
    builder.build()
    inn, enc, out = ( builder.nodes[in_name], builder.nodes[enc_name],
                      builder.nodes[out_name] )
    self.assertEqual(inn._oslot_to_otensor[0].shape.as_list()[-1],
                     enc._islot_to_itensor[0].shape.as_list()[-1], 
                     "The input tensors have not been assigned correctly")
    self.assertEqual(enc._oslot_to_otensor[0].shape.as_list()[-1],
                     out._islot_to_itensor[0].shape.as_list()[-1], 
                     "The input tensors have not been assigned correctly")
    
  @unittest.skipIf(9 not in tests_to_run, "Skipping")
  def test_BuildModel5(self):
    """
    Try to break it, the algorithm... !! Guess not mdrfkr.
    """
    print("\nTest 7: Building a more complicated Model")
    builder = StaticBuilder("BreakIt")
    in1 = builder.addInput(10)
    in2 = builder.addInput(20)
    enc1 = builder.addInner(3,
                            node_class=NormalTriLNode)
    enc2 = builder.addInner(5, num_inputs=2)
    out1 = builder.addOutput()
    out2 = builder.addOutput()
    
    builder.addDirectedLink(in1, enc1)
    builder.addDirectedLink(in2, enc2, islot=0)
    builder.addDirectedLink(enc1, enc2, islot=1)
    builder.addDirectedLink(enc1, out1)
    builder.addDirectedLink(enc2, out2)
    
    builder.build()
    
if __name__ == "__main__":
  unittest.main(failfast=True)