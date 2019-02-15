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

import numpy as np
import tensorflow as tf

from neurolib.builders.static_builder import StaticBuilder
from neurolib.encoder.normal import LDSNode, NormalPrecisionNode, LLDSNode, NormalTriLNode
from neurolib.builders.sequential_builder import SequentialBuilder

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 6
range_from = 0
range_to = 6
tests_to_run = list(range(range_from, range_to))

class NormalNodeTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()

  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_BuildModel4(self):
    """
    Test building the simplest stochastic model possible.
    """
    print("\nTest 8: Building a Model with a Stochastic Node")
    builder = StaticBuilder(scope="BasicNormal")
    in_name = builder.addInput(10)
    enc_name = builder.addInner(3, node_class=NormalTriLNode)
    builder.addDirectedLink(in_name, enc_name, islot=0)
        
    builder.build()
    inn, enc = builder.nodes[in_name], builder.nodes[enc_name]
    self.assertEqual(inn._oslot_to_otensor['main'].shape.as_list()[-1],
                     enc._islot_to_itensor[0]['main'].shape.as_list()[-1], 
                     "The input tensors have not been assigned correctly")
    print("enc._oslot_to_otensor", enc._oslot_to_otensor)
    print("enc._islot_to_itensor", enc._islot_to_itensor)
    
    Z = tf.placeholder(tf.float32, [None, 10, 3])
    loc = enc.build_loc(Z)
    sc = enc.build_scale_tril(Z)
    print("loc, Z", loc)
    print("sc, Z", sc)
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
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
    
    builder.addDirectedLink(in1, enc1, islot=0)
    builder.addDirectedLink(in2, enc2, islot=0)
    builder.addDirectedLink(enc1, enc2, islot=1)
    
    builder.build()
    enc1, enc2 = builder.nodes[enc1], builder.nodes[enc2]
    print("enc1._oslot_to_otensor", enc1._oslot_to_otensor)
    print("enc1._islot_to_itensor", enc1._islot_to_itensor)
    print("enc2._oslot_to_otensor", enc2._oslot_to_otensor)
    print("enc2._islot_to_itensor", enc2._islot_to_itensor)
    
  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_LDSNode(self):
    """
    Test LDSNode initialization
    """
    builder = StaticBuilder(scope='Main')
    i1 = builder.addInput([[3]], name='In')
    in1 = builder.addInner([[3]],
                           node_class=LDSNode,
                           name='LDS')
    builder.addDirectedLink(i1, in1, islot=0)
    
    builder.build()
    in1, i1 = builder.nodes[in1], builder.nodes[i1]
    print("{}._oslot_to_otensor:".format(i1.name), i1._oslot_to_otensor)
    print("{}._oslot_to_otensor:".format(in1.name), in1._oslot_to_otensor)
    
    Z = tf.placeholder(tf.float64, [None, 10, 3])
    A = in1.get_output_tensor('A')
    loc = in1.build_loc(Z, A)
    print("loc, Z", loc)

    sess = tf.Session(graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    
    z = np.array([[1.0, 2.0, 3.0]])
    opt = builder.eval_node_oslot(sess, 'LDS',
                                  feed_dict={'Main/In_main:0' : z})
    print("output", opt)
  
  @unittest.skipIf(3 not in tests_to_run, "Skipping")
  def test_LDSNode_seq(self):
    """
    Test LDSNode initialization
    """
    builder = SequentialBuilder(scope='Main',
                                max_steps=30)
    is1 = builder.addInputSequence([[3]])
#     ins1 = builder.addInnerSequence([[3]],
#                                     node_class=LDSNode)
    ins1 = builder.addInner([[3]],
                                node_class=LDSNode)
    
    builder.addDirectedLink(is1, ins1, islot=0)
    builder.build()
    ins1 = builder.nodes[ins1]
    print("{}._oslot_to_otensor:".format(ins1.name), ins1._oslot_to_otensor)

  @unittest.skipIf(4 not in tests_to_run, "Skipping")
  def test_NormalPrecisionNode_init(self):
    """
    Test NormalPrecisionNode initialization
    """
    builder = SequentialBuilder(max_steps=30,
                                scope='Main')
    is1 = builder.addInputSequence([[3]])
    ins2 = builder.addInnerSequence([[3]],
                                    num_inputs=1,
                                    node_class=NormalPrecisionNode)
    builder.addDirectedLink(is1, ins2, islot=0)
    builder.build()
  
  @unittest.skipIf(5 not in tests_to_run, "Skipping")
  def test_LLDSNode_init(self):
    """
    Test LLDSNode initialization
    """
    builder = SequentialBuilder(max_steps=30,
                                scope='Builder')
    i1 = builder.addInput([[3]], name='In')
    in1 = builder.addInner([[3]],
                           node_class=LLDSNode,
                           name='LLDS')
    builder.addDirectedLink(i1, in1, islot=0)
    
    builder.build()
    in1, i1 = builder.nodes[in1], builder.nodes[i1]
    print("{}._oslot_to_otensor:".format(i1.name), i1._oslot_to_otensor)
    print("{}._oslot_to_otensor:".format(in1.name), in1._oslot_to_otensor)
    
    sess = tf.Session(graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    
    z = np.array([[1.0, 2.0, 3.0],
                  [1.0, 0.0, 0.0]])
    opt = builder.eval_node_oslot(sess,
                                  'LLDS',
                                  oslot='A',
                                  feed_dict={'Builder/In_main:0' : z})
    print("output", opt)


if __name__ == '__main__':
  unittest.main(failfast=True)
