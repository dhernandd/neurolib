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
from neurolib.encoder.normal import NormalPrecisionNode, NormalTriLNode
from neurolib.builders.sequential_builder import SequentialBuilder

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 3
range_from = 0
range_to = 3
tests_to_run = list(range(range_from, range_to))

class NormalNodeTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()

  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_BuildModel1(self):
    """
    Test building the simplest stochastic model possible.
    """
    print("\nTest 1: Building a Model with a Stochastic Node")
    builder = StaticBuilder(scope="BasicNormal")
    i1 = builder.addInput(10)
    in1 = builder.addTransformInner(3,
                                    main_inputs=i1, 
                                    node_class=NormalTriLNode)
        
    builder.build()
    inn, enc = builder.nodes[i1], builder.nodes[in1]
    self.assertEqual(inn._oslot_to_otensor['main'].shape.as_list()[-1],
                     enc._islot_to_itensor[0]['main'].shape.as_list()[-1], 
                     "The input tensors have not been assigned correctly")
    print("enc._oslot_to_otensor", enc._oslot_to_otensor)
    print("enc._islot_to_itensor", enc._islot_to_itensor)
        
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_BuildModel5(self):
    """
    Try to break it, the algorithm... !! Guess not mdrfkr.
    """
    print("\nTest 2: Building a Model with 2 inputs")
    builder = StaticBuilder("BreakIt")
    in1 = builder.addInput(10)
    in2 = builder.addInput(20)
    enc1 = builder.addTransformInner(3,
                                     main_inputs=[in1, in2],
                                     node_class=NormalTriLNode)
    enc2 = builder.addTransformInner(5,
                                     main_inputs=enc1)    
    builder.build()
    
    enc1, enc2 = builder.nodes[enc1], builder.nodes[enc2]
    print("enc1._islot_to_itensor", enc1._islot_to_itensor)
    print("enc1._oslot_to_otensor", enc1._oslot_to_otensor)
    print("enc2._islot_to_itensor", enc2._islot_to_itensor)
    print("enc2._oslot_to_otensor", enc2._oslot_to_otensor)
    
  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_NormalPrecisionNode_init(self):
    """
    Test 3: NormalPrecisionNode build
    """
    print("\nTest 3: NormalPrecisionNode build")
    builder = SequentialBuilder(max_steps=30,
                                scope='Main')
    i1 = builder.addInputSequence([[3]])
    es1 = builder.addInnerSequence2([[3]],
                                    i1,
                                    node_class=NormalPrecisionNode)
    builder.build()
    enc1 = builder.nodes[es1]
    print("enc1._islot_to_itensor", enc1._islot_to_itensor)
    print("enc1._oslot_to_otensor", enc1._oslot_to_otensor)
    
  

if __name__ == '__main__':
  unittest.main(failfast=True)
