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

from neurolib.builders.sequential_builder import SequentialBuilder
from neurolib.encoder.input import NormalInputNode

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 3
range_from = 0
range_to = 3
tests_to_run = list(range(range_from, range_to))

class DefaultSequentialModelBuilderTest(tf.test.TestCase):
  """
  Tests the building of Sequential models using tf native cells
  """
  def setUp(self):
    """
    """
    print()
    tf.reset_default_graph()
    
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_DeclareModel3(self):
    """
    """
    print("Test 0:")
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    in1 = builder.addInputSequence(10)
    prior = builder.addInput([[3]], NormalInputNode, name='Prior')
    rnn_name = builder.addRNN(main_inputs=in1,
                              state_inputs=prior)
    builder.build()
    
    rnn = builder.nodes[rnn_name]
    rnn_cell = rnn.cell
    X = tf.placeholder(tf.float64, [None, 3])
    Y = tf.placeholder(tf.float64, [None, 10])
    Z = rnn_cell(X, Y)
    print("Z", Z)
    
        
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_BuildModel2(self):
    """
    """
    print("Test 1:")
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    in1 = builder.addInputSequence(10)
    prior1 = builder.addInput([[3]], NormalInputNode, name='Prior1')
    prior2 = builder.addInput([[3]], NormalInputNode, name='Prior2')
    enc1 = builder.addRNN(main_inputs=in1,
                          state_inputs=prior1)
    enc2 = builder.addRNN(main_inputs=enc1,
                          state_inputs=prior2)
    
    builder.build()
    
    enc1, enc2 = builder.nodes[enc1], builder.nodes[enc2]
    print("enc1._islot_to_itensor", enc1._islot_to_itensor)
    print("enc1._oslot_to_otensor", enc1._oslot_to_otensor)
    print("enc2._islot_to_itensor", enc2._islot_to_itensor)
    print("enc2._oslot_to_otensor", enc2._oslot_to_otensor)

    
  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_BuildModel3(self):
    """
    """
    print("Test 2:")
    builder = SequentialBuilder(max_steps=10,
                                scope="Basic")
    
    in1 = builder.addInputSequence(6)
    prior1 = builder.addInput([[3]], NormalInputNode, name='Prior1')
    prior2 = builder.addInput([[3]], NormalInputNode, name='Prior2')
    enc1 = builder.addRNN(main_inputs=in1,
                          state_inputs=[prior1, prior2],
                          cell_class='lstm')
    enc2 = builder.addInnerSequence([[4]],
                                    main_inputs=enc1)
        
    builder.build()
    
    e1 = builder.nodes[enc1]
    e2 = builder.nodes[enc2]
    print("e1._islot_to_itensor", e1._islot_to_itensor)
    print("e1._oslot_to_otensor", e1._oslot_to_otensor)
    print("e2._islot_to_itensor", e2._islot_to_itensor)
    print("e2._oslot_to_otensor", e2._oslot_to_otensor)

         
if __name__ == "__main__":
  unittest.main(failfast=True)