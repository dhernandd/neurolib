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
from neurolib.encoder.seq_cells import BasicEncoderCell, NormalTriLCell

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 2
range_from = 1
range_to = 2
tests_to_run = list(range(range_from, range_to))

class SequentialModelBuilderTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    print()
    tf.reset_default_graph()
    
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_DeclareModel1(self):
    """
    """
    print("Test 0:")
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    in1 = builder.addInputSequence(10)
    prior = builder.addInput([[3]],
                             NormalInputNode,
                             name='Prior')
    rnn_name = builder.addRNN(main_inputs=in1,
                              state_inputs=prior,
                              cell_class=BasicEncoderCell)
    builder.build()
    
    rnn = builder.nodes[rnn_name]
    print("rnn._islot_to_itensor", rnn._islot_to_itensor)
    print("rnn._oslot_to_otensor", rnn._oslot_to_otensor)
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_NormalTriLRNN(self):
    """
    """
    print("Test 0:")
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    in1 = builder.addInputSequence(10)
    prior = builder.addInput([[3]],
                             NormalInputNode,
                             name='Prior')
    rnn_name = builder.addRNN(main_inputs=in1,
                              state_inputs=prior,
                              cell_class=NormalTriLCell)
    builder.build()
    
    rnn = builder.nodes[rnn_name]
    print("rnn._islot_to_itensor", rnn._islot_to_itensor)
    print("rnn._oslot_to_otensor", rnn._oslot_to_otensor)
    
if __name__ == "__main__":
  unittest.main(failfast=True)
