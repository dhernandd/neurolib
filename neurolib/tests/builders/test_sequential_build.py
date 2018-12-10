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

# pylint: disable=bad-indentation, no-member, protected-access

NUM_TESTS = 9
run_up_to_test = 7
tests_to_run = list(range(run_up_to_test))

class SequentialModelBuilderTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()
    
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_DeclareModel0(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    builder.addInputSequence(10)

  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_DeclareModel1(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    builder.addInputSequence(10)
    builder.addEvolutionSequence(3, 2)

  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_DeclareModel2(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    builder.addInputSequence(10)
    builder.addEvolutionSequence(3, 2)
    builder.addOutputSequence()

  @unittest.skipIf(3 not in tests_to_run, "Skipping")
  def test_DeclareModel3(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    in1 = builder.addInputSequence(10)
    enc1 = builder.addEvolutionSequence(3, 2)
    o1 = builder.addOutputSequence()
    
    print("in1, enc1", in1, enc1)
    builder.addDirectedLink(in1, enc1, islot=1)
    builder.addDirectedLink(enc1, o1)

  @unittest.skipIf(4 not in tests_to_run, "Skipping")
  def test_BuildModel1(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    in1 = builder.addInputSequence(10)
    enc1 = builder.addEvolutionSequence(3, 2)
    o1 = builder.addOutputSequence()
    
    builder.addDirectedLink(in1, enc1, islot=1)
    builder.addDirectedLink(enc1, o1)
    
    builder.build()

  @unittest.skipIf(5 not in tests_to_run, "Skipping")
  def test_BuildModel2(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    in1 = builder.addInputSequence(10)
    enc1 = builder.addEvolutionSequence(3, 2)
    enc2 = builder.addEvolutionSequence(4, 2)
    o1 = builder.addOutputSequence()
    
    builder.addDirectedLink(in1, enc1, islot=1)
    builder.addDirectedLink(enc1, enc2, islot=1)
    builder.addDirectedLink(enc2, o1)
    
    builder.build()
    
  @unittest.skipIf(6 not in tests_to_run, "Skipping")
  def test_BuildModel3(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    in1 = builder.addInputSequence(10)
    enc1 = builder.addEvolutionSequence([[3],[3]], 3,
                                        ev_seq_class='lstm', cell_class='lstm')
    enc2 = builder.addInnerSequence(4, num_inputs=2)
    o1 = builder.addOutputSequence()
    
    builder.addDirectedLink(in1, enc1, islot=2)
    builder.addDirectedLink(in1, enc2)
    builder.addDirectedLink(enc1, enc2, islot=1)
    builder.addDirectedLink(enc2, o1)
    
    builder.build()
    
    e2 = builder.nodes[enc2]
    self.assertEqual(e2._oslot_to_otensor[0].shape.as_list()[-1],
                     4, "Error")
    print("e2._islot_to_itensor", e2._islot_to_itensor)

    
if __name__ == "__main__":
  unittest.main(failfast=True)