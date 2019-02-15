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
from neurolib.encoder.seq_cells import LDSCell, LLDSCell

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 6
range_from = 0
range_to = 7
tests_to_run = list(range(range_from, range_to))

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
    builder.addEvolutionSequence(3, num_inputs=2)

  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_DeclareModel3(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    in1 = builder.addInputSequence(10)
    enc1 = builder.addEvolutionSequence(3, num_inputs=2)
    
    print("in1, enc1", in1, enc1)
    builder.addDirectedLink(in1, enc1, islot=1)

  @unittest.skipIf(3 not in tests_to_run, "Skipping")
  def test_BuildModel1(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    in1 = builder.addInputSequence(10)
    enc1 = builder.addEvolutionSequence(3, num_inputs=2)
    
    builder.addDirectedLink(in1, enc1, islot=1)
    builder.build()
    
    enc1 = builder.nodes[enc1]
    print("enc1._islot_to_itensor", enc1._islot_to_itensor)
    print("enc1._oslot_to_otensor", enc1._oslot_to_otensor)
    
  @unittest.skipIf(4 not in tests_to_run, "Skipping")
  def test_BuildModel2(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    in1 = builder.addInputSequence(10)
    enc1 = builder.addEvolutionSequence(3, num_inputs=2)
    enc2 = builder.addEvolutionSequence(4, num_inputs=2)
    
    builder.addDirectedLink(in1, enc1, islot=1)
    builder.addDirectedLink(enc1, enc2, islot=1)
    builder.build()
    
    enc1, enc2 = builder.nodes[enc1], builder.nodes[enc2]
    print("enc1._islot_to_itensor", enc1._islot_to_itensor)
    print("enc1._oslot_to_otensor", enc1._oslot_to_otensor)
    print("enc2._islot_to_itensor", enc2._islot_to_itensor)
    print("enc3._oslot_to_otensor", enc2._oslot_to_otensor)

    
  @unittest.skipIf(5 not in tests_to_run, "Skipping")
  def test_BuildModel3(self):
    """
    """
    builder = SequentialBuilder(max_steps=10,
                                scope="Basic")
    
    in1 = builder.addInputSequence(6)
    enc1 = builder.addEvolutionSequence([[3],[3]], 
                                        num_inputs=3,
                                        cell_class='lstm')
    enc2 = builder.addInnerSequence(4, num_inputs=2)
    
    builder.addDirectedLink(in1, enc1, islot=2)
    builder.addDirectedLink(in1, enc2, islot=0)
    builder.addDirectedLink(enc1, enc2, islot=1)
    builder.build()
    
    e2 = builder.nodes[enc2]
    print("e2._islot_to_itensor", e2._islot_to_itensor)
    print("e2._oslot_to_otensor", e2._oslot_to_otensor)

     
  @unittest.skipIf(6 not in tests_to_run, "Skipping")
  def test_lds_cell_build(self):
    """
    """
    scope = "Main"
    max_steps = 25
 
    builder = SequentialBuilder(max_steps=max_steps,
                                scope=scope)
    ev1 = builder.addEvolutionSequence([[3]],
                                       num_inputs=1,
                                       cell_class=LDSCell)
    builder.build()
    
    ev1 = builder.nodes[ev1]
    print("ev1._islot_to_itensor", ev1._islot_to_itensor)
    print("ev1._oslot_to_otensor", ev1._oslot_to_otensor)
     
  @unittest.skipIf(7 not in tests_to_run, "Skipping")
  def test_llds_cell_build(self):
    """
    """
    scope = "Main"
    max_steps = 25
 
    builder = SequentialBuilder(max_steps=max_steps,
                                scope=scope)
    ev1 = builder.addEvolutionSequence([[3]],
                                       num_inputs=1,
                                       cell_class=LLDSCell)
    builder.build()
    
    ev1 = builder.nodes[ev1]
    print("ev1._islot_to_itensor", ev1._islot_to_itensor)
    print("ev1._oslot_to_otensor", ev1._oslot_to_otensor)
     
    
if __name__ == "__main__":
  unittest.main(failfast=True)