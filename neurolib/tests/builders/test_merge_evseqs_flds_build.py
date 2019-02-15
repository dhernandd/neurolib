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
from neurolib.encoder.normal import NormalPrecisionNode, LDSNode
from neurolib.encoder.merge import MergeSeqsNormalwNormalEv
from neurolib.encoder.input import NormalInputNode

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 2
run_from = 0
run_to = 2
tests_to_run = list(range(run_from, run_to))

class MergeSeqsNormalwNormalEvTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()
  
  def test_mergeNormalEvSeqs(self):
    """
    """
    ydim = 10
    xdim = 2
    
    builder = SequentialBuilder(scope='Main',max_steps=25) 
    is1 = builder.addInputSequence([[ydim]], name='Observation')
    is2 = builder.addInputSequence([[xdim]], name='State')
    n1 = builder.addInput([[xdim]],
                          NormalInputNode,
                          name='Prior')
    ins1 = builder.addInnerSequence([[xdim]],
                                    num_inputs=1,
                                    node_class=NormalPrecisionNode,
                                    name='InnSeq')
    evs1 = builder.addInnerSequence([[xdim]],
                                    num_inputs=1,
                                    node_class=LDSNode,
                                    name='LDS')
    m1 = builder.addMergeNode(node_list=[ins1, evs1, n1],
                              merge_class=MergeSeqsNormalwNormalEv,
                              name='Recognition')
    ins2 = builder.addInnerSequence([[ydim]],
                                    num_inputs=1,
                                    node_class=NormalPrecisionNode,
                                    name='Generative')
    builder.addDirectedLink(is1, ins1, islot=0)
    builder.addDirectedLink(is2, evs1, islot=0)
    builder.addDirectedLink(m1, ins2, islot=0)

  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_merge_build(self):
    """
    Test Merge Node Build
    """
    ydim = 10
    xdim = 2
    
    builder = SequentialBuilder(scope='Main',max_steps=25) 
    is1 = builder.addInputSequence([[ydim]], name='Observation')
    is2 = builder.addInputSequence([[xdim]], name='State')
    n1 = builder.addInput([[xdim]],
                          NormalInputNode,
                          name='Prior')
    ins1 = builder.addInnerSequence([[xdim]],
                                    num_inputs=1,
                                    node_class=NormalPrecisionNode,
                                    name='InnSeq')
    evs1 = builder.addInnerSequence([[xdim]],
                                    num_inputs=1,
                                    node_class=LDSNode,
                                    name='LDS')
    m1 = builder.addMergeNode(node_list=[ins1, evs1, n1],
                              merge_class=MergeSeqsNormalwNormalEv,
                              name='Recognition')
    ins2 = builder.addInnerSequence([[ydim]],
                                    num_inputs=1,
                                    node_class=NormalPrecisionNode,
                                    name='Generative')
    builder.addDirectedLink(is1, ins1, islot=0)
    builder.addDirectedLink(is2, evs1, islot=0)
    builder.addDirectedLink(m1, ins2, islot=0)      
    
    builder.build()


if __name__ == "__main__":
  unittest.main(failfast=True)
