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
from neurolib.encoder.normal import NormalPrecisionNode
from neurolib.encoder.seq_cells import LDSCell
from neurolib.encoder.merge import MergeSeqsNormalLDSEv

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 2
run_from = 0
run_to = 2
tests_to_run = list(range(run_from, run_to))

class MergeEvSeqsFLDSTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()
  
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_merge_init(self):
    """
    Test Merge Node initialization
    """
    builder = SequentialBuilder(max_steps=30,
                                scope='Main')
    ins1 = builder.addInnerSequence([[3]],
                                    num_inputs=1,
                                    node_class=NormalPrecisionNode)
    evs1 = builder.addEvolutionSequence([[3]],
                                        num_inputs=1,
                                        cell_class=LDSCell)
    builder.addMergeNode([[3]],
                         node_dict={'in_seq' : ins1, 'ev_seq' : evs1},
                         merge_class=MergeSeqsNormalLDSEv)

  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_merge_build(self):
    """
    Test Merge Node Build
    """
    builder = SequentialBuilder(max_steps=30,
                                scope='Main')
    is1 = builder.addInputSequence([[3]])
    ins1 = builder.addInnerSequence([[3]], num_inputs=1, node_class=NormalPrecisionNode)
    evs1 = builder.addEvolutionSequence([[3]],
                                        num_inputs=1,
                                        cell_class=LDSCell)
    m1 = builder.addMergeNode([[3]],
                              node_dict={'in_seq' : ins1, 'ev_seq' : evs1},
                              merge_class=MergeSeqsNormalLDSEv)
    ins2 = builder.addInnerSequence([[3]], num_inputs=1, node_class=NormalPrecisionNode)
    os1 = builder.addOutputSequence()
    builder.addDirectedLink(is1, ins1)
    builder.addDirectedLink(m1, ins2)
    builder.addDirectedLink(ins2, os1)
    
    builder.build()
    ent = builder.nodes[m1].entropy()
    print(ent)


if __name__ == "__main__":
  unittest.main(failfast=True)
