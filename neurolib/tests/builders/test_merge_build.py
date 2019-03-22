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
from neurolib.encoder.merge import MergeNormals
from neurolib.encoder.input import NormalInputNode

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 3
run_from = 0
run_to = 0
tests_to_run = list(range(run_from, run_to))

class MergeNormalsTest(tf.test.TestCase):
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
    builder = StaticBuilder(scope='Main')
    in1 = builder.addInner([[3]], num_inputs=1, node_class=NormalTriLNode)
    in2 = builder.addInner([[3]], num_inputs=1, node_class=NormalTriLNode)
    builder.addMergeNode(node_list=[in1, in2],
                         merge_class=MergeNormals)
  
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_merge_build1(self):
    """
    Test Merge Node build
    """
    builder = StaticBuilder(scope='Main')
    i1 = builder.addInput([[1]])
    in1 = builder.addInner([[3]], node_class=NormalTriLNode)
    in2 = builder.addInner([[3]], node_class=NormalTriLNode)
    builder.addDirectedLink(i1, in1, islot=0)
    builder.addDirectedLink(i1, in2, islot=0)
    m1 = builder.addMergeNode([in1, in2],
                              merge_class=MergeNormals)
    builder.build()
    m1 = builder.nodes[m1]
    self.assertIn('loc', m1._oslot_to_otensor)
    self.assertIn('main', m1._oslot_to_otensor)
    self.assertIn('cov', m1._oslot_to_otensor)

  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_merge_build2(self):
    """
    Test Merge Node build
    """
    builder = StaticBuilder(scope='Main')
    i1 = builder.addInput([[3]], iclass=NormalInputNode, name='N1')
    i2 = builder.addInput([[3]], iclass=NormalInputNode, name='N2')
    m1 = builder.addMergeNode(node_list=[i1, i2],
                              merge_class=MergeNormals)
    builder.build()
    
    sess = tf.Session(graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    
    s3 = builder.eval_node_oslot(sess, m1,
                                 oslot='cov')
    print("merge output", s3)
    print("merge output shape", s3.shape)
    

if __name__ == "__main__":
  unittest.main(failfast=False)
