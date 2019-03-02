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

from neurolib.encoder.stochasticevseqs import LDSEvolution, LLDSEvolution
from neurolib.builders.sequential_builder import SequentialBuilder
from neurolib.encoder.input import NormalInputNode

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 6
range_from = 2
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
  def test_LDSEvolution(self):
    """
    Test LDSNode build
    """
    builder = SequentialBuilder(scope='Main',
                                max_steps=2)
    i0 = builder.addInput([[3]], NormalInputNode, name='Prior')
    i1 = builder.addInputSequence([[3]], name='InSeq')
    in1 = builder.addEvolutionwPriors([[3]],
                                      main_inputs=i1,
                                      prior_inputs=i0,
                                      node_class=LDSEvolution,
                                      name='LDS')
    
    builder.build()

    in1, i1 = builder.nodes[in1], builder.nodes[i1]
    print("{}._oslot_to_otensor:".format(i1.name), i1._oslot_to_otensor)
    print("{}._oslot_to_otensor:".format(in1.name), in1._oslot_to_otensor)
    print("{}._islot_to_itensor:".format(in1.name), in1._islot_to_itensor)
    
    
    Z = tf.placeholder(tf.float64, [None, 2, 3])
    in1.build_loc(imain0=Z)
    lp = in1.logprob(Z)
    print("logprob,", lp)
 
    sess = tf.Session(graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
     
    z = np.array([[[1.0, 2.0, 3.0],
                   [1.0, 1.0, 1.0]]])
    opt = builder.eval_node_oslot(sess, 'LDS',
                                  feed_dict={'Main/InSeq_main:0' : z})
    print("output", opt)
  
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_LLDSEvolution_build(self):
    """
    Test LLDSNode initialization
    """
    builder = SequentialBuilder(scope='Main',
                                max_steps=2)
    i0 = builder.addInput([[3]], NormalInputNode, name='Prior')
    i1 = builder.addInputSequence([[3]], name='InSeq')
    in1 = builder.addEvolutionwPriors([[3]],
                                      main_inputs=i1,
                                      prior_inputs=i0,
                                      node_class=LLDSEvolution,
                                      name='LLDS')
    
    builder.build()
        
    in1, i1 = builder.nodes[in1], builder.nodes[i1]
    print("{}._oslot_to_otensor:".format(i1.name), i1._oslot_to_otensor)
    print("{}._oslot_to_otensor:".format(in1.name), in1._oslot_to_otensor)
    
    sess = tf.Session(graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
     
    z = np.array([[[1.0, 2.0, 3.0],
                  [1.0, 0.0, 0.0]]])
    opt = builder.eval_node_oslot(sess,
                                  'LLDS',
                                  oslot='A',
                                  feed_dict={'Main/InSeq_main:0' : z})
    print("output", opt)

  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_LLDSNode_build2(self):
    """
    Test LLDSNode initialization
    """
    builder = SequentialBuilder(max_steps=2,
                                scope='Main')
    i1 = builder.addInput([[3]],
                          NormalInputNode,
                          name='In1')
    is2 = builder.addInputSequence([[3]], name='In2')
    is3 = builder.addInputSequence([[4]], name='In3')
    in1 = builder.addEvolutionwPriors([[3]],
                                      main_inputs=is2,
                                      prior_inputs=i1,
                                      sec_inputs=is3,
                                      node_class=LLDSEvolution,
                                      name='LLDS')
    
    builder.build()
    
    in1, i1 = builder.nodes[in1], builder.nodes[i1]
    print("{}._oslot_to_otensor:".format(i1.name), i1._oslot_to_otensor)
    print("{}._oslot_to_otensor:".format(in1.name), in1._oslot_to_otensor)
    


if __name__ == '__main__':
  unittest.main(failfast=True)
