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

# pylint: disable=bad-indentation, no-member, protected-access

from neurolib.builders.static_builder import StaticBuilder
from neurolib.encoder.normal import NormalTriLNode

# NUM_TESTS : 1
run_from = 0
run_to = 1
tests_to_run = list(range(run_from, run_to))


class CustomEncoderNormalTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()

  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_add_encoder(self):
    """
    Add an Encoder Node to the Custom Encoder
    """
    builder = StaticBuilder("MyModel")

    cust = builder.createCustomNode(num_inputs=1,
                                    num_outputs=1,
                                    name="Custom")
    cust_in0 = cust.addInner(2)
    cust_in1 = cust.addInner(3, node_class=NormalTriLNode)
    cust.addDirectedLink(cust_in0, cust_in1, 0)
    cust.declareIslot(islot=0, innernode_name=cust_in0, inode_islot=0)
    cust.declareOslot(oslot='main', innernode_name=cust_in1, inode_oslot='scale')
    
    i1 = builder.addInput(10)
    builder.addDirectedLink(i1, cust, islot=0)     
    builder.build()
    print(cust._oslot_to_otensor)
    

if __name__ == "__main__":
  unittest.main(failfast=True)
