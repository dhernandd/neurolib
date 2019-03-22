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
import os
path = os.path.dirname(os.path.realpath(__file__))
import unittest
import pickle

import tensorflow as tf

from neurolib.models.vae import VariationalAutoEncoder 
from neurolib.encoder.normal import NormalTriLNode
from neurolib.builders.static_builder import StaticBuilder

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 2
range_from = 1
range_to = 2
tests_to_run = list(range(range_from, range_to))
        
with open(path + '/datadict_vae', 'rb') as f1:
  dataset = pickle.load(f1)

class VAETestCustTrain(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """
  def setUp(self):
    """
    """
    print()
    tf.reset_default_graph()

  @unittest.skipIf(0 not in tests_to_run, "Skipping") 
  def test_train1(self):
    """
    """
    print("Test 0:")
    dirs = {}
    builder = StaticBuilder(scope='vae')
    
    odim = 10
    idim = 3
    hdim = 16
    i1 = builder.addInput(odim, name='Observation', **dirs)
    enc0 = builder.addTransformInner(hdim,
                                     main_inputs=i1,
                                     name='Inner')
    enc1 = builder.addTransformInner(idim,
                                     main_inputs=enc0,
                                     name='Recognition',
                                     node_class=NormalTriLNode)
    builder.addTransformInner(odim,
                              main_inputs=enc1,
                              name='Generative',
                              node_class=NormalTriLNode)
  
    vae = VariationalAutoEncoder(builder=builder)
#                                  save_on_valid_improvement=True) # OK!
    vae.train(dataset, num_epochs=20)

  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_train_custom_node2(self):
    """
    Test commit
    """
    print("Test 1: with CustomNode\n")
    
    builder = StaticBuilder("MyModel")
    enc_dirs = {'loc_numlayers' : 2,
                'loc_numnodes' : 64,
                'loc_activations' : 'softplus',
                'loc_netgrowrate' : 1.0 }
    
    input_dim = 10
    in0 = builder.addInput(input_dim, name='Observation')
    
    # Define Custom Recognition Model
    cust_rec = builder.createCustomNode(inputs=[in0], 
                                        num_outputs=1,
                                        name="Recognition")
    cust_in2 = cust_rec.addTransformInner(3,
                                          main_inputs=[0],
                                          node_class=NormalTriLNode,
                                          **enc_dirs)
    cust_rec.declareOslot(oslot=0, innernode_name=cust_in2, inode_oslot_name='main')
    
    # Define Custom Generative Model
    cust_gen = builder.createCustomNode(inputs=cust_rec.name,
                                        num_outputs=1,
                                        name="Generative")
    cust_ginn1 = cust_gen.addTransformInner(16,
                                            main_inputs=[0])
    cust_ginn2 = cust_gen.addTransformInner(10,
                                            main_inputs=cust_ginn1,
                                            node_class=NormalTriLNode,
                                            **enc_dirs)
    cust_gen.declareOslot(oslot=0, innernode_name=cust_ginn2, inode_oslot_name='main')
    
    # Define VAE and train
    vae = VariationalAutoEncoder(builder=builder)
    
    vae.train(dataset, num_epochs=10)

  

if __name__ == '__main__':
  unittest.main(failfast=True)
