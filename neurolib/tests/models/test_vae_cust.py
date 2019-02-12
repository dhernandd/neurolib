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
range_from = 0
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
    tf.reset_default_graph()

  @unittest.skipIf(0 not in tests_to_run, "Skipping") 
  def test_train1(self):
    """
    """
    dirs = {}
    builder = StaticBuilder(scope='vae')
    
    odim = 10
    idim = 3
    hdim = 16
    i1 = builder.addInput(odim, name='Observation', **dirs)
    enc0 = builder.addInner(hdim,
                            name='Inner')
    enc1 = builder.addInner(idim,
                            name='Recognition',
                            node_class=NormalTriLNode)
    enc2 = builder.addInner(odim,
                            name='Generative',
                            node_class=NormalTriLNode)
  
    builder.addDirectedLink(i1, enc0, islot=0)
    builder.addDirectedLink(enc0, enc1, islot=0)
    builder.addDirectedLink(enc1, enc2, islot=0)

    vae = VariationalAutoEncoder(builder=builder)
#                                  save_on_valid_improvement=True) # OK!
    vae.train(dataset, num_epochs=10)

  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_train_custom_node2(self):
    """
    Test commit
    """
    print("Test 2: with CustomNode\n")
    
    builder = StaticBuilder("MyModel")
    enc_dirs = {'loc_numlayers' : 2,
                'loc_numnodes' : 64,
                'loc_activations' : 'leaky_relu',
                'loc_netgrowrate' : 1.0 }
    
    input_dim = 10
    in0 = builder.addInput(input_dim, name='Observation')
    
    # Define Custom Recognition Model
    cust_rec = builder.createCustomNode(1, 1, name="Recognition")
    cust_rinn1 = cust_rec.addInner(3,
                                   node_class=NormalTriLNode,
                                   **enc_dirs)
    cust_rec.declareIslot(islot=0, innernode_name=cust_rinn1, inode_islot=0)
    cust_rec.declareOslot(oslot='main', innernode_name=cust_rinn1, inode_oslot='main')
    
    # Define Custom Generative Model
    cust_gen = builder.createCustomNode(1, 1, name="Generative")
    cust_ginn1 = cust_gen.addInner(16)
    cust_ginn2 = cust_gen.addInner(10,
                                   node_class=NormalTriLNode,
                                   **enc_dirs)
    cust_gen.addDirectedLink(cust_ginn1, cust_ginn2, islot=0)
    cust_gen.declareIslot(islot=0, innernode_name=cust_ginn1, inode_islot=0)
    cust_gen.declareOslot(oslot='main', innernode_name=cust_ginn2, inode_oslot='main')
    
    # Link all of it
    builder.addDirectedLink(in0, cust_rec, islot=0)
    builder.addDirectedLink(cust_rec, cust_gen, islot=0)

    # Define VAE and train
    vae = VariationalAutoEncoder(builder=builder)
    vae.train(dataset, num_epochs=10)

  

if __name__ == '__main__':
  unittest.main(failfast=True)
