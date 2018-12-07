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

from neurolib.models.vae import VariationalAutoEncoder 
from neurolib.encoder.normal import NormalTriLNode
from neurolib.builders.static_builder import StaticBuilder

# pylint: disable=bad-indentation, no-member, protected-access

NUM_TESTS = 1
test_to_run = 1
        
def generate_some_data():
  """
  """
  nsamps = 100
  idim = 3
  odim = 10
  x = 1.0*np.random.randn(nsamps, idim)
  W = np.random.randn(3, odim)
  y = np.tanh(np.dot(x, W) + 0.1*np.random.randn(nsamps, odim)) # + 3*x[:,1:]**2 + 0.5*np.random.randn(100,1)
  dataset = {'train_observation' : y[:80],
             'valid_observation' : y[80:]}
  return dataset

class VAETestCustTrain(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()

  @unittest.skipIf(test_to_run == 1, "Skipping") 
  def test_train1(self):
    """
    """
    dataset = generate_some_data()
    
    dirs = {}
    builder = StaticBuilder(scope='vae')
    
    odim = 10
    idim = 3
    hdim = 16
    i1 = builder.addInput(odim, name='observation', **dirs)
    enc0 = builder.addInner(hdim,
                            name='Inner')
    enc1 = builder.addInner(idim,
                            name='Recognition',
                            node_class=NormalTriLNode)
    enc2 = builder.addInner(odim,
                            name='Generative',
                            node_class=NormalTriLNode)
    o1 = builder.addOutput(name='copy')
  
    builder.addDirectedLink(i1, enc0)
    builder.addDirectedLink(enc0, enc1)
    builder.addDirectedLink(enc1, enc2, oslot=0)
    builder.addDirectedLink(enc2, o1, oslot=0)

    vae = VariationalAutoEncoder(builder=builder)
    vae.build()
    vae.train(dataset, num_epochs=20)

  @unittest.skipIf(test_to_run == 2, "Skipping")
  def test_train_custom_node2(self):
    """
    Test commit
    """
    print("Test 2: with CustomNode\n")
    dataset = generate_some_data()
    
    builder = StaticBuilder("MyModel")
    enc_dirs = {'num_layers' : 2,
                'num_nodes' : 64,
                'activation' : 'leaky_relu',
                'net_grow_rate' : 1.0 }
    
    input_dim = 2
    in0 = builder.addInput(input_dim, name='observation')
    
    cust_rec = builder.createCustomNode(1, 1, name="Recognition")
    cust_rinn1 = cust_rec.addInner(3,
                                   node_class=NormalTriLNode,
                                   directives=enc_dirs)
#     cust_rec.addDirectedLink(cust_rinn1, cust_rinn2)
    cust_rec.declareIslot(islot=0, innernode_name=cust_rinn1, inode_islot=0)
    cust_rec.declareOslot(oslot=0, innernode_name=cust_rinn1, inode_oslot=0)
    cust_rec.commit()
    
    cust_gen = builder.createCustomNode(1, 1, name="Generative")
    cust_ginn1 = cust_gen.addInner(16, directives=enc_dirs)
    cust_ginn2 = cust_gen.addInner(10,
                                   node_class=NormalTriLNode,
                                   directives=enc_dirs)
    cust_gen.addDirectedLink(cust_ginn1, cust_ginn2)
    cust_gen.declareIslot(islot=0, innernode_name=cust_ginn1, inode_islot=0)
    cust_gen.declareOslot(oslot=0, innernode_name=cust_ginn2, inode_oslot=0)
    cust_gen.commit()
    
    out0 = builder.addOutput(name='prediction')
    
    builder.addDirectedLink(in0, cust_rec)
    builder.addDirectedLink(cust_rec, cust_gen)
    builder.addDirectedLink(cust_gen, out0)

    vae = VariationalAutoEncoder(builder=builder)
    vae.build()
    vae.train(dataset, num_epochs=20)

  

if __name__ == '__main__':
  unittest.main(failfast=True)