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

from neurolib.models.regression import Regression
from neurolib.builders.static_builder import StaticBuilder

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 3
range_from = 0
range_to = 3
tests_to_run = list(range(range_from, range_to))

with open(path + '/datadict_regression', 'rb') as f1:
  dataset = pickle.load(f1)

class RegressionTestTrainCust(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    print()
    tf.reset_default_graph()
  
  @unittest.skipIf(0 not in tests_to_run, "Skipping Test 0")
  def test_train_custom_builder(self):
    """
    """
    print("Test 0: Chain of Encoders\n")
    
    # Define a builder. Get more control on the directives of each node
    enc_dirs = {'numlayers' : 2,
                'numnodes' : 128,
                'activations' : 'leaky_relu',
                'netgrowrate' : 1.0 }
    input_dim, output_dim = 2, 1

    builder = StaticBuilder('CustBuild')
    in0 = builder.addInput(input_dim, name="Features")
    enc1 = builder.addTransformInner(5,
                                     main_inputs=in0,
                                     **enc_dirs)
    builder.addTransformInner(state_size=1,
                              main_inputs=enc1,
                              directives=enc_dirs,
                              name='Prediction')
    
    # Pass the builder object to the Regression Model
    reg = Regression(builder=builder,
                     output_dim=output_dim)
#                      save_on_valid_improvement=True) # ok!
    
    reg.train(dataset,
              num_epochs=10)

  @unittest.skipIf(1 not in tests_to_run, "Skipping Test 1")
  def test_train_custom_builder2(self):
    """
    Build a custom Regression model whose Model graph has the rhombic design
    """
    print("Test 1: Rhombic Design\n")
        
    # Define the builder
    builder = StaticBuilder('CustBuild')
    enc_dirs = {'numlayers' : 2,
                'numnodes' : 128,
                'activations' : 'leaky_relu',
                'netgrowrate' : 1.0 }
    input_dim = 2
    in0 = builder.addInput(input_dim, name="Features")
    enc1 = builder.addTransformInner(10,
                                     main_inputs=in0,
                                     directives=enc_dirs)
    enc2 = builder.addTransformInner(10,
                                     main_inputs=in0,
                                     directives=enc_dirs)
    builder.addTransformInner(1,
                              main_inputs=[enc1, enc2],
                              numlayers=1,
                              activations='linear',
                              name='Prediction')

    reg = Regression(input_dim=2,
                     output_dim=1,
                     builder=builder)
#                      save_on_valid_improvement=True)  # OK!
    reg.train(dataset, num_epochs=10)
    
  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_train_custom_node2(self):
    """
    Test a custom Regression model with a CustomNode
    """
    print("Test 2: with CustomNode\n")
    
    builder = StaticBuilder("MyModel")
    enc_dirs = {'numlayers' : 2,
                'numnodes' : 128,
                'activations' : 'leaky_relu',
                'netgrowrate' : 1.0 }
    
    input_dim = 2
    in0 = builder.addInput(input_dim, name='Features')
    
    cust = builder.createCustomNode(inputs=[in0],
                                    num_outputs=1,
                                    name="Prediction")
    cust_in1 = cust.addTransformInner(3,
                                      main_inputs=[0],
                                      directives=enc_dirs)
    cust_in2 = cust.addTransformInner(1,
                                      main_inputs=cust_in1,
                                      directives=enc_dirs)
    cust.declareOslot(oslot=0, innernode_name=cust_in2, inode_oslot_name='main')
    
    
    reg = Regression(builder=builder,
                     input_dim=2,
                     output_dim=1)
#                      save_on_valid_improvement=True) # ok!
    reg.train(dataset, num_epochs=10)

    
if __name__ == '__main__':
  unittest.main(failfast=True)
