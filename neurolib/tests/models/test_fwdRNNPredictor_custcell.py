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

from neurolib.encoder.seq_cells import TwoEncodersCell, NormalTriLCell,\
  TwoEncodersCell2, LDSCell, BasicEncoderCell
from neurolib.builders.sequential_builder import SequentialBuilder
from neurolib.models.predictor_rnn import PredictorRNN

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 9
range_from = 0
range_to = 9
tests_to_run = list(range(range_from, range_to))

with open(path + '/datadict_seq01', 'rb') as f1:
  dataset = pickle.load(f1)
with open(path + '/datadict_seq01cat', 'rb') as f1:
  dataset_cat = pickle.load(f1)
  
  
class CustomCellTrainTest(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()

  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_build_custom0(self):
    """
    Test build Custom Cell
    """
    builder = SequentialBuilder(max_steps=25,
                                scope="Main")
    is1 = builder.addInputSequence([[1]], name='Features')
    ev1 = builder.addEvolutionSequence([[10]],
                                       cell_class=BasicEncoderCell,
                                       num_inputs=2,
                                       name='RNN')
    inn2 = builder.addInnerSequence(state_sizes=[[1]],
                                    name='Prediction')    
    builder.addDirectedLink(is1, ev1, islot=1)
    builder.addDirectedLink(ev1, inn2, islot=0)
    
    PredictorRNN(builder=builder)
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_train_custom0(self):
    """
    Test build Custom Cell
    """
    builder = SequentialBuilder(max_steps=25,
                                scope="Main")
    is1 = builder.addInputSequence([[1]], name='Features')
    ev1 = builder.addEvolutionSequence([[10]],
                                       cell_class=BasicEncoderCell,
                                       num_inputs=2,
                                       name='RNN')
    inn2 = builder.addInnerSequence(state_sizes=[[1]],
                                    name='Prediction')    
    builder.addDirectedLink(is1, ev1, islot=1)
    builder.addDirectedLink(ev1, inn2, islot=0)
    
    rnn = PredictorRNN(builder=builder)
    rnn.train(dataset, num_epochs=10)
    
  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_build_custom1(self):
    """
    Test build Custom Cell
    """
    builder = SequentialBuilder(max_steps=25,
                                scope="Main")
    is1 = builder.addInputSequence([[1]], name='Features')
    ev1 = builder.addEvolutionSequence([[2],[3]],
                                       num_inputs=3,
                                       cell_class=TwoEncodersCell,
                                       name='RNN')
    inn2 = builder.addInnerSequence(state_sizes=[[1]],
                                    name='Prediction')    
    builder.addDirectedLink(is1, ev1, islot=2)
    builder.addDirectedLink(ev1, inn2, islot=0)
    
    PredictorRNN(builder=builder)
    print("builder.nodes[ev1]._oslot_to_otensor", builder.nodes[ev1]._oslot_to_otensor)

  @unittest.skipIf(3 not in tests_to_run, "Skipping")
  def test_train_customz(self):
    """
    Test train Custom cell.
    
    This test tests a configuration that exemplifies neurolib's capabilities. A
    CustomCell is defined consisting of two forward RNNs working in tandem. The
    second RNN also takes as inputs, the outputs of the first one (See
    TwoEncodersCell). The outputs of these two RNNS are passed to an MSE cost
    function.
    """
    builder = SequentialBuilder(max_steps=25,
                                scope="Main")
    is1 = builder.addInputSequence([[1]], name='Features')
    ev1 = builder.addEvolutionSequence([[2],[3]],
                                       num_inputs=3,
                                       cell_class=TwoEncodersCell,
                                       name='RNN')
    inn2 = builder.addInnerSequence(state_sizes=[[1]],
                                    name='Prediction')    
    builder.addDirectedLink(is1, ev1, islot=2)
    builder.addDirectedLink(ev1, inn2, islot=0)
    
    rnn = PredictorRNN(builder=builder)
    rnn.train(dataset, num_epochs=10)

  @unittest.skipIf(4 not in tests_to_run, "Skipping")
  def test_build_custom2(self):
    """
    Test train Custom cell with forward links.
     
    This test tests a configuration that exemplifies neurolib's capabilities. A
    CustomCell is defined consisting of two forward RNNs working in tandem. The
    second RNN also takes as inputs, the outputs of the first one (See
    TwoEncodersCell). The outputs of these two RNNS are passed to an MSE cost
    function.
    """
    builder = SequentialBuilder(max_steps=25,
                                scope="Main")     
    is1 = builder.addInputSequence([[1]], name='Features')
    ev1 = builder.addEvolutionSequence([[2],[3]],
                                       num_inputs=3,
                                       cell_class=TwoEncodersCell2,
                                       name='RNN')
    inn2 = builder.addInnerSequence(state_sizes=[[1]],
                                    name='Prediction')    
    builder.addDirectedLink(is1, ev1, islot=2)
    builder.addDirectedLink(ev1, inn2, islot=0)
    
    PredictorRNN(builder=builder)

  @unittest.skipIf(5 not in tests_to_run, "Skipping")
  def test_train_custom2(self):
    """
    Test train Custom cell with forward links.
     
    This test tests a configuration that exemplifies neurolib's capabilities. A
    CustomCell is defined consisting of two forward RNNs working in tandem. The
    second RNN also takes as inputs, the outputs of the first one (See
    TwoEncodersCell). The outputs of these two RNNS are passed to an MSE cost
    function.
    """
    builder = SequentialBuilder(max_steps=25,
                                scope="Main")     
    is1 = builder.addInputSequence([[1]], name='Features')
    ev1 = builder.addEvolutionSequence([[2],[3]],
                                       num_inputs=3,
                                       cell_class=TwoEncodersCell2,
                                       name='RNN')
    inn2 = builder.addInnerSequence(state_sizes=[[1]],
                                    name='Prediction')
    builder.addDirectedLink(is1, ev1, islot=2)
    builder.addDirectedLink(ev1, inn2, islot=0)
    
    rnn = PredictorRNN(builder=builder)
    rnn.train(dataset, num_epochs=10)

  @unittest.skipIf(6 not in tests_to_run, "Skipping")
  def test_build_custom3(self):
    """
    Test train Custom NormalTriLCell
     
    This test tests a configuration that exemplifies neurolib's capabilities. A
    CustomCell is defined that consists of a forward stochastic RNN.
    """
    builder = SequentialBuilder(max_steps=25,
                                scope="Main")     
    is1 = builder.addInputSequence([[1]], name='Features')
    ev1 = builder.addEvolutionSequence([[3]],
                                       num_inputs=2,
                                       cell_class=NormalTriLCell,
                                       name='RNN')
    inn1 = builder.addInnerSequence(state_sizes=[[1]],
                                    name='Prediction')
    builder.addDirectedLink(is1, ev1, islot=1)
    builder.addDirectedLink(ev1, inn1, islot=0)
    
    PredictorRNN(builder=builder)

  @unittest.skipIf(7 not in tests_to_run, "Skipping")
  def test_train_custom3(self):
    """
    Test train Custom NormalTriLCell
     
    This test tests a configuration that exemplifies neurolib's capabilities. A
    CustomCell is defined that consists of a forward stochastic RNN.
    """
    builder = SequentialBuilder(max_steps=25,
                                scope="Main")     
    is1 = builder.addInputSequence([[1]], name='Features')
    ev1 = builder.addEvolutionSequence([[3]],
                                       num_inputs=2,
                                       cell_class=NormalTriLCell,
                                       name='RNN')
    inn1 = builder.addInnerSequence(state_sizes=[[1]],
                                    name='Prediction')
    builder.addDirectedLink(is1, ev1, islot=1)
    builder.addDirectedLink(ev1, inn1, islot=0)
    
    rnn = PredictorRNN(builder=builder)
    rnn.train(dataset, num_epochs=20)

     
  @unittest.skipIf(8 not in tests_to_run, "Skipping")
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
     
    o = builder.get_node_output(ev1, oslot='A')
    print(o)
    o = builder.eval_node_oslot(ev1, oslot='A')
    print(o[0,0])

    
if __name__ == '__main__':
  unittest.main(failfast=True)
  