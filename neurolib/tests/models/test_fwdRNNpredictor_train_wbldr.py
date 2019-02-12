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

from neurolib.models.predictor_rnn import PredictorRNN
from neurolib.builders.sequential_builder import SequentialBuilder

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS: 4
range_from = 0
range_to = 4
tests_to_run = list(range(range_from, range_to))

with open(path + '/datadict_seq01', 'rb') as f1:
  dataset = pickle.load(f1)
with open(path + '/datadict_seq01cat', 'rb') as f1:
  dataset_cat = pickle.load(f1)


class RNNPredictorTrainTest(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()
  
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_train0(self):
    """
    Test train RNNPredictor with custom build : continuous data
    """
    max_steps = dataset['train_Observation'].shape[1]
    
    # Define a MG with 2 RNNs
    input_dims = [[1]]
    state_dims_1 = [[5]]
    state_dims_2 = [[10]]
    output_dims = [[1]]
    ninputs_evseq_1 = ninputs_evseq_2 = 2
    
    scope = 'RNNPredictor'
    builder = SequentialBuilder(max_steps=max_steps, scope=scope)
    
    is1 = builder.addInputSequence(input_dims, name='Features')
    evs1 = builder.addEvolutionSequence(state_sizes=state_dims_1,
                                        num_inputs=ninputs_evseq_1,
                                        ev_seq_class='rnn',
                                        cell_class='basic',
                                        name='RNN1')
    evs2 = builder.addEvolutionSequence(state_sizes=state_dims_2,
                                        num_inputs=ninputs_evseq_2,
                                        ev_seq_class='rnn',
                                        cell_class='basic',
                                        name='RNN2')
    inn1 = builder.addInnerSequence(output_dims, name='Prediction')
          
    builder.addDirectedLink(is1, evs1, islot=1)
    builder.addDirectedLink(evs1, evs2, islot=1)
    builder.addDirectedLink(evs2, inn1, islot=0)
    
    # Pass the builder to the PredictorRNN model
    model = PredictorRNN(builder=builder)
    model.train(dataset, num_epochs=10)
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_train1(self):
    """
    Test train RNNPredictor with custom build : categorical data
    """
    num_labels = 4
    max_steps = dataset['train_Observation'].shape[1]
    
    # Define a MG with 2 RNNs ('forward' : ['lstm', 'basic'])
    input_dims = [[1]]
    state_dims_1 = [[5], [5]]
    state_dims_2 = [[10]]
    ninputs_evseq_1, ninputs_evseq_2 = 3, 2

    scope = 'RNNPredictor'
    builder = SequentialBuilder(max_steps=max_steps, scope=scope)
    
    is1 = builder.addInputSequence(input_dims, name='Features')
    evs1 = builder.addEvolutionSequence(state_sizes=state_dims_1,
                                        num_inputs=ninputs_evseq_1,
                                        ev_seq_class='rnn',
                                        cell_class='lstm',
                                        name='RNN1')
    evs2 = builder.addEvolutionSequence(state_sizes=state_dims_2,
                                        num_inputs=ninputs_evseq_2,
                                        ev_seq_class='rnn',
                                        cell_class='basic',
                                        name='RNN2')
    inn1 = builder.addInnerSequence(num_labels,
                                    name='Prediction')
          
    builder.addDirectedLink(is1, evs1, islot=2)
    builder.addDirectedLink(evs1, evs2, islot=1)
    builder.addDirectedLink(evs2, inn1, islot=0)
    
    # Pass the builder to the PredictorRNN model
    model = PredictorRNN(builder=builder,
                         num_labels=num_labels,
                         is_categorical=True)
    model.train(dataset_cat, num_epochs=10)
    
  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_train2(self):
    """
    Test train RNNPredictor with custom build (rhombic design): categorical data
    """
    num_labels = 4
    max_steps = dataset['train_Observation'].shape[1]
    
    # Define a MG with 2 RNNs ('forward' : ['lstm', 'basic'])
    input_dims = [[1]]
    state_dims_1 = [[5], [5]]
    state_dims_2 = [[10]]
    ninputs_evseq_1, ninputs_evseq_2 = 3, 2

    scope = 'RNNPredictor'
    builder = SequentialBuilder(max_steps=max_steps, scope=scope)
    
    is1 = builder.addInputSequence(input_dims, name='Features')
    evs1 = builder.addEvolutionSequence(state_sizes=state_dims_1,
                                        num_inputs=ninputs_evseq_1,
                                        ev_seq_class='rnn',
                                        cell_class='lstm',
                                        name='RNN1')
    evs2 = builder.addEvolutionSequence(state_sizes=state_dims_2,
                                        num_inputs=ninputs_evseq_2,
                                        ev_seq_class='rnn',
                                        cell_class='basic',
                                        name='RNN2')
    inn1 = builder.addInnerSequence(num_labels, num_inputs=2,
                                    name='Prediction')
          
    builder.addDirectedLink(is1, evs1, islot=2)
    builder.addDirectedLink(is1, evs2, islot=1)
    builder.addDirectedLink(evs1, inn1, islot=0)
    builder.addDirectedLink(evs2, inn1, islot=1)
    
    # Pass the builder to the PredictorRNN model
    model = PredictorRNN(builder=builder,
                         num_labels=num_labels,
                         is_categorical=True)
    model.train(dataset_cat, num_epochs=10)
    
  @unittest.skipIf(3 not in tests_to_run, "Skipping")
  def test_train3(self):
    """
    Test train RNNPredictor with custom build (rhombic design): categorical data
    """
    num_labels = 4
    max_steps = dataset['train_Observation'].shape[1]
    
    # Define a MG with 2 RNNs ('forward' : ['lstm', 'basic'])
    input_dims = [[1]]
    state_dims_1 = [[5], [5]]
    state_dims_2 = [[10]]
    ninputs_evseq_1, ninputs_evseq_2 = 3, 3

    scope = 'RNNPredictor'
    builder = SequentialBuilder(max_steps=max_steps, scope=scope)

    is1 = builder.addInputSequence(input_dims, name='Features')
    evs1 = builder.addEvolutionSequence(state_sizes=state_dims_1,
                                        num_inputs=ninputs_evseq_1,
                                        ev_seq_class='rnn',
                                        cell_class='lstm',
                                        name='RNN1')
    evs2 = builder.addEvolutionSequence(state_sizes=state_dims_2,
                                        num_inputs=ninputs_evseq_2,
                                        ev_seq_class='rnn',
                                        cell_class='basic',
                                        name='RNN2')
    inn1 = builder.addInnerSequence(num_labels,
                                    num_inputs=1,
                                    name='Prediction')
          
    builder.addDirectedLink(is1, evs1, islot=2)
    builder.addDirectedLink(evs1, evs2, islot=1)
    builder.addDirectedLink(is1, evs2, islot=2)
    builder.addDirectedLink(evs2, inn1, islot=0)
    
    # Pass the builder to the PredictorRNN model
    model = PredictorRNN(builder=builder,
                         num_labels=num_labels,
                         is_categorical=True)
#                          save_on_valid_improvement=True) # ok!
    model.train(dataset_cat, num_epochs=10)

if __name__ == '__main__':
  unittest.main(failfast=True)
