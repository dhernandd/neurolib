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

from neurolib.models.predictor_rnn import PredictorRNN
from neurolib.builders.sequential_builder import SequentialBuilder

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS: 3
range_from = 0
range_to = 4
tests_to_run = list(range(range_from, range_to))
test_to_run = 10

def generate_echo_data(length, echo_step, max_steps):
  """
  Generate some echo data
  """
  x = np.array(np.random.normal(size=length))
  y = np.roll(x, echo_step)
  y[0:echo_step] = 0
  
  X = np.reshape(x, [-1, max_steps, 1])
  Y = np.reshape(y, [-1, max_steps, 1])

  dataset = {'train_inputSeq' : X[:300],
             'train_outputSeq' : Y[:300],
             'valid_inputSeq' : X[300:],
             'valid_outputSeq' : Y[300:]}

  return dataset

def generate_echo_data_cat(num_labels, length, echo_step, max_steps):
  """
  Generate some echo categorical data
  """
  p = num_labels*[1/num_labels]
  x = np.array(np.random.choice(num_labels, length, p=p))
  y = np.roll(x, echo_step)
  y[0:echo_step] = 0

  X = np.reshape(x, [-1, max_steps, 1])
  Y = np.reshape(y, [-1, max_steps, 1])

  dataset = {'train_inputSeq' : X[:300],
             'train_outputSeq' : Y[:300],
             'valid_inputSeq' : X[300:],
             'valid_outputSeq' : Y[300:]}

  return dataset


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
    max_steps = 25
    echo_step = 3
    dataset = generate_echo_data(10000, echo_step, max_steps)
    
    
    # Define a MG with 2 RNNs
    input_dims = [[1]]
    state_dims_1 = [[5]]
    state_dims_2 = [[10]]
    output_dims = [[1]]
    ninputs_evseq_1 = ninputs_evseq_2 = 2
    scope = 'RNNPredictor'
    builder = SequentialBuilder(max_steps=25, scope=scope)
    is1 = builder.addInputSequence(input_dims, name='inputSeq')
    evs1 = builder.addEvolutionSequence(state_sizes=state_dims_1,
                                        num_inputs=ninputs_evseq_1,
                                        ev_seq_class='rnn',
                                        cell_class='basic',
                                        name='ev_seq1')
    evs2 = builder.addEvolutionSequence(state_sizes=state_dims_2,
                                        num_inputs=ninputs_evseq_2,
                                        ev_seq_class='rnn',
                                        cell_class='basic',
                                        name='ev_seq2')
    inn1 = builder.addInnerSequence(output_dims)
    os1 = builder.addOutputSequence(name='prediction')
          
    builder.addDirectedLink(is1, evs1, islot=1)
    builder.addDirectedLink(evs1, evs2, islot=1)
    builder.addDirectedLink(evs2, inn1)
    builder.addDirectedLink(inn1, os1)      
    
    # Pass the builder to the PredictorRNN model
    model = PredictorRNN(builder=builder)
    model.build()
    model.train(dataset, num_epochs=10)
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_train1(self):
    """
    Test train RNNPredictor with custom build : categorical data
    """
    num_labels = 4
    max_steps = 25
    echo_step = 3
    dataset = generate_echo_data_cat(num_labels, 10000, echo_step, max_steps)
    scope = 'RNNPredictor'
    
    # Define a MG with 2 RNNs ('forward' : ['lstm', 'basic'])
    input_dims = [[1]]
    state_dims_1 = [[5], [5]]
    state_dims_2 = [[10]]
    ninputs_evseq_1, ninputs_evseq_2 = 3, 2
    builder = SequentialBuilder(max_steps=25, scope=scope)
    is1 = builder.addInputSequence(input_dims, name='inputSeq')
    evs1 = builder.addEvolutionSequence(state_sizes=state_dims_1,
                                        num_inputs=ninputs_evseq_1,
                                        ev_seq_class='rnn',
                                        cell_class='lstm',
                                        name='ev_seq1')
    evs2 = builder.addEvolutionSequence(state_sizes=state_dims_2,
                                        num_inputs=ninputs_evseq_2,
                                        ev_seq_class='rnn',
                                        cell_class='basic',
                                        name='ev_seq2')
    inn1 = builder.addInnerSequence(num_labels)
    os1 = builder.addOutputSequence(name='prediction')
          
    builder.addDirectedLink(is1, evs1, islot=2)
    builder.addDirectedLink(evs1, evs2, islot=1, oslot=0)
    builder.addDirectedLink(evs2, inn1)
    builder.addDirectedLink(inn1, os1)      
    
    # Pass the builder to the PredictorRNN model
    model = PredictorRNN(builder=builder,
                         num_labels=num_labels,
                         is_categorical=True)
    model.build()
    model.train(dataset, num_epochs=10)
    
  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_train2(self):
    """
    Test train RNNPredictor with custom build (rhombic design): categorical data
    """
    num_labels = 4
    max_steps = 25
    echo_step = 3
    dataset = generate_echo_data_cat(num_labels, 10000, echo_step, max_steps)
    scope = 'RNNPredictor'
    
    # Define a MG with 2 RNNs ('forward' : ['lstm', 'basic'])
    input_dims = [[1]]
    state_dims_1 = [[5], [5]]
    state_dims_2 = [[10]]
    ninputs_evseq_1, ninputs_evseq_2 = 3, 2
    builder = SequentialBuilder(max_steps=25, scope=scope)
    is1 = builder.addInputSequence(input_dims, name='inputSeq')
    evs1 = builder.addEvolutionSequence(state_sizes=state_dims_1,
                                        num_inputs=ninputs_evseq_1,
                                        ev_seq_class='rnn',
                                        cell_class='lstm',
                                        name='ev_seq1')
    evs2 = builder.addEvolutionSequence(state_sizes=state_dims_2,
                                        num_inputs=ninputs_evseq_2,
                                        ev_seq_class='rnn',
                                        cell_class='basic',
                                        name='ev_seq2')
    inn1 = builder.addInnerSequence(num_labels, num_inputs=2)
    os1 = builder.addOutputSequence(name='prediction')
          
    builder.addDirectedLink(is1, evs1, islot=2)
    builder.addDirectedLink(is1, evs2, islot=1)
    builder.addDirectedLink(evs1, inn1)
    builder.addDirectedLink(evs2, inn1, islot=1)
    builder.addDirectedLink(inn1, os1)      
    
    # Pass the builder to the PredictorRNN model
    model = PredictorRNN(builder=builder,
                         num_labels=num_labels,
                         is_categorical=True)
    model.build()
    model.train(dataset, num_epochs=10)
    
  @unittest.skipIf(3 not in tests_to_run, "Skipping")
  def test_train3(self):
    """
    Test train RNNPredictor with custom build (rhombic design): categorical data
    """
    num_labels = 4
    max_steps = 25
    echo_step = 3
    dataset = generate_echo_data_cat(num_labels, 10000, echo_step, max_steps)
    scope = 'RNNPredictor'
    
    # Define a MG with 2 RNNs ('forward' : ['lstm', 'basic'])
    input_dims = [[1]]
    state_dims_1 = [[5], [5]]
    state_dims_2 = [[10]]
    ninputs_evseq_1, ninputs_evseq_2 = 3, 3
    builder = SequentialBuilder(max_steps=25, scope=scope)
    is1 = builder.addInputSequence(input_dims, name='inputSeq')
    evs1 = builder.addEvolutionSequence(state_sizes=state_dims_1,
                                        num_inputs=ninputs_evseq_1,
                                        ev_seq_class='rnn',
                                        cell_class='lstm',
                                        name='ev_seq1')
    evs2 = builder.addEvolutionSequence(state_sizes=state_dims_2,
                                        num_inputs=ninputs_evseq_2,
                                        ev_seq_class='rnn',
                                        cell_class='basic',
                                        name='ev_seq2')
    inn1 = builder.addInnerSequence(num_labels, num_inputs=2)
    os1 = builder.addOutputSequence(name='prediction')
          
    builder.addDirectedLink(is1, evs1, islot=2)
    builder.addDirectedLink(evs1, evs2, islot=1)
    builder.addDirectedLink(is1, evs2, islot=2)
    builder.addDirectedLink(evs2, inn1, islot=1)
    builder.addDirectedLink(inn1, os1)      
    
    # Pass the builder to the PredictorRNN model
    model = PredictorRNN(builder=builder,
                         num_labels=num_labels,
                         is_categorical=True)
    model.build()
    model.train(dataset, num_epochs=10)
if __name__ == '__main__':
  unittest.main(failfast=True)
