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

from neurolib.builders.static_builder import StaticBuilder
from neurolib.trainer.gd_trainer import GDTrainer

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 3
range_from = 0
range_to = 3
tests_to_run = list(range(range_from, range_to))

def prepare_datasets(dataset, scope):
  """
  Split the dataset dictionary into train, validation and test datasets.
  """
  train_dataset = {}
  valid_dataset = {}
  test_dataset = {}
  for key in dataset:
    key_split = key.split('_')
    d_set, inode_name = key_split[0], '_'.join(key_split[1:])
    if d_set == 'train':
      train_dataset[scope + '/' + inode_name + ':0'] = dataset[key]
    elif d_set == 'valid':
      valid_dataset[scope + '/' + inode_name + ':0'] = dataset[key]
    elif d_set == 'test':
      test_dataset[scope + '/' + inode_name + ':0'] = dataset[key]
    else:
      raise KeyError("The dataset contains the key `{}`. The only allowed "
                     "prefixes for keys in the dataset are 'train', "
                     "'valid' and 'test'".format(key))
  
  return {'train' : train_dataset,
          'valid' : valid_dataset, 
          'test' : test_dataset}
  
def generate_some_data():
  """
  """
  x = 10.0*np.random.randn(100, 2)
  y = x[:,0:1] + 1.5*x[:,1:]# + 3*x[:,1:]**2 + 0.5*np.random.randn(100,1)
  xtrain, xvalid, ytrain, yvalid = x[:80], x[80:], y[:80], y[80:]
  dataset = {'train_observation_in' : xtrain,
             'train_observation_out' : ytrain,
             'valid_observation_in' : xvalid,
             'valid_observation_out' : yvalid}
  return dataset


class GDTrainerTest(tf.test.TestCase):
  """
  """
  batch_size = 1
  def setUp(self):
    """
    """
    tf.reset_default_graph()

  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_init(self):
    """
    """
    print("Test 0: Trainer initialization")
    dirs = {'num_layers' : 2}
    in_dim = 2
    out_dim = 1
    builder = StaticBuilder(scope="regression",
                            batch_size=self.batch_size)
    in0 = builder.addInput(in_dim, name="observation_in")
    enc1 = builder.addInner(out_dim, num_inputs=1, **dirs)
    out0 = builder.addOutput(name="prediction")

    builder.addDirectedLink(in0, enc1)
    builder.addDirectedLink(enc1, out0)

    in1 = builder.addInput(out_dim, name="observation_out")
    out1 = builder.addOutput(name="response")
    builder.addDirectedLink(in1, out1)

    # Build the tensorflow graph
    builder.build()

    GDTrainer(builder,
              cost=('mse', ('prediction','response')),
              **dirs)
    
  @unittest.skipIf(1 not in tests_to_run, "Skipping")
  def test_train(self):
    """
    """
    print("\nTest 1: Trainer train")

    dataset = generate_some_data()
    scope = 'regression'
    dataset_dict = prepare_datasets(dataset, scope)

    dirs = {'num_layers' : 2}
    in_dim = 2
    out_dim = 1
    num_epochs = 10
    builder = StaticBuilder(scope=scope)
    in0 = builder.addInput(in_dim, name="observation_in")
    enc1 = builder.addInner(out_dim, num_inputs=1, **dirs)
    out0 = builder.addOutput(name="prediction")
    builder.addDirectedLink(in0, enc1)
    builder.addDirectedLink(enc1, out0)

    in1 = builder.addInput(out_dim, name="observation_out")
    out1 = builder.addOutput(name="response")
    builder.addDirectedLink(in1, out1)

    builder.build()

    trainer = GDTrainer(builder,
                        cost=('mse', ('prediction', 'response')),
                        **dirs)
    trainer.train(dataset_dict, num_epochs, batch_size=self.batch_size)

  @unittest.skipIf(2 not in tests_to_run, "Skipping")
  def test_save(self):
    """
    Test saving and summaries
    
    Skip at push
    """
    print("\nTest 2: Trainer saving capabilities")

    dataset = generate_some_data()
    scope = 'regression'
    dataset_dict = prepare_datasets(dataset, scope)

    dirs = {'num_layers' : 2,
            'summaries' : [('mabsdiff', ('prediction', 'response'))]}
    in_dim = 2
    out_dim = 1
    num_epochs = 10
    builder = StaticBuilder(scope=scope)
    in0 = builder.addInput(in_dim, name="observation_in")
    enc1 = builder.addInner(out_dim, num_inputs=1, **dirs)
    out0 = builder.addOutput(name="prediction")
    builder.addDirectedLink(in0, enc1)
    builder.addDirectedLink(enc1, out0)

    in1 = builder.addInput(out_dim, name="observation_out")
    out1 = builder.addOutput(name="response")
    builder.addDirectedLink(in1, out1)

    builder.build()

    trainer = GDTrainer(builder,
                        cost=('mse', ('prediction', 'response')),
#                         save=True,
                        **dirs)
    trainer.train(dataset_dict, num_epochs, batch_size=self.batch_size)


if __name__ == "__main__":
  unittest.main(failfast=True)