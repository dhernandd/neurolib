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
import os.path

my_path = os.path.abspath(os.path.dirname(__file__))

import unittest
import pickle

import tensorflow as tf

from neurolib.models.dkf import DeepKalmanFilter

# pylint: disable=bad-indentation, no-member, protected-access

# NUM_TESTS : 2
range_from = 0
range_to = 1
tests_to_run = list(range(range_from, range_to))

class DKFTestTrain(tf.test.TestCase):
  """
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()
  
  @unittest.skipIf(0 not in tests_to_run, "Skipping")
  def test_train(self):
    """
    """
    print("\nTest 1: DKF build")

    dataset = {}
    fname = my_path + '/datadict_gaussianobs2D'
    with open(fname, 'rb') as f:
      datadict = pickle.load(f)
      Ytrain = datadict['Ytrain']
      Yshape = Ytrain.shape
      dataset['train_observation_0'] = Ytrain
      dataset['valid_observation_0'] = datadict['Yvalid']
      max_steps, input_dims = Yshape[-2], Yshape[-1]
      
    dkf = DeepKalmanFilter(input_dims=[[input_dims]],
                           max_steps=max_steps,
                           state_dims=[[40], [4]])
    dkf.build()
    dkf.train(dataset, num_epochs=10)
    

if __name__ == '__main__':
  unittest.main(failfast=True)

#   fname = 'datadict_gaussianobs2D'
#   dataset = {}
#   with open(fname, 'rb') as f:
#     datadict = pickle.load(f)
#     Ytrain = datadict['Ytrain']
#     Yshape = Ytrain.shape
#     print("Yshape", Yshape)
#     dataset['train_observation_0'] = Ytrain
#     dataset['valid_observation_0'] = datadict['Yvalid']
#     max_steps, input_dims = Yshape[-2], Yshape[-1]
# 
# 
#   max_steps = 30
#   xdim = 20
#   X = tf.placeholder(tf.float64, [None, max_steps, xdim], 'X')
#   h1 = fully_connected(X, 32, activation_fn=tf.nn.relu)
#   h2 = fully_connected(h1, 32, activation_fn=tf.nn.relu)
#   out = fully_connected(h2, 4, activation_fn=None)
#   out = tf.reshape(out, [-1, 2, 2])
#   out = tf.matrix_band_part(out, -1, 0)
#   covs = tf.matmul(out, out, transpose_a=True)
#   cost = tf.reduce_sum(tf.log(tf.matrix_determinant(covs)))
#   
#   opt = tf.train.AdamOptimizer()
#   train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#   gradvars = opt.compute_gradients(cost, train_vars)
#   train_op = opt.apply_gradients(gradvars)
#   
#   sess = get_session()
#   sess.run(tf.global_variables_initializer())
#   c = sess.run(cost, feed_dict={'X:0' : Ytrain})
#   print("cost", c)
#   for _ in range(5):
#     sess.run(train_op, feed_dict={'X:0' : Ytrain})
#     c = sess.run(cost, feed_dict={'X:0' : Ytrain})
#     print("cost", c)
  
  