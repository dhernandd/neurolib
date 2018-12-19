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

import numpy as np
import tensorflow as tf

from neurolib.utils.utils import addDateTime
from neurolib.trainer.trainer import Trainer
from neurolib.trainer.costs import mse, mabsdiff, elbo,\
  cross_entropy_with_logits
from neurolib.utils.graphs import get_session
from neurolib.trainer.tr_utils import batch_iterator_from_dataset,\
  get_keys_and_data

# pylint: disable=bad-indentation, no-member, protected-access

def make_data_iterator(data, batch_size=1, shuffle=True):
    """
    """
    if batch_size is None:
      batch_size = 1
    nsamps = len(data[0])
    l_inds = np.arange(nsamps)
    if shuffle: 
        np.random.shuffle(l_inds)
    
    for i in range(0, nsamps, batch_size):
        yield [ d[l_inds[i:i+batch_size]] for d in data ]

            
class GDTrainer(Trainer):
  """
  A Trainer that learns parameter by applying simple Gradient Descent to a cost function.
  """
  summaries_dict = {'mse' : mse,
                    'mabsdiff' : mabsdiff,
                    'elbo' : elbo,
                    'cross_entropy_wlogits' : cross_entropy_with_logits}
  opt_dict = {'adam' : tf.train.AdamOptimizer,
              'adagrad' : tf.train.AdagradOptimizer,
              'momentum' : tf.train.MomentumOptimizer,
              'gd' : tf.train.GradientDescentOptimizer}

  def __init__(self,
               builder,
               cost,
               name='test',
               batch_size=1,
               save=False,
               rslt_dir=None,
               **dirs):
    """
    Initialize a GDTrainer
    """
    self.builder = builder
    self.nodes = builder.nodes
    self.batch_size = batch_size
    self.name = name
    self.cost = cost
    self.cost_name = cost[0]
    self.cost_inputs = cost[1]
    self.rslt_dir = rslt_dir
    self._update_default_directives(**dirs)
    
    self._define_train_op()
    
    self.save = save
    if save:
      self.saver = tf.train.Saver(tf.global_variables())
      self.writer = tf.summary.FileWriter(addDateTime('./logs/log'))

  def _update_default_directives(self, **dirs):
    """
    Update the default directives.
    """
    self.directives = {'optimizer' : 'adam',
                       'lr' : 1e-4,
                       'summaries' : []}
    self.directives.update(dirs)
    self.directives['summaries'].append(self.cost)

  def _define_train_op(self):
    """
    Define the train op using tensorflow standard machinery.
    """
    directives = self.directives

    cost_func = self.summaries_dict[self.cost_name]
    self.cost = cost_func(self.nodes, self.cost_inputs)
    
    optimizer_class = self.opt_dict[directives['optimizer']]
    opt = optimizer_class(self.directives['lr'])
    
    self.train_step = tf.get_variable("global_step", [], tf.int64,
                                      tf.zeros_initializer(),
                                      trainable=False)
    self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=tf.get_variable_scope().name)
    print('Scope', tf.get_variable_scope().name)
    for i in range(len(self.train_vars)):
      shape = self.train_vars[i].get_shape().as_list()
      print("    ", i, self.train_vars[i].name, shape)

    gradsvars = opt.compute_gradients(self.cost, self.train_vars)
    self.train_op = opt.apply_gradients(gradsvars, global_step=self.train_step,
                                        name='train_op')
    
  def make_feed_dict(self, dataset):
    """
    Make a feed_dict for sess.run from a dataset.
    
    In particular, add inputs for all the dummy variables defined by the Builder
    associated to this model
    """
    _, data = get_keys_and_data(dataset)
    batch_size = data[0].shape[0]
    for key in self.builder.dummies:
      T = tf.get_default_graph().get_tensor_by_name(key)
      T_sh = T.shape.as_list()[1:]
      dummy_data = np.zeros([batch_size]+T_sh)
      dataset[key] = dummy_data
    return dataset

  def prepare_datasets(self, dataset):
    """
    Splits the dataset dictionary into train, validation and test datasets.
    """
    scope = self.name
    train_dataset = {}
    valid_dataset = {}
    test_dataset = {}
    for key in dataset:
      d_set, inode = key.split('_')[0], key.split('_')[-1]
      if d_set == 'train':
        train_dataset[scope + '/' + inode + ':0'] = dataset[key]
      elif d_set == 'valid':
        valid_dataset[scope + '/' + inode + ':0'] = dataset[key]
      elif d_set == 'test':
        test_dataset[scope + '/' + inode + ':0'] = dataset[key]
      else:
        raise KeyError("The dataset contains the key `{}`. The only allowed "
                       "prefixes for keys in the dataset are 'train', "
                       "'valid' and 'test'".format(key))
    return {'train' : train_dataset,
            'valid' : valid_dataset, 
            'test' : test_dataset}

        
  def update(self, sess, dataset, batch_size):
    """
    Perform a single gradient descent update for the variables in this cost.
    
    TODO: Document!
    TODO: Get rid of the feed_dict in favor of tensorflow Queues! Add
    multithreading capabilities
    """
    dataset_iter = batch_iterator_from_dataset(dataset,
                                               self.batch_size)
    for batch_dct in dataset_iter:
      feed_dict = self.make_feed_dict(batch_dct)
      sess.run([self.train_op], feed_dict=feed_dict)
      
  def train(self, dataset_dict, num_epochs, batch_size=None):
    """
    Train a Model
    """
    train_dataset = dataset_dict['train']
    valid_dataset = dataset_dict['valid']
    
    batch_size = batch_size or self.batch_size or 1
    merged_summaries = self.merge_summaries()
    sess = get_session()
    sess.run(tf.global_variables_initializer())
    for ep in range(num_epochs):
      if ep == 0:
        cvalid = self.reduce_op_from_batches(sess,
                                             [self.cost],
                                             valid_dataset)

      # GD update
      self.update(sess,
                  train_dataset,
                  batch_size=batch_size)
      ctrain = self.reduce_op_from_batches(sess, 
                                           [self.cost],
                                           train_dataset)
      print("cost:", ctrain)
      
      # Add summaries
      if 'summaries' in self.directives and self.save:
        self.run_summaries(sess, train_dataset, merged_summaries, ep)
      
      # Save on validation improvement
      if self.save:
        new_cvalid = self.reduce_op_from_batches(sess,
                                                 [self.cost],
                                                 valid_dataset)
        if new_cvalid < cvalid:
          cvalid = new_cvalid
          print('Valid. cost:', cvalid, '... Saving...')
          rlt_dir = self.rslt_dir + self.name + '/' + addDateTime() + '/'
          if not os.path.exists(rlt_dir):
            os.makedirs(rlt_dir)
          self.saver.save(sess,
                          rlt_dir,
                          global_step=self.train_step)
    sess.close()
  
  def run_summaries(self, sess, dataset, merged_summaries, epoch):
    """
    Run all the defined summaries and write to a log
    """
    fd_dct = self.make_feed_dict(dataset)
    summaries = sess.run(merged_summaries, feed_dict=fd_dct)
    self.writer.add_summary(summaries, epoch)
    
  def reduce_op_from_batches(self,
                             sess,
                             ops,
                             dataset,
                             reduction='mean'):
    """
    Reduce ops from batches
    
    TODO: Document!
    """
    if self.batch_size is None:
      fd_dct = self.make_feed_dict(dataset)
      return sess.run(ops, feed_dict=fd_dct)
    else:
      reduced = 0
      dataset_iter = batch_iterator_from_dataset(dataset,
                                                 self.batch_size,
                                                 shuffle=False)
      if reduction == 'mean' or reduction == 'sum':
        c = 0
        for batch_dct in dataset_iter:
          c += 1
          fd_dct = self.make_feed_dict(batch_dct)
          reduced += sess.run(ops, feed_dict=batch_dct)[0]
        if c == 0:
          raise ValueError("No batches in dataset. Possibly one or more data arrays "
                           "are empty")
        if reduction == 'mean': return reduced/(self.batch_size*c)
        else: return reduced      
        
  def merge_summaries(self):
    """
    Merge summaries
    """
    merged_summaries = []
    if 'summaries' not in self.directives:
      return merged_summaries

    summaries_list = self.directives['summaries']
    for summ in summaries_list:
      name = summ[0]
      curr_summ = tf.summary.scalar(name,
                                    self.summaries_dict[name](self.nodes, summ[1]))
      merged_summaries.append(curr_summ)

    return tf.summary.merge(merged_summaries)
