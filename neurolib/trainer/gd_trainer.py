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
import time

import numpy as np
import tensorflow as tf

from neurolib.utils.utils import addDateTime
from neurolib.trainer.trainer import Trainer
from neurolib.trainer.costs import (mse, mabsdiff, elbo, 
                                    cross_entropy_with_logits,
                                    entropy, logprob)
from neurolib.utils.dataset_manip import (get_dataset_batch_size,
                                          batch_iterator_from_dataset)

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
                    'cross_entropy_wlogits' : cross_entropy_with_logits,
                    'entropy' : entropy,
                    'logprob' : logprob}
  opt_dict = {'adam' : tf.train.AdamOptimizer,
              'adagrad' : tf.train.AdagradOptimizer,
              'momentum' : tf.train.MomentumOptimizer,
              'gd' : tf.train.GradientDescentOptimizer}

  def __init__(self,
               model=None,
               mode='new',
               restore_dir=None,
               **dirs):
    """
    Initialize a GDTrainer
    
    Args:
    """
    super(GDTrainer, self).__init__()
    
    self.model = model
    self.sess = model.sess
    self.save = model.save
    self.mode = mode
    self.batch_size = model.batch_size
    self.scope = model.main_scope

    # Define train ops before savers
    if mode == 'new':
      self.nodes = model.nodes
      self.dummies = self.model.builder.dummies
      if self.save:
        root_rslts_dir = model.root_rslts_dir
        self.rslts_dir = root_rslts_dir + self.scope + '/' + addDateTime() + '/'
        if not os.path.exists(self.rslts_dir):
          os.makedirs(self.rslts_dir)

      self._update_default_directives(**dirs)
      self._define_train_op()

    elif mode == 'restore':
      if self.save:
        assert restore_dir is not None, (
          "Argument `restore_dir` is required if " 
          "`save_on_valid_improvememt` is set to True")
      self.rslts_dir = restore_dir
      self.dummies = self.model.dummies
      
      self._update_default_directives(**dirs)
    
    # Saver capabilities after defining train_op
    if self.save:
      self.saver = tf.train.Saver(tf.global_variables())
    self.keep_logs = model.keep_logs
    if self.keep_logs:
      self.writer = tf.summary.FileWriter(addDateTime('./logs/log'))
    
  def _update_default_directives(self, **dirs):
    """
    Update the default directives.
    """
    this_trainer_dirs = {'optimizer' : 'adam',
                         'lr' : 1e-3,
                         'wlr_schedule' : False}
    if self.mode == 'new':
      cost_name = self.model.otensor_names['cost']
      cost = tf.get_default_graph().get_tensor_by_name(cost_name)
      this_trainer_dirs['summaries'] = {'cost' : cost}
    this_trainer_dirs.update(dirs)
    super(GDTrainer, self)._update_directives(**this_trainer_dirs)

  def _define_train_op(self):
    """
    Define the train op using tensorflow standard machinery.
    """
    directives = self.directives
    cost = self.model.cost
    optimizer_class = self.opt_dict[directives['optimizer']]
    
    # TODO: Put inside model scope??
    init_lr = self.directives['lr']
    lr = tf.get_variable('lr', dtype=tf.float32, initializer=init_lr)
    opt = optimizer_class(lr)
    
    self.train_step = tf.get_variable("global_step", [], tf.int64,
                                      tf.zeros_initializer(),
                                      trainable=False)
    self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=tf.get_variable_scope().name)
    print('\nScope:', self.scope)
    for i in range(len(self.train_vars)):
      shape = self.train_vars[i].get_shape().as_list()
      print("    ", i, self.train_vars[i].name, shape)

    gradsvars = opt.compute_gradients(cost, self.train_vars)
    self.train_op = opt.apply_gradients(gradsvars, global_step=self.train_step,
                                        name='train_op')
    self.model.otensor_names['train_op'] = self.train_op.name
    self.model.otensor_names['global_step'] = self.train_step.name
    
    # After defining the optimizer, initialize all variables
    self.sess.run(tf.global_variables_initializer())

  def prepare_datasets(self, dataset):
    """
    Splits the dataset dictionary into train, validation and test datasets.
    """
    scope = self.scope
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
    
  def train(self, dataset_dict, num_epochs, batch_size=None):
    """
    Train a Model
    """
    sess = self.sess
    
    # get subdatasets
    train_dataset = dataset_dict['train']
    valid_dataset = dataset_dict['valid']
    
    # get cost/summaries
    cost_name = self.model.otensor_names['cost']
    merged_summaries = self.merge_summaries()

    tr_batch_size = batch_size or self.batch_size or 1
    cvalid = np.inf
    for ep in range(num_epochs):
      # gd update
      self.update_gd(train_dataset,
                     batch_size=tr_batch_size)
      ctrain = self.run_ops_from_batches(cost_name,
                                         train_dataset)
      print("ep, cost: {}, {}".format(ep, ctrain))
      
      # Add summaries
      if self.keep_logs:
        self.run_summaries(sess, train_dataset, merged_summaries, ep)
      
      # Save on validation improvement
      if self.save:
        new_cvalid = self.reduce_op_from_batches(sess,
                                                 cost_name,
                                                 valid_dataset)
        if new_cvalid < cvalid:
          print('Valid. cost:', new_cvalid, '... Saving...')
          
          cvalid = new_cvalid
          rslts_path = self.rslts_dir + self.model.main_scope
          global_step = self.model.otensor_names['global_step']
          self.saver.save(sess, rslts_path,
                          global_step=global_step)
  
  def update_gd(self, dataset, batch_size,
                lr=None):
    """
    Perform a single gradient descent update for the variables in this cost.
    
    TODO: Document!
    TODO: Get rid of the feed_dict in favor of tensorflow Queues! Add
    multithreading capabilities
    """
    dataset_iter = batch_iterator_from_dataset(dataset, batch_size,
                                               scope=self.scope)
    
    for feed_dict in dataset_iter:
      if lr is not None:
        feed_dict['lr:0'] = lr
      self.run_ops_from_batches('train_op', feed_dict)
      
  def run_ops_from_batches(self, opnames, feed_dict,
                           reduction=None,
                           axis=None):
    """
    Run an op or a list of ops
    """
    sess = self.sess
    batch_size = get_dataset_batch_size(feed_dict, self.scope)
    if reduction is None:
      self.add_dummy_feeds_to_dataset(feed_dict, batch_size=batch_size)
      vals = sess.run(opnames, feed_dict=feed_dict)
      self.pop_dummy_feeds_from_dataset(feed_dict)
    else:
      raise NotImplementedError("")
    
    return vals
    
  def add_dummy_feeds_to_dataset(self, dataset, batch_size=None):
    """
    Make a feed_dict for sess.run from a dataset.
    
    In particular, add inputs for all the dummy variables defined by the Builder
    associated to this model
    """
    if 'dummy_bsz' in self.dummies:
      dummy_name = self.dummies['dummy_bsz']
      dataset[dummy_name] = np.array([batch_size], dtype=np.int32)      
        
  def pop_dummy_feeds_from_dataset(self, feed_dict):
    """
    Pop the dummy keys from a feeddict
    """
    fd_keys = list(feed_dict.keys())
    for key in fd_keys:
      if key.startswith('dummy'): feed_dict.pop(key)
    
  def run_summaries(self, sess, dataset, merged_summaries, epoch):
    """
    Run all the defined summaries and write to a log
    """
    obskey = next(key for key in dataset)
    batch_size = dataset[obskey].shape[0]

    fd_dct = self.add_dummy_feeds_to_dataset(dataset, batch_size=batch_size)
    summaries = sess.run(merged_summaries, feed_dict=fd_dct)
    self.writer.add_summary(summaries, epoch)
    
  def merge_summaries(self):
    """
    Merge summaries
    """
    merged_summaries = []
    if 'summaries' not in self.directives:
      return merged_summaries

#     summaries_list = self.directives['summaries']
    summaries = self.directives['summaries']
    for summ in summaries:
#       name = summ[0]
      curr_summ = tf.summary.scalar(summ, summaries[summ])
#       curr_summ = tf.summary.scalar(name,
#                                     self.summaries_dict[name](self.nodes, summ[1]))
      merged_summaries.append(curr_summ)

    return tf.summary.merge(merged_summaries)


class fLDSTrainer(GDTrainer):
  """
  """
  def train(self, dataset_dict, num_epochs,
            batch_size=None):
    """
    Train a fLDS-class Model
    """
    model = self.model
    sess = self.sess
    
    # get subdatasets
    train_dataset = dataset_dict['train']
    valid_dataset = dataset_dict['valid']

    # get cost/summaries
    cost_name = self.model.otensor_names['cost']
    merged_summaries = self.merge_summaries()

    tr_batch_size = batch_size or self.batch_size or 1
    cvalid = np.inf
    if self.save:
      print("Saving in", self.rslts_dir)
    for ep in range(num_epochs):
      # compute the posterior
      state_seqs_tr = model.eval_feed(train_dataset, 'Posterior:loc')
      state_seqs_vd = model.eval_feed(valid_dataset, 'Posterior:loc')
      train_dataset[model.otensor_names['StateSeq:main']] = state_seqs_tr
      valid_dataset[model.otensor_names['StateSeq:main']] = state_seqs_vd
      Ntrain, Nvalid = state_seqs_tr.shape[0], state_seqs_vd.shape[0]
      
      # gd step
      t0 = time.time()
      self.update_gd(train_dataset,
                     batch_size=tr_batch_size)
      t1 = time.time()
      print("time:", t1-t0)
      
      ctrain = self.run_ops_from_batches(cost_name,
                                         train_dataset)
      print("ep, cost: {}, {}".format(ep, ctrain/Ntrain))

      # add summaries
      if self.keep_logs:
        self.run_summaries(sess, train_dataset, merged_summaries, ep)
      
      # save on validation improvement
      if self.save:
        new_cvalid = self.run_ops_from_batches(cost_name, valid_dataset)
        if new_cvalid < cvalid:
          print('Valid. cost:', new_cvalid/Nvalid, '... Saving...')
          
          cvalid = new_cvalid
          rslts_path = self.rslts_dir + self.model.main_scope
          global_step = self.model.otensor_names['global_step']
          self.saver.save(sess, rslts_path,
                          global_step=global_step)
          print('')


class VINDTrainer(GDTrainer):
  """
  A Trainer for the VIND class of models
  """
  def _update_default_directives(self, **dirs):
    """
    Update the default directives.
    """
    this_trainer_dirs = {'numfpis' : 5,
                         'wlr_schedule' : False,
                         'endlr' : 1e-4}
    if self.mode == 'new':
      cost_name = self.model.otensor_names['cost']
      cost = tf.get_default_graph().get_tensor_by_name(cost_name)
      this_trainer_dirs['summaries'] = {'cost' : cost}
    this_trainer_dirs.update(dirs)
    super(VINDTrainer, self)._update_directives(**this_trainer_dirs)

  def train(self, dataset_dict, num_epochs,
            batch_size=None,
            lr=None):
    """
    Train a VIND-class Model
    """
    model = self.model
    sess = self.sess
    
    # get subdatasets
    train_dataset = dataset_dict['train']
    valid_dataset = dataset_dict['valid']

    # get cost/summaries
    cost_name = self.model.otensor_names['cost']
    genlp_name = self.model.otensor_names['Generative:logprob']
    lldslp_name = self.model.otensor_names['LLDS:logprob']
    postent_name = self.model.otensor_names['Posterior:entropy']
    merged_summaries = self.merge_summaries()

    tr_batch_size = batch_size or self.batch_size or 1
    cvalid = np.inf
    if self.save:
      print("Saving in", self.rslts_dir)
    for ep in range(num_epochs):
      # fpi step
      if model.otensor_names['StateSeq:main'] not in train_dataset:
        print('First epoch')
        state_seqs_tr = model.eval_feed(train_dataset, 'Recognition:loc')
        state_seqs_vd = model.eval_feed(valid_dataset, 'Recognition:loc')
        train_dataset[model.otensor_names['StateSeq:main']] = state_seqs_tr
        valid_dataset[model.otensor_names['StateSeq:main']] = state_seqs_vd
        Ntrain, Nvalid = state_seqs_tr.shape[0], state_seqs_vd.shape[0]
      else:
        for _ in range(self.directives['numfpis']):
          state_seqs_tr = model.eval_feed(train_dataset, 'Posterior:loc')
          state_seqs_vd = model.eval_feed(valid_dataset, 'Posterior:loc')
          train_dataset[model.otensor_names['StateSeq:main']] = state_seqs_tr
          valid_dataset[model.otensor_names['StateSeq:main']] = state_seqs_vd
        Ntrain, Nvalid = state_seqs_tr.shape[0], state_seqs_vd.shape[0]
      
      # get lr
      if self.directives['wlr_schedule']:
        init_lr, end_lr = self.directives['lr'], self.directives['endlr'] 
        newlr = init_lr - (init_lr - end_lr)*ep/num_epochs
        print('lr:', lr)
        lr = lr or newlr  

      # gd step
      t0 = time.time()
      self.update_gd(train_dataset,
                     batch_size=tr_batch_size,
                     lr=lr)
      t1 = time.time()
      print("time:", t1-t0)
      
      ctrain = self.run_ops_from_batches(cost_name,
                                         train_dataset)
      print("ep, cost: {}, {}".format(ep, ctrain/Ntrain))
      others = self.run_ops_from_batches([genlp_name, lldslp_name, postent_name],
                                         train_dataset)

      # add summaries
      if self.keep_logs:
        self.run_summaries(sess, train_dataset, merged_summaries, ep)
      
      # save on validation improvement
      if self.save:
        new_cvalid = self.run_ops_from_batches(cost_name, valid_dataset)
        if new_cvalid < cvalid:
          print('Valid. cost:', new_cvalid/Nvalid, '... Saving...')
          
          cvalid = new_cvalid
          rslts_path = self.rslts_dir + self.model.main_scope
          global_step = self.model.otensor_names['global_step']
          self.saver.save(sess, rslts_path,
                          global_step=global_step)
          print('')
