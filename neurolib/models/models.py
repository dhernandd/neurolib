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
import abc
import pickle
import os
from abc import abstractmethod

import numpy as np

from neurolib.utils.dataset_manip import (get_dataset_batch_size,
                                          prepare_restore_user_dataset_for_eval,
                                          merge_datasets)
from neurolib.trainer.costs import *  #pylint: disable=wildcard-import
from _collections import defaultdict

# pylint: disable=bad-indentation, no-member, protected-access


class Model(abc.ABC):
  """
  An abstract class for Machine Learning Models.
  
  Classes that inherit from the abstract class Model will be seen by the client.
  Models are created through a Builder object, which in turn is endowed with an
  interface via which the client may add Nodes and Links to the Model as
  desired.
  
  There are two ways of invoking a Model.
  
  a) Default
    An instance of Model is created using a set of mandatory directives specific
    to the Model as in
      model = Model(*args, **dirs)
    Some control over the model hyperparameters is provided through the
    directives `**dirs`. The Model Builder object is automatically created.
    
  b) Custom
    The instance of Model is created through a custom Builder object
      model = Model(builder=mybuilder)
    where the custom builder object is used to declare the graphical model as
    desired by the user. This provides a lot more control over the model
    hyperparameters since each node of the Model graph can be independently
    tweaked.
    
  The Model classes should implement at the very least the following methods
  
  build(...)
  train(...)
  """
  summaries_dict = {'mse' : mse,
                    'mabsdiff' : mabsdiff,
                    'elbo' : elbo,
                    'elbo_flds' : elbo_flds,
                    'elbo_vind' : elbo_vind,
                    'cross_entropy_wlogits' : cross_entropy_with_logits,
                    'entropy' : entropy,
                    'logprob' : logprob}

  def __init__(self,
               **dirs):
    """
    Initialize a Model.
    
    Starts a tf.Session for this Model
    """
#     tf.reset_default_graph()
    self.sess = tf.Session()
    
    # Deal with the Model directives
    self.directives = {}
    self._update_default_directives(**dirs)
    self.split_model_directives()
    
    self._is_built = False
    
  @property
  def main_scope(self):
    """
    Return the main_scope of the model
    """
    return self._main_scope
      
  def _update_default_directives(self, **dirs):
    """
    Update the Model directives
    """
    self.directives = {'tr_trainer' : 'gd',
                       'tr_optimizer' : 'adam'}
    self.directives.update(dirs)

  def split_model_directives(self):
    """
    Split the model directives by prefix
    """
    dirs = self.directives
    split_directives = defaultdict(dict)
    for d in dirs:
      dsplit = d.split('_')
      if len(dsplit) == 1:
        split_directives['model'][d] = dirs[d]
      else:
        split_directives[dsplit[0]]['_'.join(dsplit[1:])] = dirs[d]
    
    self.directives = split_directives
    
  def add_dummies_to_dataset(self, dataset, batch_size=None):
    """
    Make a feed_dict for sess.run from a dataset.
    
    In particular, add inputs for all the dummy variables defined by the Builder
    associated to this model
    """
    if batch_size is None:
      batch_size = list(dataset.values())[0].shape[0]
    for key in self.dummies:
      dummy_name = self.dummies[key]
      dataset[dummy_name] = np.array([batch_size], dtype=np.int32)
      
    return dataset
  
  def make_input_nodes_feed(self, dataset):
    """
    Make the feed dict for the Model InputNode placeholders.
    
    This method changes the provided dict in place
    """
    scope = self.main_scope
    for key in dataset:
      dataset[scope + '/' + key + '_main:0'] = dataset.pop(key)
    return dataset
    
  def prepare_dataset(self, dataset, batch_size=None):
    """
    Modify the keys of the dataset to fit neurolib + tensorflow expectations.
    
    Specifically, data to a neurolib model is always through an InputNode with a
    single oslot. The name of the outgoing tensor from this oslot, as accessed
    via the `name` property of the Op always has the structure
    
      'ModelScope/NodeName_main:0'
      
    This function takes a dataset with keys corresponding to the InputNodes
    names and returns the same dataset with the keys modified to fit the pattern
    above.
    """
    scope = self.main_scope
    dset = {}
    
    # Always feed to InputNode oslots with well defined, unique otensor names
    for key in dataset:
      dset[scope + '/' + key + '_main:0'] = dataset[key]
    if batch_size is None:
      bsz = dataset[key].shape[0]  #pylint: disable=undefined-loop-variable
    dset = self.add_dummies_to_dataset(dataset, bsz)
    
    return dset
        
  def batch_iterator_from_dataset(self, dataset, shuffle=True):
    """
    Define a batch iterator from a dataset
    
    Args:
      dataset (dict) : A dict whose keys are the names of InputNodes and whose
          values are the data to be fed to them.
      
      shuffle (bool) : Should the data be shuffled?
    """
    nsamps = len(list(dataset.values())[0])
    l_inds = np.arange(nsamps)
    if shuffle:
      np.random.shuffle(l_inds)
    for idx in range(nsamps//self.batch_size):
      yield {key : value[l_inds[idx:idx+self.batch_size]] for key, value
             in dataset.items()}

  @abstractmethod
  def build(self):
    """
    Build the Model from a Builder object. Must be implemented
    """
    raise NotImplementedError("")
        
  def train(self, user_provided_dataset, *additional_usr_datasets,
            num_epochs=100,
            feed_schedule=None,
            **kwargs):
    """
    Train a VIND Model
    """
    self.check_dataset_correctness(user_provided_dataset)
    
    if not additional_usr_datasets:
      self.do_train(user_provided_dataset,
                    num_epochs=num_epochs,
                    **kwargs)
    else:
      if feed_schedule is None:
        raise ValueError("Arg `feed_schedule` is mandatory if more than one "
                         "user_provided_dataset is provided")
      for i, num_epochs in enumerate(feed_schedule):
        self.do_train(user_provided_dataset, num_epochs)
        toadd_dataset = additional_usr_datasets[i]
        user_provided_dataset = merge_datasets(user_provided_dataset,
                                               toadd_dataset)
        
  def check_dataset_correctness(self, user_dataset):
    """
    Checks a that a user-provided dataset is compatible with the Model
    """
    raise NotImplementedError("Please implement me")

  @prepare_restore_user_dataset_for_eval
  def do_train(self, dataset,
               num_epochs=100,
               **kwargs):
    """
    Train the RNNClassifier model from a user-provided dataset.
    
    Functions using a user_provided dataset should 
    
    The user_provided_dataset, provided by the client, should have keys:
    """
    self.trainer.train(dataset,
                       num_epochs,
                       batch_size=self.batch_size,
                       **kwargs)

  @prepare_restore_user_dataset_for_eval
  def eval(self, dataset, names,
           key,
           reduction=None,
           axis=None):
    """
    Eval an output name given a user provided dataset
    """
    if key is None:
      return self.eval_feed(dataset, names,
                            reduction=reduction,
                            axis=axis)
    else:
      return self.eval_feed(dataset[key], names,
                            reduction=reduction,
                            axis=axis)
    
  def eval_feed(self, feed_dict, names,
                reduction=None,
                axis=None):
    """
    Evaluate an output name given an input feed_dict.
    
    This method adds to the feeddict keys for the dummy placeholders of the
    models in-place. Make sure to pop them before exiting!
    
    TODO: Implement this in terms of Builder.eval()
    """
    if isinstance(names, list):
      opnames = [self.otensor_names[name] for name in names]
    else:
      opnames = self.otensor_names[names]
    
    vals = self.run_ops_from_batches(opnames, feed_dict, reduction, axis)
    return vals
  
  @prepare_restore_user_dataset_for_eval
  def eval_ops(self, feed_dict, ops):
    """
    """
    return self.sess.run(ops, feed_dict=feed_dict)
  
  def run_ops_from_batches(self, opnames, feed_dict,
                           reduction=None,
                           axis=None):  #pylint: disable=unused-argument
    """
    Run an op
    """
    sess = self.sess
    batch_size = get_dataset_batch_size(feed_dict, self.main_scope)
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
    
  def restore(self, metafile=None):
    """
    Restore a saved model 
    """
    rslt_dir = self.rslts_dir
    if metafile is None:
      metafile = self.get_latest_metafile_in_rslt_dir(rslt_dir)
      print("... from metafile {}".format(metafile))

      saver = tf.train.import_meta_graph(rslt_dir+metafile)
    else:
      print("... from metafile {}".format(metafile))
    
      saver = tf.train.import_meta_graph(rslt_dir+metafile)
    saver.restore(self.sess, tf.train.latest_checkpoint(rslt_dir))

  def extract_dirs(self, prefix):
    """
    Make a new dictionary with directives matching a prefix
    """
    return {'_'.join(key.split('_')[1:]) : value for key, value 
            in self.directives.items() if key.startswith(prefix)}
    
  @staticmethod
  def get_latest_metafile_in_rslt_dir(rslt_dir):
    """
    Return the latest metafile in the provided directory
    """
    prefixes = [file[:-5] for file in os.listdir(rslt_dir) if 'meta'==file.split('.')[-1]]
    return max([f for f in prefixes], key=lambda f : int(f.split('-')[-1])) + '.meta'

  def save_otensor_names(self):
    """
    Pickle user friendly hashes with extra information about the neurolib model
    graph.
    """
    rslts_dir = self.trainer.rslts_dir
    with open(rslts_dir + 'output_names', 'wb') as f1:
      pickle.dump(self.otensor_names, f1)
    with open(rslts_dir + 'dummies', 'wb') as f2:
      pickle.dump(self.builder.dummies, f2)
    
    return self.otensor_names
