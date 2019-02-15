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
import tensorflow as tf

from neurolib.utils.graphs import get_session
from neurolib.trainer.costs import (mse, mabsdiff, elbo, 
                                    cross_entropy_with_logits,
                                    entropy, logprob)
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
                    'cross_entropy_wlogits' : cross_entropy_with_logits,
                    'entropy' : entropy,
                    'logprob' : logprob}

  def __init__(self,
               **dirs):
    """
    Initialize a Model.
    
    Starts a tf.Session for this Model
    """
    self.sess = tf.Session()
    
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
      
  def prepare_datasets(self, dataset):
    """
    Take a dataset whose keys have the form
    
        prefix_InputNode
        
    where prefix is one of 'train', 'valid' or 'test' and splits it into
    training, validation and test datasets.
    
    Args:
      dataset (dict) : A dataset whose keys are strings with the form 'prefix_InputNode'
          
    Returns:
      A dict with keys 'train', 'valid', 'test' whose values are dicts
      themselves, the values of the latter being the data, keyed by the names of
      the InputNodes that will be fed with it.
    """
    scope = self.main_scope
    train_dataset = {}
    valid_dataset = {}
    test_dataset = {}
    common_keys = []
    for key in dataset:
      d_set, inode = key.split('_')[0], "_".join(key.split('_')[1:])
      if d_set == 'train':
        train_dataset[scope + '/' + inode + '_main:0'] = dataset[key]
      elif d_set == 'valid':
        valid_dataset[scope + '/' + inode + '_main:0'] = dataset[key]
      elif d_set == 'test':
        test_dataset[scope + '/' + inode + '_main:0'] = dataset[key]
      else:
        for dset in [train_dataset, valid_dataset, test_dataset]:
          dset[key] = dataset[key]
          common_keys.append(key)
    
    if train_dataset: train_dataset = self.add_dummies_to_dataset(train_dataset)
    if valid_dataset: valid_dataset = self.add_dummies_to_dataset(valid_dataset)
    if test_dataset: test_dataset = self.add_dummies_to_dataset(test_dataset)

    return {'train' : train_dataset,
            'valid' : valid_dataset, 
            'test' : test_dataset}
    
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
        
  @abstractmethod
  def train(self, dataset, num_epochs):
    """
    Train the Model. Must be implemented
    """
    raise NotImplementedError("")

  def reduce_op_from_batches(self,
                             op,
                             dataset,
                             reduction='mean',
                             num_batches=100):
    """
    Reduce op from batches
    
    Args:
        sess (tf.Session) :
    """
    sess = self.sess
    if self.batch_size is None:
      return sess.run(op, feed_dict=dataset)
    else:
      reduced = 0
      dataset_iter = self.batch_iterator_from_dataset(dataset)
      if reduction == 'mean' or reduction == 'sum':
        for batch_data in dataset_iter:
          reduced += sess.run(op, feed_dict=batch_data)[0]
        if reduction == 'mean': return reduced/(self.batch_size*num_batches)
        else: return reduced
          
  def sample(self, input_data, node, islot=0):
    """
    Deprecated (use model.eval() instead)
    
    Sample from the model graph. For user provided features generates a
    response.
    """
    addcolon0 = lambda s : self.main_scope +  '/' + s + ':0'
    node = self.nodes[node]
    sess = get_session()
    input_data = {addcolon0(key) : value for key, value in input_data.items()}
    if self.batch_size is None:
      return sess.run(node._islot_to_itensor[islot], feed_dict=input_data)
    else:
      num_samples =len(list(input_data.values())[0]) 
      if num_samples % self.batch_size:
        raise ValueError("The number of samples ({})is not divisible by "
                         "self.batch_size({})".format(num_samples,
                                                      self.batch_size))
      res = np.zeros([num_samples] + node._islot_to_shape[islot][1:])
      i = 0
      for batch_data in self.batch_iterator_from_dataset(input_data,
                                                         shuffle=False):
        r = sess.run(node._islot_to_itensor[islot],
                     feed_dict=batch_data)
        res[i:i+self.batch_size] = r
        i += 1
      return res
  
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

  def save_otensor_names(self):
    """
    Pickle user friendly hashes with extra information about the neurolib model
    graph.
    """
    rslts_dir = self.trainer.rslts_dir
    with open(rslts_dir + 'output_names', 'wb') as f1:
      print("self.builder.otensor_names", self.builder.otensor_names)
      pickle.dump(self.otensor_names, f1)
    with open(rslts_dir + 'dummies', 'wb') as f2:
      print("self.builder.dummies", self.builder.dummies)
      pickle.dump(self.builder.dummies, f2)
    
    return self.otensor_names

  def eval(self, names, dataset, key=None):
    """
    Evaluate a tensor given an input dataset.
    
    TODO: Implement this in terms of Builder.eval()
    """
    sess = self.sess
#     if isinstance(names, str): names = [names]
#     opnames = [self.otensor_names[name] for name in names]
    if isinstance(names, list):
      opnames = [self.otensor_names[name] for name in names]
    else:
      opnames = self.otensor_names[names]
    
    if key is None:
      fd = self.prepare_dataset(dataset)
    else:
      dataset_dict = self.prepare_datasets(dataset)
      fd = dataset_dict[key]
    
    return sess.run(opnames, feed_dict=fd)
  