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
import pickle

import numpy as np
import tensorflow as tf

# pylint: disable=bad-indentation, no-member, protected-access

class Restore():
  """
  """
  def __init__(self,
               rslt_dir,
               metafile=None,
               model='DKF'):
    """
    Initialize the Restorer
    """
    print('Initiating Restore...')
    self.scope = model
    
    tf.reset_default_graph()
    self.sess = tf.Session()
    rslt_dir = rslt_dir if rslt_dir[-1] == '/' else rslt_dir + '/'
    self.rslt_dir = rslt_dir
    if metafile is None:
      metafile = self.get_latest_metafile_in_rslt_dir(rslt_dir)
      print("metafile", metafile)
      saver = tf.train.import_meta_graph(rslt_dir+metafile)
    else:
      saver = tf.train.import_meta_graph(rslt_dir+metafile)
    print('Restoring Model `{}` from metafile: {}'.format(model, metafile))
    
    saver.restore(self.sess, tf.train.latest_checkpoint(rslt_dir))
    
    self.get_outputs_dict()
      
  @staticmethod
  def get_latest_metafile_in_rslt_dir(rslt_dir):
    """
    """
    prefixes = [file[:-5] for file in os.listdir(rslt_dir) if 'meta'==file.split('.')[-1]]
    return max([f for f in prefixes], key=lambda f : int(f.split('-')[-1])) + '.meta'
    
  def get_outputs_dict(self):
    """
    """
    with open(self.rslt_dir + 'output_names', 'rb') as f:
      self.output_names = pickle.load(f)
    print('The available output tensors for this graph are:\n')
    for name in self.output_names:
      print('\t', name)
    
    with open(self.rslt_dir + 'feed_keys', 'rb') as f:
      self.feed_keys = pickle.load(f)
    print("\nTo evaluate a tensor, you must feed a dataset dictionary"
          " to `self.eval(...)` with the following keys")
    for key in self.feed_keys:
      print('\t', key, self.feed_keys[key])
    
  def prepare_datasets(self, dataset, chunk='valid'):
    """
    Split the dataset dictionary into train, validation and test datasets.
    """
    dset = {}
    for key in dataset:
      key_split = key.split('_')
      if key_split[0] == chunk:
        inode = "_".join(key_split[1:])
        dset[self.scope + '/' + inode + ':0'] = dataset[key]
    
    return dset
    
  def add_dummy_data(self, dataset):
    """
    Finish off the feed_dict for a sess.run(...) from a dataset.
    
    In particular, add inputs for all the dummy variables defined by the Builder
    associated to this model
    """
    data_key_prefix = self.scope + '/' + 'observation'
    
    # get batch size from data
    obskey = next(key for key in dataset if key.startswith(data_key_prefix)) 
    batch_size = dataset[obskey].shape[0]
    print("batch_size", batch_size)
    
    # define int32 numpy array for the dummy batch size tensors
    dummy_names = self.feed_keys['dummies']
    print("dummy_names", dummy_names)
    for key in dummy_names:
      dataset[key] = np.array([batch_size], dtype=np.int32)
      
    return dataset
  
  def check_input_correctness(self, dset):
    """
    """
    pass
  
  def eval(self, name, _input, dataset_type='valid'):
    """
    """
    self.check_input_correctness(_input)
    prep_dataset = self.prepare_datasets(_input, chunk=dataset_type)
    print("prep_dataset.keys()", prep_dataset.keys())
    fd = self.add_dummy_data(prep_dataset)
    with self.sess.as_default():  #pylint: disable=not-context-manager
      rslt = self.sess.run(name, feed_dict=fd)
    
    return rslt
  