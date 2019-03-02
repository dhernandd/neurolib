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
import numpy as np

# pylint: disable=bad-indentation, no-member

def prepare_restore_user_dataset_for_eval(f):
  """
  Method decorator to use in every method of a Model subclass that takes a user-
  provided dataset as an input
  """
  is_tr_vd_dset = lambda dset : (('train' in 
                                  [key.split('_')[0] for key in dset.keys()])
                                  and 
                                  ('valid' in 
                                   [key.split('_')[0] for key in dset.keys()]))
  def wrapped_eval(self, usr_dataset, *args, **kwargs):
    """
    Runs an eval-type function from a user-provided dataset.
    
    This function must change a user provided dataset in-place so as to save
    memory for larger datasets. The dataset must hence be restored to user-
    provided form before exiting the function.
    """
    scope = self.main_scope
    if is_tr_vd_dset(usr_dataset):
      feed_dict = make_train_valid_test_feeddicts(usr_dataset, scope)
      vals = f(self, feed_dict, *args, **kwargs)
      restore_dataset(usr_dataset, is_train_valid=True)
    else:
      feed_dict = make_feeddict(usr_dataset)
      vals = f(self, feed_dict, *args, **kwargs)
      restore_dataset(usr_dataset)
    return vals

  return wrapped_eval

def make_train_valid_test_feeddicts(user_provided_dset, scope=None):
  """
  Take a user_provided_dset whose keys have the form
  
      prefix_InputNode
      
  where prefix is one of 'train', 'valid' or 'test' and splits it into
  training, validation and test tensorflow feeddicts.
  
  NOTE: This method works in-place
  
  Args:
    user_provided_dset (dict) : A user_provided_dset whose keys are strings with the form 'prefix_InputNode'
        
  Returns:
    A dict with keys 'train', 'valid', 'test' whose values are dicts
    themselves, the values of the latter being the data, keyed by the names of
    the InputNodes that will be fed with it.
  """
  user_keys = list(user_provided_dset.keys())
  user_provided_dset['train'] = {}
  user_provided_dset['valid'] = {}
  for key in user_keys:
    d_set, inode = key.split('_')[0], "_".join(key.split('_')[1:])
    feed_key = make_input_node_feed_key(inode, scope)
    if d_set == 'train':
      user_provided_dset['train'][feed_key] = user_provided_dset.pop(key)
    elif d_set == 'valid':
      user_provided_dset['valid'][feed_key] = user_provided_dset.pop(key)

  return user_provided_dset
  
def make_feeddict(user_provided_dset):
  """
  Take a user_provided_dset whose keys have the form
  
      InputNode
  
  and forms a tensorflow feeddict with it
  
  NOTE: This method operates in-place
  """
  for key in user_provided_dset:
    feed_key = make_input_node_feed_key(key)
    user_provided_dset[feed_key] = user_provided_dset.pop(key)
  return user_provided_dset

def restore_dataset(feed_dict, is_train_valid=False):
  """
  Restore a user-provided dataset from a feeddict used by the neurolib's methods
  
  NOTE: This method operates in-place
  """
  if not is_train_valid:
    fd_keys = list(feed_dict.keys())
    for key in fd_keys:
      feed_dict[strip_feedkey(key)] = feed_dict.pop(key)
  else:
    for subdset in ['train', 'valid']:
      fd_keys = list(feed_dict[subdset].keys())
      for key in fd_keys:
        feed_dict[subdset + '_' + strip_feedkey(key)] = feed_dict[subdset].pop(key)
      del feed_dict[subdset]
  return feed_dict
      
def make_input_node_feed_key(oname, scope=None):
  """
  Turn a Input Node name to a tensorflow feeddict key inside a given scope
  """
  return oname + '_main:0' if scope is None else scope + '/' + oname + '_main:0'

def strip_feedkey(key):
  """
  Returns an InputNode name out of a feeddict key
  """
  return '_'.join(key.split('/')[1].split('_')[:-1])

def merge_datasets(dataset, add_dataset):
  """
  NOTE: This function should not work in-place!
  """
  print("Merging datasets...")
  new_dataset = {}
  for key in dataset:
    new_dataset[key] = np.concatenate((dataset[key], add_dataset[key]))
  return new_dataset
    
def get_dataset_batch_size(dataset, scope=''):
  """
  """
  obskey = next(key for key in dataset if key.startswith(scope))
  return dataset[obskey].shape[0]

def batch_iterator_from_dataset(dataset, batch_size, scope='',
                                shuffle=True):
  """
  Make a batch iterator from a dataset
  """
  nsamps = get_dataset_batch_size(dataset, scope=scope)
  l_inds = np.arange(nsamps)
  if shuffle:
    np.random.shuffle(l_inds)
  for idx in range(nsamps//batch_size):
    batch = {key : value[l_inds[idx:idx+batch_size]] for key, value
             in dataset.items()}
    yield batch
  