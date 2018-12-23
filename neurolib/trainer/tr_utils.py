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

# pylint: disable=bad-indentation, no-member, protected-access

def batch_iterator_from_dataset(dataset, batch_size, shuffle=True):
  """
  Make a batch iterator from a dataset
  """
  nsamps = len(list(dataset.values())[0])
  l_inds = np.arange(nsamps)
  if shuffle:
    np.random.shuffle(l_inds)
  for idx in range(nsamps//batch_size):
    yield {key : value[l_inds[idx:idx+batch_size]] for key, value
           in dataset.items()}

def get_keys_and_data(dataset):
  """
  """
  return tuple(zip(*dataset.items()))
