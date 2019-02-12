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

def compute_mean_from_dset(data, axis=None):
  """
  Compute mean of dataset
  """
  return np.mean(data, axis=axis)

def compute_R2_from_sequences(data,
                              preds,
                              axis=None,
                              start_bin=0):
  """
  R2 for a sequence
  """
  mean = np.mean(data[:,start_bin:], axis=axis, keepdims=True)
  d = np.sum((data[:,start_bin:] - mean)**2, axis=axis)
  n = np.sum((data[:,start_bin:] - preds[:,start_bin:])**2, axis=axis)
  return 1.0 - n/d 
