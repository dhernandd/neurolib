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
import tensorflow as tf

# pylint: disable=bad-indentation, protected-access

  
def infer_shape(x):
  """
  Infers the shape of a tensor for use in reshaping
  """
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.shape.dims is None:
    return tf.shape(x)

  static_shape = x.shape.as_list()
  dynamic_shape = tf.shape(x)

  ret = []
  for i in range(len(static_shape)):
    dim = static_shape[i]
    if dim is None:
      dim = dynamic_shape[i]
    ret.append(dim)

  return ret

def merge_first_two_dims(tensor):
    shape = infer_shape(tensor)
    shape[0] *= shape[1]
    shape.pop(1)
    return tf.reshape(tensor, shape)


def split_first_two_dims(tensor, dim_0, dim_1):
    shape = infer_shape(tensor)
    new_shape = [dim_0] + [dim_1] + shape[1:]
    return tf.reshape(tensor, new_shape)

def match_tensor_shape(shape, tensor, final_rank):
  rt = len(infer_shape(tensor))
  if rt == final_rank:
    return tensor
  dims_to_add = final_rank - rt
  for _ in range(dims_to_add):
    tensor = tf.expand_dims(tensor, axis=0)
  tensor = tf.tile(tensor, shape[:dims_to_add] + [1]*rt)
  return tensor
    