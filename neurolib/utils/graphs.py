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

def get_session():
  """
  This is a handling similar to that of edward.
  
  Gets the globally defined TensorFlow session.

  If the session is not already defined, then the function will create
  a global session.

  Returns:
    _THIS_SESSION: tf.InteractiveSession.
  """
  global _THIS_SESSION
  if tf.get_default_session() is None:
    _THIS_SESSION = tf.InteractiveSession()
  else:
    _THIS_SESSION = tf.get_default_session()
    
  return _THIS_SESSION