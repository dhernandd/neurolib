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

# pylint: disable=bad-indentation, no-member, protected-access


class Trainer(abc.ABC):
  """
  TODO: Implement training with tensorflow Queues. This is IMPORTANT! Get rid of
  the feed_dict!
  
  TODO: Put the abc functionality to use
  """
  def __init__(self, **dirs):
    """
    """
    self._update_directives(**dirs)
    
  def _update_directives(self, **dirs):
    """
    """
    self.directives = {}
    self.directives.update(dirs)

#   @abc.abstractmethod 
#   def update_gd(self, dataset, batch_size):
#     """
#     """
#     raise NotImplementedError("")
