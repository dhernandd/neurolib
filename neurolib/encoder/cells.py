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
from neurolib.encoder.input import NormalInputNode
from neurolib.encoder.seq_cells import NormalTriLCell

# pylint: disable=bad-indentation, no-member, protected-access

def get_cell_init_states(builder,
                         cell_class,
                         hidden_dims):
  """
  Take a cell and return a list of its initial states.  
  """
  if isinstance(cell_class, str):
    if cell_class == 'basic':
      if len(hidden_dims) != 1: raise ValueError("len(hidden_dims) != 1", len(hidden_dims))
      return [builder.addInput(hidden_dims[0], iclass=NormalInputNode)]
    if cell_class == 'lstm':
      if len(hidden_dims) != 2: raise ValueError("len(hidden_dims) != 2", len(hidden_dims))
      return [builder.addInput(hidden_dims[0], iclass=NormalInputNode),
              builder.addInput(hidden_dims[1], iclass=NormalInputNode)]
  else:
    return cell_class.get_init_states(builder)
  
def are_seq_and_cell_compatible(seq_class, cell_class):
  """
  """
  compatibilities = {'basic' : ['basic',
                                NormalTriLCell],
                     'lstm' : ['lstm']}
  if seq_class in compatibilities and cell_class not in compatibilities[seq_class]:
    raise ValueError("seq_class and cell class not compatible")
