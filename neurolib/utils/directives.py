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

# pylint: disable=bad-indentation, no-member

from collections import defaultdict


class NodeDirectives():
  """
  """
  def __init__(self,
               directives):
    """
    """
    # dict of dicts where the keys are the prefixes
    dirsets2 = defaultdict(dict)
    for directive in directives:
      split_dir = directive.split('_')
      if len(split_dir) == 1:
        prefix, attribute = '', split_dir[0]
      else:
        prefix, attribute = split_dir[0], '_'.join(split_dir[1:])
        prefix = prefix + '_'
      dirsets2[prefix][attribute] = directives[directive]
      
    for dirset in dirsets2:
      self._process_directives(dirset, dirsets2[dirset])

  def _process_directives(self, dirset, directives):
    """
    """
    if dirset != 'outputname_': 
      self._process_network_directives(dirset, directives)
      if directives:
        self._process_the_rest(dirset, directives)
    if dirset == 'outputname_':
      self._process_names(directives)
    
  def _process_network_directives(self, dirset, directives):
    """
    """
    if any([key in directives for key in ('layers', 'numlayers')]):
      self._process_layers(dirset, directives)
      self._process_numlayers(dirset, directives)
      self._process_netgrowrate(dirset, directives)
      self._process_numnodes(dirset, directives)
      self._process_activations(dirset, directives)
      self._process_rangeNNws(dirset, directives)
    
  def _process_layers(self, dirset, directives):
    """
    """
    if 'layers' in directives:
      list_of_layers = directives.pop('layers')
      setattr(self, dirset+'layers', list_of_layers)
      setattr(self, dirset+'numlayers', len(list_of_layers))
  
  def _process_numlayers(self, dirset, directives):
    """
    """
    if hasattr(self, dirset+'numlayers'):
      return
    if 'numlayers' in directives:
      numlayers = directives.pop('numlayers')
      setattr(self, dirset+'numlayers', numlayers)
      setattr(self, dirset+'layers', ['full' for _ in range(numlayers)])
      
  def _process_netgrowrate(self, dirset, directives):
    """
    """
    if 'netgrowrate' in directives:
      ngr = directives.pop('netgrowrate')
      setattr(self, dirset+'netgrowrate', ngr)
      
  def _process_numnodes(self, dirset, directives):
    """
    """
    if 'numnodes' in directives:
      numlayers = getattr(self, dirset+'numlayers')
      netgrowrate = getattr(self, dirset+'netgrowrate', 1.0)
      numnodes = directives.pop('numnodes')
      if isinstance(numnodes, int):
        setattr(self, dirset+'numnodes', [numnodes*int(netgrowrate) for _ 
                                          in range(numlayers-1)])
      else:
        assert len(numnodes) == numlayers - 1
        setattr(self, dirset+'numnodes', numnodes)
  
  def _process_activations(self, dirset, directives):
    """
    """
    if 'activations' in directives:
      numlayers = getattr(self, dirset+'numlayers')
      activations = directives.pop('activations')
      if isinstance(activations, str):
        acts = [activations for _ in range(numlayers-1)] + ['None']
        setattr(self, dirset+'activations', acts)
      else:
        assert len(activations) == numlayers
        setattr(self, dirset+'activations', activations)
        
  def _process_rangeNNws(self, dirset, directives):
    """
    """
    if 'rangeNNws' in directives:
      numlayers = getattr(self, dirset+'numlayers')
      r = directives.pop('rangeNNws')
      if isinstance(r, float):
        ranges = [r for _ in range(numlayers-1)] + [1.0]
        setattr(self, dirset+'rangeNNws', ranges)
      else:
        assert len(r) == numlayers
        setattr(self, dirset+'rangeNNws', r)
    
  def _process_names(self, directives):
    """
    """
    s = sorted([key for key in directives])
    self.output_names = [directives[k] for k in s]  #pylint: disable=attribute-defined-outside-init
    
  def _process_the_rest(self, dirset, directives):
    """
    """
    for d in directives: setattr(self, dirset+d, directives[d])


class TrainerDirectives():
  """
  """
  def __init__(self,
               directives):
    """
    """
    raise NotImplementedError("")
    