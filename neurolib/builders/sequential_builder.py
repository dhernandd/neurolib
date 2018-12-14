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
from neurolib.encoder.evolution_sequence import (RNNEvolutionSequence, 
                                                 CustomEvolutionSequence)
from neurolib.utils.utils import check_name
from neurolib.builders.static_builder import StaticBuilder
from neurolib.encoder.input import PlaceholderInputNode

# pylint: disable=bad-indentation, no-member, protected-access

sequence_dict = {'rnn' : RNNEvolutionSequence,
                 'custom' : CustomEvolutionSequence}

class SequentialBuilder(StaticBuilder):
  """
  A SequentialBuilder is a Builder for Sequential Models, Models that involve
  sequential data (such as a time series).
  
  Building of a static Model through a SequentialBuilder is done in two stages:
  Declaration and Construction. In the Declaration stage, the input, output and
  inner nodes of the Model are 'added' to the Model graph (MG) and directed
  links - representing the flow of tensors - are defined between them. In the
  Construction stage, a BFS-like algorithm is called that generates a tensorflow
  graph out of the MG specification
  
  A SequentialBuilder defines the following key methods
  
    addOutput(): ...
    
    addInput(): ...
    
    addDirectedLink(): ...
    
    build()
    
  Ex: The following builds a regression Model
      
      builder = SequentialBuilder(scope='regression')
      in0 = builder.addInput(input_dim, name="features")
      enc1 = builder.addInner(output_dim, **dirs)
      out0 = builder.addOutput(name="prediction")
      builder.addDirectedLink(in0, enc1)
      builder.addDirectedLink(enc1, out0)
      
      builder.build()
    
    The 2 input nodes define placeholders for the features and response data
  
  """
  def __init__(self,
               max_steps,
               scope,
               batch_size=None):
    """
    Initialize the SequentialBuilder
    
    Args:
      max_steps (int): The maximum number of steps in the sequence
      
      scope (str): The tensorflow scope of the Model to be built
      
      batch_size (int): The batch size. Defaults to None (unspecified)
    """
    self.max_steps = max_steps
    super(SequentialBuilder, self).__init__(scope, batch_size=batch_size)
    
    self.input_sequences = {}
    self.sequences = {}
    self.output_sequences = {}

  @check_name
  def addInputSequence(self,
                       state_size,
                       name=None,
                       node_class=PlaceholderInputNode,
                       **dirs):
    """
    Add an InputSequence
    """
    node_name = StaticBuilder.addInput(self,
                                       state_size=state_size,
                                       iclass=node_class,
                                       name=name,
                                       is_sequence=True,
                                       **dirs)
    self.input_sequences[node_name] = self.input_nodes[node_name]
    
    return node_name
  
  @check_name
  def addOutputSequence(self, name=None):
    """
    Add an OutputSequence
    """
    return self.addOutput(name=name)
  
  @check_name
  def addInnerSequence(self, 
                       state_sizes, 
                       num_inputs=1,
                       node_class='deterministic', 
                       name=None,
                       **dirs):
    """
    Add an InnerSequence
    """
    return self.addInner(state_sizes,
                         num_inputs=num_inputs,
                         node_class=node_class,
                         is_sequence=True,
                         name=name,
                         **dirs)
  
  def addEvolutionSequence(self,
                           state_sizes, 
                           num_inputs,
                           num_outputs=1,
                           mode='forward',
                           ev_seq_class='rnn',
                           cell_class='basic', 
                           name=None,
                           **dirs):
    """
    Add an EvolutionSequence
    """
    if isinstance(ev_seq_class, str):
      ev_seq_class = sequence_dict[ev_seq_class]
    node = ev_seq_class(self,
                        state_sizes,
                        num_inputs=num_inputs,
                        num_outputs=num_outputs,
                        name=name,
                        mode=mode,
                        cell_class=cell_class,
                        **dirs)
    name = node.name
    self.nodes[name] = node
    self._label_to_node[node.label] = node
    
    return name
      
  def _check_items_do_exist(self):
    """
    TODO:
    """
    pass
      
  def check_graph_correctness(self):
    """
    Checks the coding graph outlined so far. 
    
    TODO:
    """
    pass