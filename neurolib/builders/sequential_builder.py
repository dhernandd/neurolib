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
from neurolib.encoder.rnn import RNNEvolution, NormalRNN
from neurolib.utils.utils import check_name
from neurolib.builders.static_builder import StaticBuilder
from neurolib.encoder.input import PlaceholderInputNode
from neurolib.encoder.stochasticevseqs import LDSEvolution
from neurolib.encoder.seq_cells import NormalTriLCell

# pylint: disable=bad-indentation, no-member, protected-access

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
  ev_seq_dict = {'rnn' : RNNEvolution}
  
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
    node_name = self.addInput(state_size=state_size,
                              iclass=node_class,
                              name=name,
                              is_sequence=True,
                              **dirs)
    self.input_sequences[node_name] = self.input_nodes[node_name]
    return node_name
  
  @check_name
  def addRNN(self,
             main_inputs,
             state_inputs,
             rnn_class=RNNEvolution,
             cell_class='basic',
             mode='forward',
             name=None,
             name_prefix=None,
             **dirs):
    """
    """
    if isinstance(main_inputs, str):
      main_inputs = [main_inputs]
    if isinstance(state_inputs, str):
      state_inputs = [state_inputs]
    
    num_states = len(state_inputs)
    if num_states < 1:
      raise ValueError("`InnerNodes must have at least one input "
                       "(`num_inputs = {}`".format(num_states))
    
    self.add_node_to_model_graph()
    
    rnn_node = rnn_class(self,
                         main_inputs=main_inputs,
                         state_inputs=state_inputs,
                         cell_class=cell_class,
                         mode=mode,
                         name=name,
                         name_prefix=name_prefix,
                         **dirs)
    self.nodes[rnn_node.name] = self._label_to_node[rnn_node.label] = rnn_node
    
    self.add_directed_links(state_inputs, rnn_node)
    nnodes_so_far = len(state_inputs)
    self.add_directed_links(main_inputs, rnn_node,
                            start_islot=nnodes_so_far)
      
    return rnn_node.name
  
  def addNormalRNN(self, main_inputs, state_inputs, cell_class=NormalTriLCell,
                   mode='forward', name=None, name_prefix=None, **dirs):
    """
    """
    return self.addRNN(main_inputs, state_inputs, 
                       rnn_class=NormalRNN,
                       cell_class=cell_class,
                       mode=mode,
                       name=name,
                       name_prefix=name_prefix,
                       **dirs)
             
  def addInnerSequence(self,
                       state_sizes, 
                       main_inputs,
                       node_class='deterministic', 
                       name=None,
                       **dirs):
    """
    Add an InnerSequence
    """
    return self.addTransformInner(state_sizes,
                                  main_inputs,
                                  node_class=node_class,
                                  is_sequence=True,
                                  name=name,
                                  **dirs)
    
  def addEvolutionwPriors(self,                       
                         state_sizes, 
                         main_inputs,
                         prior_inputs,
                         sec_inputs=None,
#                        num_inputs=1,
                         node_class=LDSEvolution,
                         lmbda=None,
                         name=None,
                         name_prefix=None,
                         **dirs):
    """
    """
    if isinstance(main_inputs, str):
      main_inputs = [main_inputs]
    if isinstance(prior_inputs, str):
      prior_inputs = [prior_inputs]
    if isinstance(sec_inputs, str):
      sec_inputs = [sec_inputs]
    elif sec_inputs is None:
      sec_inputs = []
      
    num_inputs = len(main_inputs)
    if len(main_inputs) < 1:
      raise ValueError("`InnerNodes must have at least one input "
                       "(`num_inputs = {}`".format(num_inputs))
    
    self.add_node_to_model_graph()
    
    if isinstance(node_class, str):
      node_class = self.innernode_dict[node_class]
    enc_node = node_class(self,
                          state_sizes,
                          main_inputs=main_inputs,
                          prior_inputs=prior_inputs,
                          sec_inputs=sec_inputs,
                          lmbda=lmbda,
                          name=name,
                          name_prefix=name_prefix,
                          **dirs)
    self.nodes[enc_node.name] = self._label_to_node[enc_node.label] = enc_node
    
    self.add_directed_links(prior_inputs, enc_node)
    num_inputs = len(prior_inputs)
    self.add_directed_links(main_inputs, enc_node, start_islot=num_inputs)
    num_inputs += len(main_inputs)
    self.add_directed_links(sec_inputs, enc_node, start_islot=num_inputs)
      
    return enc_node.name
  
  def addEvolutionSequence(self,
                           state_sizes,
                           cell_class=None,
                           ev_seq_class='rnn',
                           num_inputs=1,
                           name=None,
                           name_prefix=None,
                           mode='forward',
                           **dirs):
    """
    Add an EvolutionSequence
    """
    self.add_node_to_model_graph()
    
    ev_seq_class = ev_seq_class
    if isinstance(ev_seq_class, str):
      ev_seq_class = self.ev_seq_dict[ev_seq_class]
    
    print('cell_class', cell_class)
    node = ev_seq_class(self,
                        state_sizes,
                        cell_class,
                        num_inputs=num_inputs,
                        name=name,
                        name_prefix=name_prefix,
                        mode=mode,
                        **dirs)
    name = node.name
    self.nodes[name] = node
    self._label_to_node[node.label] = node
    
    return name
