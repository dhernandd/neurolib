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
from neurolib.models.models import Model
from neurolib.builders.sequential_builder import SequentialBuilder
from neurolib.trainer.gd_trainer import GDTrainer

# pylint: disable=bad-indentation, no-member, protected-access

class PredictorRNN(Model):
  """
  RNN-based Model that predicts one or more output sequences given one or more
  sequences of inputs.
  
  [MORE THAN ONE INPUT OR OUTPUT CURRENTLY NOT IMPLEMENTED]
  
  
  """
  def __init__(self,
               input_dims=None,
               state_dims=None,
               output_dims=None,
               num_inputs=1,
               builder=None,
               batch_size=1,
               max_steps=25,
               seq_class='basic',
               cell_class='basic',
               num_labels=None,
               is_categorical=False,
               rslt_dir=None,
               save=False,
               **dirs):
    """
    Initialize the PredictorRNN
    
    Args:
        input_dims (int or list of list of ints) : The dimensions of the input
            tensors occupying oslots in the MG InputSequences.
        
        state_dims (int or list of list of ints) : The dimensions of the outputs
            of the internal EvolutionSequence.

        output_dims (int) : The dimensions of the output tensors occupying
            islots in the MG OutputSequences.
        
        num_inputs (int) : The number of Inputs 
        
        builder (SequentialBuilder) :
        
        batch_size (int) : 
        
        max_steps (int) :
        
        seq_class (str or EvolutionSequence) :
        
        cell_class (str, tf Cell or CustomCell) :
        
        num_labels (int or None) :
        
        is_categorical (bool) : Is the data categorical?
        
        dirs (dict) :
    """
    super(PredictorRNN, self).__init__()
    
    self.builder = builder
    if self.builder is None:
      self._is_custom_build = False
      if input_dims is None:
        raise ValueError("Argument input_dims is required to build the default "
                         "RNNClassifier")
      if state_dims is None:
        raise ValueError("Argument state_dims is required to build the default "
                         "RNNClassifier")
      if output_dims is None and not is_categorical:
        raise ValueError("Argument output_dims is required to build the default "
                         "RNNClassifier")
      if num_labels is None and is_categorical:
        raise ValueError("Argument num_labels is required to build the default "
                         "RNNClassifier")

      self.cell_class = cell_class
      self.seq_class = seq_class
      
      # Deal with dimensions
      self.input_dims = input_dims
      self.state_dims = state_dims
      self.output_dims = output_dims
      self._dims_to_list()
    else:
      self._is_custom_build = True
      
      self.input_dims = builder.nodes['inputSeq'].main_output_sizes
    
    self.num_inputs = num_inputs
    self.is_categorical = is_categorical
    if is_categorical:
      if not num_labels:
        raise ValueError("`num_labels` argument must be a positive integer "
                         "for categorical data")
      self.num_labels = num_labels
    
    self.batch_size = batch_size
    self.max_steps = max_steps
    
    self._main_scope = 'PredictorRNN'
    self.rslt_dir = rslt_dir
    self.save = save
    
    self._update_default_directives(**dirs)

    # Defined on build
    self.nodes = None
    self.cost = None
    self.trainer = None

  def _dims_to_list(self):
    """
    Store the dimensions of the Model in list of lists format
    """
    if isinstance(self.input_dims, int):
      self.input_dims = [[self.input_dims]]
    if isinstance(self.state_dims, int):
      self.state_dims = [[self.state_dims]]
    if isinstance(self.output_dims, int):
      self.output_dims = [[self.output_dims]]

    # Fix dimensions for special cases
    if self.cell_class == 'lstm':
      if len(self.state_dims) == 1:
        self.state_dims = self.state_dims*2
      
  def _update_default_directives(self,
                                 **directives):
    """
    Update the default directives with user-provided ones.
    """
    self.directives = {'trainer' : 'gd',
                       'loss_func' : 'cross_entropy',
                       'gd_optimizer' : 'adam',
                       'lr' : 1e-3}
    
    self.directives.update(directives)
            
  def build(self):
    """
    Builds the PredictorRNN
    """
    builder = self.builder
    dirs = self.directives
    if builder is None:
      self.builder = builder = SequentialBuilder(scope=self._main_scope,
                                                 max_steps=self.max_steps)
      nstate_dims, ninput_dims = len(self.state_dims), len(self.input_dims)
      ninputs_evseq = ninput_dims + nstate_dims
      is1 = builder.addInputSequence(self.input_dims, name='inputSeq')
      evs1 = builder.addEvolutionSequence(state_sizes=self.state_dims,
                                          num_inputs=ninputs_evseq,
                                          ev_seq_class=self.seq_class,
                                          cell_class=self.cell_class,
                                          name='ev_seq',
                                          **dirs)
      if self.is_categorical:
        inn1 = builder.addInnerSequence(self.num_labels)
      else:
        inn1 = builder.addInnerSequence(self.output_dims)
      os1 = builder.addOutputSequence(name='prediction')
            
      builder.addDirectedLink(is1, evs1, islot=nstate_dims)
      builder.addDirectedLink(evs1, inn1)
      builder.addDirectedLink(inn1, os1)      
    else:
      self._check_custom_build()
      builder.scope = self._main_scope
    data_type = 'int32' if self.is_categorical else 'float32'
    is2 = builder.addInputSequence(self.input_dims, name='outputSeq', dtype=data_type)
    os2 = builder.addOutputSequence(name='observation')
    builder.addDirectedLink(is2, os2)
    
    builder.build()
    self.nodes = self.builder.nodes
    
    if self.is_categorical:
      cost = ('cross_entropy_wlogits', ('observation', 'prediction'))
    else:
      cost = ('mse', ('observation', 'prediction'))
    self.trainer = GDTrainer(self.builder,
                             cost,
                             name=self._main_scope,
                             rslt_dir=self.rslt_dir,
                             batch_size=self.batch_size,
                             save=self.save,
                             **dirs)
      
    self._is_built = True

  def _check_custom_build(self):
    """
    Check that a user-declared build is consistent with the RNNPredictor names
    """
    if 'prediction' not in self.builder.output_nodes:
      raise AttributeError("Node 'prediction' not found in custom build")
  
  def _check_dataset_correctness(self, dataset):
    """
    Check that the provided dataset is coconsistent with the RNNPredictor names
    """
    for key in ['train_outputSeq', 'valid_outputSeq']:
      if key not in dataset:
        raise AttributeError("dataset must contain key `{}` ".format(key))
    if not self._is_custom_build:
      for key in ['train_inputSeq', 'valid_inputSeq']:
        if key not in dataset:
          raise AttributeError("dataset must contain key `{}` ".format(key))
      
  def train(self, dataset, num_epochs=100, **dirs):
    """
    Train the RNNClassifier model. 
    
    The dataset, provided by the client, should have keys:
    """
    self._check_dataset_correctness(dataset)
    
    dataset_dict = self.prepare_datasets(dataset)

    self.trainer.train(dataset_dict,
                       num_epochs,
                       batch_size=self.batch_size)
  
  def sample(self, input_data, node='prediction', islot=0):
    """
    """
    return Model.sample(self, input_data, node, islot=islot)
