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
import pickle

from neurolib.models.models import Model
from neurolib.builders.sequential_builder import SequentialBuilder
from neurolib.trainer.gd_trainer import GDTrainer
from neurolib.encoder.input import NormalInputNode

# pylint: disable=bad-indentation, no-member, protected-access

class PredictorRNN(Model):
  """
  RNN-based Model that predicts one or more output sequences given one or more
  sequences of inputs.
  
  [MORE THAN ONE INPUT OR OUTPUT CURRENTLY NOT IMPLEMENTED]
  """
  def __init__(self,
               mode='new',
               builder=None,
               batch_size=1,
               max_steps=25,
#                 seq_class='rnn',
               cell_class='basic',
               input_dims=None,
               state_dims=None,
               output_dims=None,
               num_labels=None,
               is_categorical=False,
               save_on_valid_improvement=False,
               root_rslts_dir='rslts/',
               keep_logs=False,
               restore_dir=None,
               restore_metafile=None,               
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
        
        builder (SequentialBuilder) :
        
        batch_size (int) : 
        
        max_steps (int) :
        
        seq_class (str or EvolutionSequence) :
        
        cell_class (str, tf Cell or CustomCell) :
        
        num_labels (int or None) :
        
        is_categorical (bool) : Is the data categorical?
        
        dirs (dict) :
    """
    self._main_scope = 'PredictorRNN'
    self.mode = mode
    self.builder = builder
    self.save = save_on_valid_improvement
    self.keep_logs = keep_logs

    super(PredictorRNN, self).__init__(**dirs)
    
    self.batch_size = batch_size
    if mode == 'new':
      self.root_rslts_dir = root_rslts_dir

      self.max_steps = max_steps
      self.is_categorical = is_categorical
      if self.builder is None:
        self._is_custom_build = False

        # cell, seq classes
        self.cell_class = cell_class
#         self.seq_class = seq_class

        # shapes
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
        self.input_dims = input_dims
        self.state_dims = state_dims
        self.output_dims = output_dims
        self._dims_to_list()
        
      else:
        self._is_custom_build = True
        self._check_custom_build()
        
        self.input_dims = builder.nodes['Features'].state_sizes
      
      if is_categorical:
        if not num_labels:
          raise ValueError("`num_labels` argument must be a positive integer "
                         "for categorical data")
      self.num_labels = num_labels

      self.build()
    
    elif mode == 'restore':
      print('Initiating Restore...')
      self._is_built = True

      # directory to store results
      if restore_dir is None:
        raise ValueError("Argument `restore_dir` must be provided in "
                         "'restore' mode.")
      self.rslt_dir = restore_dir if restore_dir[-1] == '/' else restore_dir + '/'
      
      # restore
      self.restore(restore_metafile)
      
      # restore output_names and dummies
      with open(self.rslt_dir + 'output_names', 'rb') as f1:
        self.otensor_names = pickle.load(f1)
        print("The following names are available for evaluation:")
        print("\t" + '\n\t'.join(sorted(self.otensor_names.keys())))
      with open(self.rslt_dir + 'dummies', 'rb') as f2:
        self.dummies = pickle.load(f2)
      
      # trainer
      tr_dirs = self.directives['tr']
      self.trainer = GDTrainer(model=self,
                               mode='restore',
                               save_on_valid_improvement=self.save,
                               restore_dir=self.rslt_dir,
                               **tr_dirs)
      
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
    this_model_dirs = {'tr_optimizer' : 'adam',
                       'tr_lr' : 1e-3}
    this_model_dirs.update(directives)
    super(PredictorRNN, self)._update_default_directives(**this_model_dirs)
            
  def build(self):
    """
    Builds the PredictorRNN
    """
    if self._is_built:
      raise ValueError("Node is already built")
    builder = self.builder

    obs_dirs = self.directives['obs']
    ftrs_dirs = self.directives['ftrs']
    rnn_dirs = self.directives['rnn']
    pred_dirs = self.directives['model']
    if builder is None:
      self.builder = builder = SequentialBuilder(scope=self._main_scope,
                                                 max_steps=self.max_steps)
      
      is1 = builder.addInputSequence(self.input_dims,
                                     name='Features',
                                     **ftrs_dirs)
      rnn_priors = []
      for i, state_dim in enumerate(self.state_dims):
        prior = builder.addInput(state_dim[0],
                                 iclass=NormalInputNode,
                                 name='Prior'+str(i))
        rnn_priors.append(prior)
         
      print("PredictorRNN; rnn_priors", rnn_priors)
      rnn = builder.addRNN(main_inputs=is1,
                           state_inputs=rnn_priors,
                           cell_class=self.cell_class,
                           name='RNN',
                           **rnn_dirs)

      if self.is_categorical:
        builder.addInnerSequence(self.num_labels,
                                 main_inputs=rnn,
                                 name='Prediction',
                                 **pred_dirs)
      else:
        builder.addInnerSequence(self.output_dims,
                                 main_inputs=rnn,
                                 name='Prediction',
                                 **pred_dirs)            
    else:
      builder.scope = self._main_scope
    data_type = 'int32' if self.is_categorical else 'float64'
    builder.addInputSequence(self.input_dims,
                             name='Observation',
                             dtype=data_type,
                             **obs_dirs)
    
    # build the tensorflow graph
    builder.build()
    self.nodes = builder.nodes
    self.otensor_names = builder.otensor_names
    self.dummies = self.builder.dummies
    
    # define cost and trainer attribute
    if self.is_categorical:
      cost_declare = ('cross_entropy_wlogits', ('Observation', 'Prediction'))
    else:
      cost_declare = ('mse', ('Observation', 'Prediction'))
    cost_func = self.summaries_dict[cost_declare[0]]
    self.cost = cost_func(self.nodes, cost_declare[1])
    builder.add_to_output_names('cost', self.cost)

    # trainer
    tr_dirs = self.directives['tr']
    self.trainer = GDTrainer(self,
                             root_rslts_dir=self.root_rslts_dir,
                             **tr_dirs)
    if self.save:
      self.save_otensor_names()
    
    print("\nThe following names are available for evaluation:")
    for name in sorted(self.otensor_names.keys()): print('\t', name)

    self._is_built = True

  def _check_custom_build(self):
    """
    Check that a user-declared build is consistent with the RNNPredictor names
    """
    for nname in ['Prediction', 'Features']:
      if nname not in self.builder.nodes:
        raise AttributeError("Node {} not found in custom build".format(nname))
  
  def check_dataset_correctness(self, user_dataset):
    """
    Check that the provided dataset is coconsistent with the RNNPredictor names
    """
    for key in ['train_Observation', 'valid_Observation',
                'train_Features', 'valid_Features']:
      if key not in user_dataset:
        raise AttributeError("dataset must contain key `{}` ".format(key))
