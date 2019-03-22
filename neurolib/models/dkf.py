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
from neurolib.encoder.normal import NormalTriLNode
# from neurolib.encoder.evolution_sequence import NonlinearDynamicswGaussianNoise
from neurolib.utils.analysis import compute_R2_from_sequences
from neurolib.encoder.input import NormalInputNode
from neurolib.encoder.seq_cells import NormalTriLCell

# pylint: disable=bad-indentation, no-member, protected-access

class DeepKalmanFilter(Model):
  """
  The Deep Kalman Filter set of models from Krishnan et al
  (https://arxiv.org/abs/1511.05121)
  
  TODO:
  """
  def __init__(self,
               mode='new',
               builder=None,
               input_dims=None,
               rnn_state_dims=None,
               ds_state_dim=None,
               batch_size=1,
               max_steps=25,
               rnn_cell_class='lstm',
               rnn_mode='fwd',
               is_categorical=False,
               num_labels=None,
               save_on_valid_improvement=False,
               root_rslts_dir=None,
               keep_logs=False,
               restore_dir=None,
               restore_metafile=None,
               **dirs):
    """
    Initialize the DeepKalmanFilter
    
    Args:
        input_dims (int or list of list of ints) : The dimensions of the input
            tensors occupying oslots in the Model Graph InputSequences.
        
        state_dims (int or list of list of ints) : The dimensions of the outputs
            of the internal EvolutionSequence.

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
    self._main_scope = 'DKF'
    self.mode = mode
    self.builder = builder
    self.save = save_on_valid_improvement
    self.keep_logs = keep_logs
    
    super(DeepKalmanFilter, self).__init__(**dirs)

    self.batch_size = batch_size
    if mode == 'new':
      self.root_rslts_dir = root_rslts_dir or 'rslts/'

      self.max_steps = max_steps
      self.is_categorical = is_categorical
      if self.builder is None:
        self._is_custom_build = False

        self.rnn_cell_class = rnn_cell_class
#         self.seq_class = seq_class
        self.rnn_mode = rnn_mode
        
        # dims
        if input_dims is None:
          raise ValueError("Missing Argument `input_dims` is required to build "
                           "the default DeepKalmanFilter")
        if rnn_state_dims is None:
          raise ValueError("Missing argument `rnn_state_dims` is required to build "
                           "the default DeepKalmanFilter")
        if ds_state_dim is None:
          raise ValueError("Missing argument `ds_state_dim` is required to build "
                           "the default DeepKalmanFilter")
        if num_labels is None and is_categorical:
          raise ValueError("Argument num_labels is required to build the default "
                           "RNNClassifier")
  
        # Deal with dimensions
        self.input_dims = input_dims
        self.rnn_state_dims = rnn_state_dims
        self.ds_state_dim = ds_state_dim
        self._dims_to_list()
        self.num_inputs = len(self.input_dims)
        self.num_rnn_state_dims = len(self.rnn_state_dims)
        self.num_inputs_rnn = self.num_rnn_state_dims + self.num_inputs

      else:
        self._is_custom_build = True
        self._check_custom_build()
  
      self.is_categorical = is_categorical
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
      self.rslts_dir = restore_dir if restore_dir[-1] == '/' else restore_dir + '/'
      
      # restore
      self.restore(restore_metafile)
      
      # restore output_names and dummies
      with open(self.rslts_dir + 'output_names', 'rb') as f1:
        self.otensor_names = pickle.load(f1)
        print("The following names are available for evaluation:")
        print("\t" + '\n\t'.join(sorted(self.otensor_names.keys())))
      with open(self.rslts_dir + 'dummies', 'rb') as f2:
        self.dummies = pickle.load(f2)
      
      # trainer
      tr_dirs = self.directives['tr']
      self.trainer = GDTrainer(model=self,
                               mode='restore',
                               save_on_valid_improvement=self.save,
                               restore_dir=self.rslts_dir,
                               **tr_dirs)

  def _dims_to_list(self):
    """
    Store the dimensions of the Model in list of lists format
    """
    if isinstance(self.input_dims, int):
      self.input_dims = [[self.input_dims]]
    if isinstance(self.rnn_state_dims, int):
      self.rnn_state_dims = [[self.rnn_state_dims]]
    if isinstance(self.ds_state_dim, int):
      self.ds_state_dim = [[self.ds_state_dim]]

    # Fix dimensions for special cases
    if self.rnn_cell_class == 'lstm':
      if len(self.rnn_state_dims) == 1:
        self.rnn_state_dims = self.rnn_state_dims*2
      
  def _update_default_directives(self,
                                 **directives):
    """
    Update the default directives.
    
    The relevant prefixes are:
    
    rnn : For the model's RNN
    rec :
    gen :
    prrnn :
    prrec :
    tr : 
    """
    this_model_dirs = {}
    if self.mode == 'new':
      if self.builder is None:
        this_model_dirs.update({'rec_cell_loc_numlayers' : 2,
                                'rec_cell_loc_numnodes' : 64,
                                'rec_cell_loc_winitranges' : 1.0,
                                'rec_cell_loc_activations' : 'softplus',
                                'rec_cell_loc_netgrowrate' : 1.0,
                                'rec_cell_shareparams' : False,
                                'rec_cell_scale_numlayers' : 2,
                                'rec_cell_scale_winitranges' : 0.5,
                                'gen_loc_numlayers' : 2,
                                'gen_loc_numnodes' : 64,
                                'gen_loc_activations' : 'softplus',
                                'gen_loc_winitranges' : 1.0,
                                'gen_scale_numlayers' : 2,
                                'gen_scale_numnodes' : 64,
                                'gen_scale_activations' : 'softplus',
                                'gen_scale_winitranges' : 0.5,
                                'trainer' : 'gd',
                                'genclass' : NormalTriLNode})
    this_model_dirs.update({'tr_optimizer' : 'adam',
                            'tr_lr' : 7e-4})
    this_model_dirs.update(directives)
    super(DeepKalmanFilter, self)._update_default_directives(**this_model_dirs)
                
  def build(self):
    """
    Build the DeepKalmanFilter
    """
    builder = self.builder
    
    rnn_dirs = self.directives['rnn']
    gen_dirs = self.directives['gen']
    rec_dirs = self.directives['rec']
    if builder is None:
      self.builder = builder = SequentialBuilder(scope=self._main_scope,
                                                 max_steps=self.max_steps)      
      for j, idim in enumerate(self.input_dims):
        is1 = builder.addInputSequence([idim],
                                       name='Observation_'+str(j)) # FIX!
      
      rnn_priors = []
      for i, rnn_dim in enumerate(self.rnn_state_dims):
        prior = builder.addInput(rnn_dim[0],
                                 iclass=NormalInputNode,
                                 name='RNNPrior'+str(i))
        rnn_priors.append(prior)
      rnn = builder.addRNN(main_inputs=is1,
                           state_inputs=rnn_priors,
                           cell_class=self.rnn_cell_class,
                           name='RNN',
                           **rnn_dirs)                               
      
      prior_evs = builder.addInput(self.ds_state_dim,
                                   iclass=NormalInputNode,
                                   name='PriorEvs')
      evs = builder.addNormalRNN(main_inputs=rnn,
                                 state_inputs=prior_evs,
                                 cell_class=NormalTriLCell,
                                 name='Recognition',
                                 **rec_dirs)
      builder.addInnerSequence(self.input_dims[:1],
                               main_inputs=evs,
                               node_class=NormalTriLNode,
                               name='Generative',
                               **gen_dirs)
            
    else:
      builder.scope = self._main_scope
    
    # build the tensorflow graph
    builder.build()
    self.nodes = builder.nodes
    self.otensor_names = builder.otensor_names
    self.dummies = self.builder.dummies
    
    cost_declare = ('elbo', ('Generative', 'Recognition', 'Observation_0'))
    cost_func = self.summaries_dict[cost_declare[0]]
    self.cost = cost_func(self.nodes, cost_declare[1])
    builder.add_to_output_names('cost', self.cost)
    
    # trainer
    tr_dirs = self.directives['tr']
    self.trainer = GDTrainer(self,
                             **tr_dirs)

    if self.save:
      self.save_otensor_names()
    
    print("\nThe following names are available for evaluation:")
    for name in sorted(self.otensor_names.keys()): print('\t', name)
    
    self._is_built = True
        
  def _check_custom_build(self):
    """
    Check that a user-declared build is consistent with the DeepKalmanFilter names
    """
    for key in ['Generative', 'Recognition']:
      if key not in self.builder.nodes:
        raise AttributeError("Node {} not found in custom build".format(key))
  
  def check_dataset_correctness(self, user_dataset):
    """
    Check that the provided user_dataset is consistent with the DeepKalmanFilter names
    """
    train_keys = ['train_Observation_'+str(i) for i in range(self.num_inputs)]
    valid_keys = ['valid_Observation_'+str(i) for i in range(self.num_inputs)]
    for key in train_keys + valid_keys:
      if key not in user_dataset:
        raise AttributeError("user_dataset must contain key `{}` ".format(key))
      
#   def train(self, dataset, num_epochs=100):
#     """
#     Train the RNNClassifier model. 
#     
#     The dataset, provided by the client, should have keys:
#     """
#     self._check_dataset_correctness(dataset)
#     dataset_dict = self.prepare_datasets(dataset)
# 
#     self.trainer.train(dataset_dict,
#                        num_epochs,
#                        batch_size=self.batch_size)
  
  def anal_R2(self,
              dataset,
              subdset='valid',
              axis=None):
    """
    """
    data = dataset[subdset + '_' + 'Observation_0']
    preds = self.eval('Generative:loc', dataset, key=subdset)[0] # eval returns a list
    R2 = compute_R2_from_sequences(data, preds, axis=axis)
    return R2
    