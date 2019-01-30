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
from neurolib.encoder.evolution_sequence import NonlinearDynamicswGaussianNoise

# pylint: disable=bad-indentation, no-member, protected-access

class DeepKalmanFilter(Model):
  """
  The Deep Kalman Filter set of models from Krishnan et al
  (https://arxiv.org/abs/1511.05121)
  
  TODO:
  """
  def __init__(self,
               builder=None,
               input_dims=None,
               state_dims=None,
               batch_size=1,
               max_steps=25,
               rnn_cell_class='basic',
               rnn_mode='fwd',
               seq_class=NonlinearDynamicswGaussianNoise,
               is_categorical=False,
               num_labels=None,
               root_rslts_dir=None,
               save_on_valid_improvement=False,
               keep_logs=False,
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
    super(DeepKalmanFilter, self).__init__()
    
    self._main_scope = 'DKF'
    self.root_rslt_dir = root_rslts_dir
    self.save = save_on_valid_improvement
    self.keep_logs = keep_logs

    self.builder = builder
    if self.builder is None:
      self._is_custom_build = False
      if input_dims is None:
        raise ValueError("Missing Argument `input_dims` is required to build "
                         "the default DeepKalmanFilter")
      if state_dims is None:
        raise ValueError("Missing argument `state_dims` is required to build "
                         "the default DeepKalmanFilter")
      else:
        if len(state_dims) < 2:
          raise ValueError("Argument `state_dims` must be a list of size >=2, "
                           "(len(state_dims) = {})".format(len(state_dims)))
      if num_labels is None and is_categorical:
        raise ValueError("Argument num_labels is required to build the default "
                         "RNNClassifier")

      self.rnn_cell_class = rnn_cell_class
      self.seq_class = seq_class
      self.rnn_mode = rnn_mode
      
      # Deal with dimensions
      self.input_dims = input_dims
      self.state_dims = state_dims
      self._dims_to_list()
      self.main_input_dim = self.input_dims[:1]

      self.num_inputs = len(self.input_dims)
      self.rnn_state_dims = (self.state_dims[0:1] if self.rnn_mode == 'fwd' 
                             else self.state_dims[0:2])
      self.num_rnn_state_dims = len(self.rnn_state_dims)
      self.num_inputs_rnn = self.num_rnn_state_dims + self.num_inputs
      self.ds_dims = (self.state_dims[1:] if self.rnn_mode == 'fwd' 
                      else self.state_dims[2:])
      self.num_ds_dims = len(self.ds_dims)
      self.num_inputs_ds = self.num_ds_dims + self.num_rnn_state_dims
    else:
      self._is_custom_build = True
      
      self.input_dims = builder.nodes['inputSeq'].main_output_sizes

    self.is_categorical = is_categorical
    if is_categorical:
      if not num_labels:
        raise ValueError("`num_labels` argument must be a positive integer "
                         "for categorical data")
      self.num_labels = num_labels
    
    self.batch_size = batch_size
    self.max_steps = max_steps
    
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
#     if isinstance(self.output_dims, int):
#       self.output_dims = [[self.output_dims]]

    # Fix dimensions for special cases
    if self.rnn_cell_class == 'lstm':
      if len(self.state_dims) == 1:
        self.state_dims = self.state_dims*2
      
  def _update_default_directives(self,
                                 **directives):
    """
    Update the default directives with user-provided ones.
    """
    self.directives = {'loss_func' : 'elbo',
                       'tr_optimizer' : 'adam',
                       'tr_lr' : 1e-3,
                       'tr_summaries' : [('entropy', ('Recognition',))]}
    
    self.directives.update(directives)
    
    self.tr_dirs = self.extract_dirs('tr') 
            
  def build(self):
    """
    Builds the DeepKalmanFilter
    """
    builder = self.builder
    dirs = self.directives
    if builder is None:
      self.builder = builder = SequentialBuilder(scope=self._main_scope,
                                                 max_steps=self.max_steps)      
      for j, idim in enumerate(self.input_dims):
        is1 = builder.addInputSequence([idim], name='observation_'+str(j)) # FIX!
      evs1 = builder.addEvolutionSequence(state_sizes=self.rnn_state_dims,
                                          num_inputs=self.num_inputs_rnn,
                                          cell_class=self.rnn_cell_class,
                                          name='RNN',
                                          **dirs)
      evs2 = builder.addEvolutionSequence(state_sizes=self.ds_dims,
                                          num_inputs=2,
                                          ev_seq_class=NonlinearDynamicswGaussianNoise,
                                          name='Recognition',
                                          **dirs)
      inn1 = builder.addInnerSequence(self.input_dims[:1],
                                      node_class=NormalTriLNode,
                                      name='Generative')
      os1 = builder.addOutputSequence(name='prediction')
            
      builder.addDirectedLink(is1, evs1, islot=self.num_rnn_state_dims)
      print("dkf; self.num_inputs_ds", self.num_inputs_ds)
      builder.addDirectedLink(evs1, evs2, islot=self.num_ds_dims)
      builder.addDirectedLink(evs2, inn1)
      builder.addDirectedLink(inn1, os1)      
    else:
      self._check_custom_build()
      builder.scope = self._main_scope
    
    builder.build()
    self.nodes = self.builder.nodes
    
    cost = ('elbo', ('Generative', 'Recognition', 'observation_0'))
    self.trainer = GDTrainer(self.builder,
                             cost,
                             name=self._main_scope,
                             root_rslts_dir=self.root_rslt_dir,
                             batch_size=self.batch_size,
                             save_on_valid_improvement=self.save,
                             keep_logs=self.keep_logs,
                             **self.tr_dirs)
    self.save_otensor_names()
    
    self._is_built = True

  def save_otensor_names(self):
    """
    Save tensor names on a separate file for easy access on restore.
    """
    if self.save:
      rslt_dir = self.trainer.rslt_dir
#       onames = set()
      onames = dict()
      with open(rslt_dir + 'output_names', 'wb') as f1:
        for node_name in self.builder.nodes:
          node = self.builder.nodes[node_name]
          for i in range(node.num_expected_outputs):
            onames.add(node.get_output(i).name)
        pickle.dump(onames, f1)
      with open(rslt_dir + 'feed_keys', 'wb') as f2:
        feed_data = set()
        dummies = set()
        for node_name in self.builder.input_nodes:
          node = self.builder.input_nodes[node_name]
          feed_data.add(node.get_output(0).name)
        for node_name in self.builder.dummies:
          dummies.add(node_name)
        feed = {'data' : feed_data, 'dummies' : dummies}
        pickle.dump(feed, f2)
        
  def _check_custom_build(self):
    """
    Check that a user-declared build is consistent with the DeepKalmanFilter names
    """
    if 'prediction' not in self.builder.output_nodes:
      raise AttributeError("Node 'prediction' not found in custom build")
  
  def _check_dataset_correctness(self, dataset):
    """
    Check that the provided dataset is consistent with the DeepKalmanFilter names
    """
    for key in ['train_observation_0', 'valid_observation_0']:
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
  
  def eval(self, node, dataset, oslot=0, lmbda=None):
    """
    """
    dataset_dict = self.prepare_datasets(dataset)
    dataset = dataset_dict['train']
    return self.builder.eval(node, dataset, oslot, lmbda)
  
  def sample(self, input_data, node='prediction', islot=0):
    """
    """
    return Model.sample(self, input_data, node, islot=islot)
  
  
