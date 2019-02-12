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

from neurolib.trainer.gd_trainer import GDTrainer
from neurolib.builders.static_builder import StaticBuilder
from neurolib.encoder.normal import NormalTriLNode

# pylint: disable=bad-indentation, no-member, protected-access

class VariationalAutoEncoder(Model):
  """
  The Static Variational Autoencoder.
  """
  def __init__(self,
               mode='new',
               builder=None,
               batch_size=1,
               input_dim=None,
               state_dim=None,
               save_on_valid_improvement=False,
               root_rslts_dir='rslts/',
               keep_logs=False,
               restore_dir=None,
               restore_metafile=None,
               **dirs):
    """
    Initialize the static variational autoencoder
    """
    # The main scope for this model. 
    self._main_scope = 'VAE'
    self.mode = mode
    self.builder = builder
    self.save = save_on_valid_improvement
    self.keep_logs = keep_logs
    
    super(VariationalAutoEncoder, self).__init__(**dirs)
    
    self.batch_size = batch_size
    if mode == 'new':
      # directory to store results
      self.root_rslts_dir = root_rslts_dir
      
      # shapes
      if self.builder is None:
        if input_dim is None or state_dim is None:
          raise ValueError("Arguments `input_dim`, `state_dim` are mandatory "
                           "in the default VAE build")
        self.input_dim = input_dim
        self.state_dim = state_dim
      
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
                               restore_dir=restore_dir,
                               **tr_dirs)
      
  def _update_default_directives(self, **directives):
    """
    Update the default specs with the ones provided by the user.
    """
    this_model_dirs = {}
    if self.mode == 'new':
      if self.builder is None:
        this_model_dirs.update({'rec_loc_numlayers' : 2,
                                'rec_loc_numnodes' : 128,
                                'rec_loc_activations' : 'leaky_relu',
                                'rec_loc_netgrowrate' : 1.0,
                                'rec_shareparams' : False,
                                'trainer' : 'gd',
                                'genclass' : NormalTriLNode,
                                'recclass' : NormalTriLNode})
        this_model_dirs.update(directives)
    this_model_dirs.update({'tr_optimizer' : 'adam'})
    super(VariationalAutoEncoder, self)._update_default_directives(**this_model_dirs)
    
  def build(self):
    """
    Builds the VariationalAutoEncoder.
    
    => E =>
    """
    if self._is_built:
      raise ValueError("Node is already built")
    builder = self.builder

    obs_dirs = self.directives['obs']
    gen_dirs = self.directives['gen']
    rec_dirs = self.directives['rec']
    model_dirs = self.directives['model']
    if builder is None:
      gen_nclass = model_dirs['genclass']
      rec_nclass = model_dirs['recclass']
      
      self.builder = builder = StaticBuilder(scope=self._main_scope)
      
      i1 = builder.addInput(self.input_dim, name='Observation', **obs_dirs)
      rec = builder.addInner(self.state_dim,
                             name='Recognition',
                             node_class=rec_nclass,
                             **rec_dirs)
      gen = builder.addInner(self.input_dim,
                             name='Generative',
                             node_class=gen_nclass,
                             **gen_dirs)

      builder.addDirectedLink(i1, rec, islot=0)
      builder.addDirectedLink(rec, gen, islot=0)
    else:
      self._check_build()
      builder.scope = self._main_scope

    # build the tensorflow graph
    builder.build()
    self.nodes = builder.nodes
    self.otensor_names = builder.otensor_names
    self.dummies = self.builder.dummies
    
    # define cost and trainer attribute
    cost_declare = ('elbo', ('Generative', 'Recognition', 'Observation'))
    cost_func = self.summaries_dict[cost_declare[0]]
    self.cost = cost_func(self.nodes, cost_declare[1])
    self.otensor_names['cost'] = self.cost.name
    
    # trainer
    tr_dirs = self.directives['tr']
    self.trainer = GDTrainer(model=self,
                             root_rslts_dir=self.root_rslts_dir,
                             keep_logs=self.keep_logs,
                             **tr_dirs)
    if self.save:
      self.save_otensor_names()

    self._is_built = True
    
  def _check_build(self):
    """
    TODO:
    """
    pass
  
  def train(self, dataset, num_epochs=100):
    """
    Trains the model. 
    
    The dataset provided by the client should have keys
    
    dset_Observation
    
    where dset is one of ['train', 'valid', 'test']
    """
    self._check_dataset_correctness(dataset)
    dataset_dict = self.prepare_datasets(dataset)

    self.trainer.train(dataset_dict,
                       num_epochs,
                       batch_size=self.batch_size)
    
  def _check_dataset_correctness(self, dataset):
    """
    TODO:
    """
    pass
  