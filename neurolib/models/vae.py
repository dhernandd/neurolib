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

from neurolib.trainer.gd_trainer import GDTrainer
from neurolib.builders.static_builder import StaticBuilder
from neurolib.encoder.normal import NormalTriLNode

# pylint: disable=bad-indentation, no-member, protected-access

class VariationalAutoEncoder(Model):
  """
  The Static Variational Autoencoder.   
  """
  def __init__(self,
               input_dim=None,
               state_dim=None,
               builder=None,
               batch_size=1,
               rslt_dir='rslts/',
               save=False,
               **dirs):
    """
    Initialize the static variational autoencoder
    """
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.batch_size = batch_size
    self.rslt_dir = rslt_dir
    self.save = save
    
    # The main scope for this model. 
    self._main_scope = 'VAE'

    super(VariationalAutoEncoder, self).__init__()
    self.builder = builder
    if builder is None:
      if input_dim is None or state_dim is None:
        raise ValueError("input_dim, state_dim are mandatory "
                         "in the default build")
        
    self._update_default_directives(**dirs)

    # Initialize at build
    self.trainer = None
    self.cost = None
    self.nodes = None
      
  def _update_default_directives(self, **dirs):
    """
    Update the default specs with the ones provided by the user.
    """
    self.directives = {'num_layers_0' : 2,
                       'num_nodes_0' : 128,
                       'activation_0' : 'leaky_relu',
                       'net_grow_rate_0' : 1.0,
                       'share_params' : False,
                       'trainer' : 'gd',
                       'gd_optimizer' : 'adam',
                       'gen_node_class' : NormalTriLNode,
                       'rec_node_class' : NormalTriLNode}
    self.directives.update(dirs)
    
  def build(self):
    """
    Builds the VariationalAutoEncoder.
    
    => E =>
    """
    dirs = self.directives
    builder = self.builder
    if builder is None:
      gen_nclass = dirs['gen_node_class']
      rec_nclass = dirs['rec_node_class']
      
      self.builder = builder = StaticBuilder(scope=self._main_scope)
      
      i1 = builder.addInput(self.state_dim, name='observation', **dirs)
      enc1 = builder.addInner(self.input_dim, name='Recognition',
                              node_class=rec_nclass,
                              **dirs)
      enc0 = builder.addInner(self.state_dim, name='Generative',
                              node_class=gen_nclass,
                              **dirs)
      o1 = builder.addOutput(name='copy')

      builder.addDirectedLink(i1, enc1)
      builder.addDirectedLink(enc1, enc0, oslot=0)
      builder.addDirectedLink(enc0, o1, oslot=0)      
    else:
      self._check_build()
      builder.scope = self._main_scope

    # Build the tensorflow graph
    builder.build()
    self.nodes = self.builder.nodes
    
    cost = ('elbo', ('Generative', 'Recognition', 'observation'))
    self.trainer = GDTrainer(self.builder,
                             cost,
                             name=self._main_scope,
                             rslt_dir=self.rslt_dir,
                             batch_size=self.batch_size,
                             save=self.save,
                             **dirs)
    
    self._is_built = True
    
  def _check_build(self):
    """
    """
    pass
  
  def train(self, dataset, num_epochs=100, **dirs):
    """
    Trains the model. 
    
    The dataset provided by the client should have keys
    
    train_features, train_response
    valid_features, valid_response
    test_features, test_response
    
    where # is the number of the corresponding Input node, see model graph.
    """
    self._check_dataset_correctness(dataset)
    
    dataset_dict = self.prepare_datasets(dataset)

    self.trainer.train(dataset_dict,
                       num_epochs,
                       batch_size=self.batch_size)
    
  def _check_dataset_correctness(self, dataset):
    """
    """
    pass