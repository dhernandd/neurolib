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

# pylint: disable=bad-indentation, no-member, protected-access

class Regression(Model):
  """
  The Regression Model implements regression with arbitrary observation_in. It is
  specified by defining a single Model Graph (MG), with a single InputNode and a
  single OutputNode. The MG itself is an acyclic directed graph formed of any
  combination of deterministic encoders nodes.
  
  Ex: A chain of encoders with a single input and output is a Regression model:
  
  I1[ -> d_0] => E1[d_0 -> d_1] => ... => O1[d_{n} -> ]
  
  since it has a single Input node and a single Output node. The following
  directed graph, with the input flowing towards the output through 2 different
  encoder routes (rhombic) is also a Regression model:
  
  I1[ -> d_0] => E1[d_0 -> d_1], E2[d_0 -> d_2]
  
  E1[d_0 -> d_1], E2[d_0 -> d_2] => O1[d_1 + d_2 -> ]
  
  Any user defined Regression must respect the names of the mandatory Input and
  Output nodes, which are fixed to "observation_in" and "response" respectively. 
  
  The default Regression instance builds a Model graph with just one inner
  Encoder
  
  I1[ -> d_0] => E1[d_0 -> d_1] => O1[d_{1} -> ]
  
  The inner encoder node is parameterized by a neural network which can be
  controlled through the directives. Specifically, linear regression is achieved
  by initializing Regression with num_layers=1 and activation=None
  """
  def __init__(self,
               builder=None,
               input_dim=None,
               output_dim=1,
               batch_size=1,
               mode='train',
               restore_dir=None,
               root_rslts_dir='rslts/',
               save_on_valid_improvement=False,
               keep_logs=False,
               restore_metafile=None,               
               **dirs):
    """
    Initialize the Regression Model
    
    Args:
      input_dim (int): The number of observation_in (dimension of the input variable)
      
      output_dim (int): The output dimension
      
      builder (StaticBuilder): An instance of Builder used to build a custom
          Regression model
    """
    self._main_scope = 'Regression'
    super(Regression, self).__init__()
    if mode == 'train':
      self.root_rslts_dir = root_rslts_dir
      self.save = save_on_valid_improvement
      self.keep_logs = keep_logs

      self.builder = builder
      if self.builder is None:
        if input_dim is None:
          raise ValueError("Argument input_dim is required to build the default "
                           "Regression")
        elif output_dim > 1:
          raise NotImplementedError("Multivariate regression is not implemented")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
      
      self._update_default_directives(**dirs)
  
      # Define upon build
      self.nodes = None
      self.trainer = None
      self.otensor_names = None
    elif mode == 'restore':
      self._is_built = True
      print('Initiating Restore...')
      if restore_dir is None:
        raise ValueError("Argument `restore_dir` must be provided in "
                         "'restore' mode.")
      self.rslt_dir = restore_dir if restore_dir[-1] == '/' else restore_dir + '/'
      
      self._restore(restore_metafile)
      
      with open(self.rslt_dir + 'output_names', 'rb') as f1:
        self.ops_names = pickle.load(f1)
        print("The following ops are available:")
        print("\t" + '\n\t'.join(sorted(self.ops_names.keys())))
      
      self.trainer = GDTrainer(self,
                               mode='restore')
        
#       self.trainer = GDTrainer()

  def _update_default_directives(self, **directives):
    """
    Update the default directives with user-provided ones.
    """
    self.directives = {'trainer' : 'gd',
                       'gd_optimizer' : 'adam'}
    if self.builder is None:
      self.directives.update({'num_layers' : 2,
                              'num_nodes' : 128,
                              'activation' : 'leaky_relu',
                              'net_grow_rate' : 1.0,
                              'share_params' : False})
    
    self.directives.update(directives)

  def build(self):
    """
    Builds the Regression.
    
    => E =>
    """
    builder = self.builder
    dirs = self.directives
    if builder is None:
      self.builder = builder = StaticBuilder(scope=self._main_scope)
      
      in0 = builder.addInput(self.input_dim,
                             name="Features",
                             **dirs)
      enc1 = builder.addInner(1,
                              num_inputs=self.output_dim,
                              **dirs)
      out0 = builder.addOutput(name="prediction")

      builder.addDirectedLink(in0, enc1)
      builder.addDirectedLink(enc1, out0)

    else:
      self._check_custom_build()
      builder.scope = self._main_scope
    in1 = builder.addInput(self.output_dim,
                           name="Observation")
    out1 = builder.addOutput(name="response")
    builder.addDirectedLink(in1, out1)

    # Build the tensorflow graph
    builder.build()
    self.nodes = builder.nodes
    self.otensor_names = builder.otensor_names
    
    cost = ('mse', ('prediction', 'response'))
    self.trainer = GDTrainer(model=self,
#                              builder=builder, 
                             cost=cost,
                             name=self._main_scope,
                             batch_size=self.batch_size,
                             root_rslts_dir=self.root_rslts_dir,
                             save_on_valid_improvement=self.save,
                             keep_logs=self.keep_logs,
                             **dirs)
    if self.save:  
      self.save_otensor_names()
      
    self._is_built = True
    
  def _check_custom_build(self):
    """
    Check that the user-declared build is consistent with the Regression class
    """
    if 'prediction' not in self.builder.nodes:
      raise AttributeError("Node 'prediction' not found in CustomBuild")
    
  def _check_dataset_correctness(self, dataset):
    """
    Check that the provided dataset adheres to the Regression class specification
    """
    must_have_keys = ['train_Observation', 'train_Features',
                      'valid_Observation', 'valid_Features']
    for key in must_have_keys:
      if key not in dataset:
        raise AttributeError("dataset does not have key {}".format(key))
  
  def train(self, dataset, num_epochs=100, **dirs):
    """
    Train the Regression model. 
    
    The dataset, provided by the client, should have keys
    
    train_Observation, train_Features
    valid_Observation, valid_Features
    test_Observation, test_Features
    """
    self._check_dataset_correctness(dataset)
    dataset_dict = self.prepare_datasets(dataset)

    self.trainer.train(dataset_dict,
                       num_epochs,
                       batch_size=self.batch_size)
    
  def sample(self, input_data, node='prediction', islot=0):
    """
    Sample from Regression
    """
    return Model.sample(self, input_data, node, islot=islot)
    