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
  Standard Regression.
  
  The Regression Model is specified by defining a single Model Graph (MG), with
  a single InputNode and a single OutputNode. The Regression MG itself is an
  acyclic directed graph formed by any combination of deterministic encoders
  nodes.

  The default Regression instance builds a Model graph with one inner Encoder
  node,
  
  I1[ -> d_0] => E1[d_0 -> d_1] => O1[d_{1} -> ]
  
  The inner encoder node is parameterized by a neural network which can be
  controlled through the directives. Specifically, linear regression is achieved
  by initializing Regression with num_layers=1 and activation=None

  On the other hand, it is possible to define a Custom Regression node by
  passing a Builder object to __init__().
  
  Ex. 1: A chain of encoders with a single input and output is a Regression model:
  
  I1[ -> d_0] => E1[d_0 -> d_1] => ... => O1[d_{n} -> ]
  
  since it has a single Input node and a single Output node. 
  
  Ex. 2: The following directed graph, with the input flowing towards the output through 2 different
  encoder routes (rhombic) is also a Regression model:
  
  I1[ -> d_0] => E1[d_0 -> d_1], E2[d_0 -> d_2]
  
  E1[d_0 -> d_1], E2[d_0 -> d_2] => O1[d_1 + d_2 -> ]
  
  Any custom built Regression model must respect the name of the mandatory
  InputNode representing the features input, which is set to 'Features'.
  """
  def __init__(self,
               mode='new',
               builder=None,
               batch_size=1,
               output_dim=1,
               input_dim=None,
               save_on_valid_improvement=False,
               root_rslts_dir='rslts/',
               keep_logs=False,
               restore_dir=None,
               restore_metafile=None,               
               **dirs):
    """
    Initialize the Regression Model
    
    Args:
      builder (StaticBuilder) : For custom Regression models, an instance of
          Builder used to build the custmo Model Graph
      
      input_dim (int) : The number of observation_in (dimension of the input variable)
      
      output_dim (int) : The output dimension
      
      batch_size (int) : The training batch_size (may be different from the
          `batch_size` property of `self.builder`)
          
      mode (str) : Either 'new' or 'restore'. Defaults to 'new'
      
      restore_dir (str) : In 'restore' mode, the directory where the save_files
          are stored
      
      root_rslts_dir='rslts/',

      save_on_valid_improvement=False,
               
      keep_logs=False,

      restore_metafile=None,               
    """
    self._main_scope = 'Regression'
    self.mode = mode
    self.builder = builder
    self.save = save_on_valid_improvement
    self.keep_logs = keep_logs
    
    super(Regression, self).__init__(**dirs)

    self.batch_size = batch_size
    if mode == 'new':
      # directory to store results
      self.root_rslts_dir = root_rslts_dir

      # shapes
      if self.builder is None:
        if input_dim is None:
          raise ValueError("Argument input_dim is mandatory in the default "
                           "Regression build")
        elif output_dim > 1:
          raise NotImplementedError("Multivariate regression is not implemented")

        self.input_dim = input_dim
      self.output_dim = output_dim
      
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
      self.trainer = GDTrainer(model=self,
                               mode='restore',
                               save_on_valid_improvement=self.save,
                               restore_dir=restore_dir,
                               **self.directives)
        
  def _update_default_directives(self, **directives):
    """
    Update the default directives with user-provided ones.
    """
    this_node_dirs = {}
    if self.mode == 'new' and self.builder is None:
      this_node_dirs.update({'enc_numlayers' : 1,
                             'enc_numnodes' : 128,
                             'enc_activations' : 'leaky_relu',
                             'enc_netgrowrate' : 1.0})
    this_node_dirs.update(directives)
    super(Regression, self)._update_default_directives(**this_node_dirs)

  def build(self):
    """
    Builds the Regression.
    
    => E =>
    """
    if self._is_built:
      raise ValueError("Node is already built")
    builder = self.builder
    
    ftrs_dirs = self.directives['ftrs']
    enc_dirs = self.directives['enc']
    tr_dirs = self.directives['tr']
    if builder is None:
      self.builder = builder = StaticBuilder(scope=self._main_scope)
      
      in0 = builder.addInput(self.input_dim,
                             name="Features",
                             **ftrs_dirs)
      builder.addTransformInner(state_size=1,
                                main_inputs=in0,
                                name='Prediction',
                                **enc_dirs)
    else:
      self._check_custom_build()
      builder.scope = self._main_scope
    builder.addInput(self.output_dim, name="Observation")

    # build the tensorflow graph
    builder.build()
    self.nodes = builder.nodes
    self.otensor_names = builder.otensor_names
    self.dummies = self.builder.dummies
    
    # Define cost and trainer attribute
    cost_declare = ('mse', ('Prediction', 'Observation'))
    cost_func = self.summaries_dict[cost_declare[0]]
    self.cost = cost_func(self.nodes, cost_declare[1])
    self.otensor_names['cost'] = self.cost.name
    
    # trainer object
    self.trainer = GDTrainer(model=self,
                             keep_logs=self.keep_logs,
                             **tr_dirs)
    if self.save: # save
      self.save_otensor_names()
      
    print("\nThe following names are available for evaluation:")
    for name in sorted(self.otensor_names.keys()): print('\t', name)
      
    self._is_built = True
    
  def _check_custom_build(self):
    """
    Check that the user-declared build is consistent with the Regression class
    """
    if 'Prediction' not in self.builder.nodes:
      raise AttributeError("Node 'Prediction' not found in CustomBuild")
    
  def check_dataset_correctness(self, user_dataset):
    """
    Check that the provided user_dataset adheres to the Regression class specification
    """
    must_have_keys = ['train_Observation', 'train_Features',
                      'valid_Observation', 'valid_Features']
    for key in must_have_keys:
      if key not in user_dataset:
        raise AttributeError("user_dataset does not have key {}".format(key))
  
#   def train(self, dataset, num_epochs=100):
#     """
#     Train the Regression model. 
#     
#     The dataset, provided by the client, should have keys
#     
#     train_Observation, train_Features
#     valid_Observation, valid_Features
#     test_Observation, test_Features
#     """
#     self._check_dataset_correctness(dataset)
#     dataset_dict = self.prepare_datasets(dataset)
# 
#     self.trainer.train(dataset_dict,
#                        num_epochs,
#                        batch_size=self.batch_size)
#         