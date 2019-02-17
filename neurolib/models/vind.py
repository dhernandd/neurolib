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
from neurolib.trainer.gd_trainer import GDTrainer, VINDTrainer
from neurolib.encoder.normal import NormalPrecisionNode, LLDSNode
from neurolib.encoder.merge import MergeSeqsNormalwNormalEv
from neurolib.encoder.input import NormalInputNode

# pylint: disable=bad-indentation, no-member, protected-access

class VIND(Model):
  """
  The VIND class of Models. References are:
  
  - Daniel Hernandez, Antonio Khalil Moretti, Ziqiang Wei, Shreya Saxena, John
  Cunningham, Liam Paninski; A Novel Variational Family for Hidden Nonlinear
  Markov Models; arXiv:1811.02459
  """
  def __init__(self,
               mode='new',
               builder=None,
               input_dims=None,
               state_dim=None,
               with_trial_ids=False,
               batch_size=1,
               max_steps=25,
               save_on_valid_improvement=False,
               root_rslts_dir=None,
               keep_logs=False,
               restore_dir=None,
               restore_metafile=None,
               **dirs):
    """
    Initialize VIND
    
    Args:
        input_dims (int or list of list of ints) : The dimensions of the input
            tensors occupying oslots in the Model Graph InputSequences.
        
        state_dim (int or list of list of ints) : The dimensions of the outputs
            of the internal EvolutionSequence.

        num_inputs (int) : The number of Inputs 
        
        builder (SequentialBuilder) :
        
        batch_size (int) : 
        
        max_steps (int) :
        
        cell_class (str, tf Cell or CustomCell) :
        
        dirs (dict) :
    """
    self._main_scope = 'VIND'
    self.mode = mode
    self.builder = builder
    self.save = save_on_valid_improvement
    self.keep_logs = keep_logs

    super(VIND, self).__init__(**dirs)
    
    self.batch_size = batch_size
    if mode == 'new':
      self.root_rslts_dir = root_rslts_dir or 'rslts/'

      self.max_steps = max_steps
      if self.builder is None:
        self._is_custom_build = False
        
        if input_dims is None:
          raise ValueError("Missing Argument `input_dims` is required to build "
                           "the default fLDS")
        if state_dim is None:
          raise ValueError("Missing argument `state_dim` is required to build "
                           "the default fLDS")
        
        # deal with dimensions
        self.input_dims = self._dims_to_list(input_dims)
        self.num_inputs = len(self.input_dims)
        self.state_dim = self._dims_to_list(state_dim)
        self.ydim = self.input_dims[0][0]
        self.xdim = self.state_dim[0][0]
  
      else:
        self._is_custom_build = True

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
    
  def _dims_to_list(self, dims):
    """
    Store the dimensions of the Model in list of lists format
    """
    if isinstance(dims, int):
      return [[dims]]
    return dims
      
  def _update_default_directives(self,
                                 **directives):
    """
    Update the default directives with user-provided ones.
    """
    this_model_dirs = {}
    if self.mode == 'new':
      if self.builder is None:
        this_model_dirs.update({'rec_loc_numlayers' : 2,
                                'rec_loc_numnodes' : 128,
                                'rec_loc_activations' : 'leaky_relu',
                                'rec_loc_netgrowrate' : 1.0,
                                'rec_shareparams' : False,
                                'rec_usett' : True,
                                'gen_wconstprecision' : True,
                                'trainer' : 'gd'})
    this_model_dirs.update({'tr_optimizer' : 'adam',
                            'tr_lr' : 5e-4})
    this_model_dirs.update(directives)
    super(VIND, self)._update_default_directives(**this_model_dirs)
            
  def build(self):
    """
    Build VIND
    """
    builder = self.builder
    ydim = self.ydim
    xdim = self.xdim
    
    rec_dirs = self.directives['rec']
    gen_dirs = self.directives['gen']
    llds_dirs = self.directives['llds']
    if builder is None:
      sec_iseqs = []
      self.builder = builder = SequentialBuilder(scope=self._main_scope,
                                                 max_steps=self.max_steps)
      obs = builder.addInputSequence([[ydim]], name='Observation')
      is2 = builder.addInputSequence([[xdim]], name='StateSeq')
      n1 = builder.addInput([[xdim]], NormalInputNode, name='Prior')
      for i, dim in enumerate(self.input_dims[1:], 1):
        sec_iseqs.append(builder.addInputSequence([dim], name='Input'+str(i)))
      ins1 = builder.addInnerSequence(xdim,
                                      num_inputs=1,
                                      node_class=NormalPrecisionNode,
                                      name='InnSeq',
                                      **rec_dirs)
      evs1 = builder.addInnerSequence([[xdim]],
                                      num_inputs=self.num_inputs,
                                      node_class=LLDSNode,
                                      name='LLDS',
                                      **llds_dirs)
#       evs1 = builder.addEvolutionSequence(xdim,
#                                           num_inputs=self.num_inputs-1,
#                                           cell_class=LLDSCell,
#                                           name='LDS',
#                                           **lds_dirs)
      m1 = builder.addMergeNode(node_list=[ins1, evs1, n1],
                                merge_class=MergeSeqsNormalwNormalEv,
                                name='Recognition',
                                **rec_dirs)
      ins2 = builder.addInnerSequence([[ydim]],
                                      num_inputs=1,
                                      node_class=NormalPrecisionNode,
                                      name='Generative',
                                      **gen_dirs)
      builder.addDirectedLink(obs, ins1, islot=0)
      builder.addDirectedLink(is2, evs1, islot=0)
      for i, iseq in enumerate(sec_iseqs, 1):
        builder.addDirectedLink(iseq, evs1, islot=i)
      builder.addDirectedLink(m1, ins2, islot=0)
      
      self._declare_additional_input_nodes()
    else:
      self._check_custom_build()
      builder.scope = self._main_scope
    
    # build the tensorflow graph
    builder.build()
    self._build_model_specific_outputs()
    
    # pull in builder attributes
    self.nodes = builder.nodes
    self.otensor_names = builder.otensor_names
    self.dummies = builder.dummies
    
    cost_declare = ('elbo', ('Generative', 'Recognition', 'Observation'))
    cost_func = self.summaries_dict[cost_declare[0]]
    self.cost = cost_func(self.nodes, cost_declare[1])
    builder.add_to_output_names('cost', self.cost)

    # trainer
    tr_dirs = self.directives['tr']
    self.trainer = VINDTrainer(self,
                               **tr_dirs)

    if self.save:
      self.save_otensor_names()
      
    self._is_built = True

  def _declare_additional_input_nodes(self):
    """
    Often we would like to call specific nodes in the model graph with user
    provided inputs. This function declares the necessary extra input nodes.
    This is coupled with self._build_model_specific_outputs which adds InnerNode
    calls to the MG
    
    TODO
    """
    pass
  
  def _build_model_specific_outputs(self):
    """
    Often we would like to call specific nodes in the model graph with user
    provided inputs. This function builds these model-specific outputs. This is
    coupled with self._build_model_specific_outputs which adds InnerNode calls
    to the MG
    
    TODO
    """
    pass
  
  def _check_custom_build(self):
    """
    Check that a user-declared build is consistent with the DeepKalmanFilter names
    """
    for key in ['Generative', 'Recognition']:
      if key not in self.builder.nodes:
        raise AttributeError("Node {} not found in custom build".format(key))
  
  def _check_dataset_correctness(self, dataset):
    """
    Check that the provided dataset is consistent with the DeepKalmanFilter names
    """
    keys = ['train_Observation', 'valid_Observation']
    for key in keys:
      if key not in dataset:
        raise AttributeError("dataset must contain key `{}` ".format(key))
      
  def train(self, dataset, num_epochs=100):
    """
    Train the RNNClassifier model. 
    
    The dataset, provided by the client, should have keys:
    """
    self._check_dataset_correctness(dataset)
    dataset_dict = self.prepare_datasets(dataset)

    self.trainer.train(dataset_dict,
                       num_epochs,
                       batch_size=self.batch_size)
  