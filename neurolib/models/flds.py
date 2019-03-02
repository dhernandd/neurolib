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

import numpy as np
import tensorflow as tf

from neurolib.models.models import Model
from neurolib.builders.sequential_builder import SequentialBuilder
from neurolib.trainer.gd_trainer import fLDSTrainer
from neurolib.encoder.normal import NormalPrecisionNode
from neurolib.encoder.stochasticevseqs import LDSEvolution
from neurolib.encoder.merge import MergeSeqsNormalwNormalEv
from neurolib.encoder.input import NormalInputNode
from neurolib.utils.analysis import compute_R2_from_sequences

# pylint: disable=bad-indentation, no-member, protected-access

class fLDS(Model):
  """
  The fLDS class of Models. References are:
  
  - Gao Y, Archer E, Paninski L, Cunningham JP (2016). Linear dynamical neural
  population models through nonlinear embeddings; https://arxiv.org/abs/1605.08454
  
  - Archer E, Park IM, Büsing L, Cunningham JP, Paninski L. (2016) Black box
  variational inference for state space models.
  
  
  TODO:
  """
  def __init__(self,
               mode='new',
               builder=None,
               main_input_dim=None,
               state_dim=None,
#                with_trial_ids=False,
               batch_size=1,
               max_steps=25,
               save_on_valid_improvement=False,
               root_rslts_dir=None,
               keep_logs=False,
               restore_dir=None,
               restore_metafile=None,
               **dirs):
    """
    Initialize the fLDS
    
    Args:
        main_input_dim (int or list of list of ints) : The dimensions of the input
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
    self._main_scope = 'fLDS'
    self.mode = mode
    self.builder = builder
    self.save = save_on_valid_improvement
    self.keep_logs = keep_logs

    super(fLDS, self).__init__(**dirs)
    
    self.batch_size = batch_size
    if mode == 'new':
      self.root_rslts_dir = root_rslts_dir or 'rslts/'

      self.max_steps = max_steps
      if self.builder is None:
        self._is_custom_build = False
        
        if main_input_dim is None:
          raise ValueError("Missing Argument `main_input_dim` is required to build "
                           "the default fLDS")
        if state_dim is None:
          raise ValueError("Missing argument `state_dim` is required to build "
                           "the default fLDS")
        
        # deal with dimensions
        self.main_input_dim = self._dims_to_list(main_input_dim)
        self.state_dims = self._dims_to_list(state_dim)
        self.ydim = self.main_input_dim[0][0]
        self.xdim = self.state_dims[0][0]
  
        self.num_inputs = len(self.main_input_dim)
        self.ds_state_dim = self.state_dims
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
      self.trainer = fLDSTrainer(model=self,
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
        this_model_dirs.update({'rec_loc_numlayers' : 3,
                                'rec_loc_numnodes' : 64,
                                'rec_loc_activations' : 'softplus',
                                'rec_loc_netgrowrate' : 1.0,
                                'rec_loc_winitializers' : ['normal', 'normal', 'orthogonal'],
                                'rec_prec_numlayers' : 3,
                                'rec_prec_numnodes' : 64,
                                'rec_prec_activations' : 'softplus',
                                'rec_prec_winitializers' : ['normal', 'normal', 'orthogonal'],
                                'rec_prec_winitranges' : 0.1,
                                'rec_shareparams' : False,
                                'gen_loc_numlayers' : 3,
                                'gen_loc_numnodes' : 64,
                                'gen_loc_activations' : 'softplus',
                                'gen_loc_winitializers' : 'orthogonal',
                                'gen_loc_binitializers' : ['zeros', 'zeros', 'normal'],
                                'gen_loc_binitranges' : [None, None, 9.0],
                                'gen_wconstprecision' : True,
                                'post_usett' : False,
                                'prior_initscale' : 2.5
                                })
    this_model_dirs.update({'tr_optimizer' : 'adam',
                            'tr_lr' : 5e-4})
    this_model_dirs.update(directives)
    super(fLDS, self)._update_default_directives(**this_model_dirs)
            
  def build(self):
    """
    Build the fLDS
    """
    builder = self.builder
    ydim = self.ydim
    xdim = self.xdim
    
    rec_dirs = self.directives['rec']
    gen_dirs = self.directives['gen']
    lds_dirs = self.directives['lds']
    prior_dirs = self.directives['prior']
    post_dirs = self.directives['post']
    if builder is None:
      self.builder = builder = SequentialBuilder(scope=self._main_scope,
                                                 max_steps=self.max_steps) 
      obs = builder.addInputSequence([[ydim]],
                                     name='Observation')
      state = builder.addInputSequence([[xdim]],
                                       name='StateSeq')
      prior = builder.addInput([[xdim]],
                               NormalInputNode,
                               name='Prior',
                               **prior_dirs)
      rec = builder.addInnerSequence2([[xdim]],
                                      main_inputs=obs,
                                      node_class=NormalPrecisionNode,
                                      name='Recognition',
                                      **rec_dirs)
      lds = builder.addEvolutionwPriors([[xdim]],
                                         main_inputs=state,
                                         prior_inputs=prior,
                                         node_class=LDSEvolution,
                                         name='LDS',
                                         **lds_dirs)
      m1 = builder.addMergeSeqwDS(seq_inputs=rec,
                                  ds_inputs=lds,
                                  prior_inputs=prior, #node_list=[rec, lds, prior],
                                  merge_class=MergeSeqsNormalwNormalEv,
                                  name='Posterior',
                                  **post_dirs)
      builder.addInnerSequence2([[ydim]],
                                main_inputs=m1,
                                node_class=NormalPrecisionNode,
                                name='Generative',
                                **gen_dirs)
    else:
      self._check_custom_build()
      builder.scope = self._main_scope
    
    # build the tensorflow graph
    builder.build()
    self._build_model_outputs()
    
    self.nodes = builder.nodes
    self.otensor_names = builder.otensor_names
    self.dummies = builder.dummies
    
    cost_declare = ('elbo_flds', ('Generative', 'Posterior', 'LDS', 'Observation'))
    cost_func = self.summaries_dict[cost_declare[0]]
    self.cost = cost_func(self.nodes, cost_declare[1])
    builder.add_to_output_names('cost', self.cost)

    # trainer
    tr_dirs = self.directives['tr']
    self.trainer = fLDSTrainer(self,
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
  
  def _build_model_outputs(self):
    """
    Build model outputs
    
    TODO: Rethink this so that no call to tensorflow is needed
    """
    # build generated data
    builder = self.builder
    is2, gen = builder.nodes['StateSeq'], builder.nodes['Generative']
    state = is2.get_output_tensor('main')
    with tf.variable_scope(self.main_scope, reuse=tf.AUTO_REUSE):
      gen_loc = gen.build_loc(imain0=state)
    builder.add_to_output_names('Generative:prediction', gen_loc)
      
  def check_dataset_correctness(self, user_dataset):
    """
    Check that the provided dataset is consistent with the DeepKalmanFilter names
    """
    keys = ['train_Observation', 'valid_Observation']
    for key in keys:
      if key not in user_dataset:
        raise AttributeError("dataset must contain key `{}` ".format(key))
      
  def eval_posterior(self, dataset, key):
    """
    """
    return self.eval(dataset, 'Posterior:loc', key=key)
      
  def anal_R2(self, dataset, key='valid'):
    """
    Compute the R2 between the data produced by the Generative Model and the
    Observations
    """
    obs = dataset[key + '_Observation']
    preds = self.eval(dataset, 'Generative:loc',
                      key=key)
    R2 = compute_R2_from_sequences(obs, preds)
    return R2
  
  def anal_kR2(self, dataset,
               key='valid',
               steps=10):
    """
    Compute the k-steps ahead R2 between the data produced by the Generative
    Model and the Observations
    """
    kR2 = np.zeros(10)
    obs = dataset[key + '_Observation']
    preds = self.eval(dataset, 'Generative:loc',
                      key=key)
    R2 = compute_R2_from_sequences(obs, preds)
    kR2[0]= R2

    zpath = self.eval(dataset, 'Recognition:loc',
                      key=key)
    dataset[key + '_StateSeq'] = zpath
    for k in range(1, steps):
      zpath = self.eval(dataset, 'LDS:loc',
                        key=key)
      dataset[key + '_StateSeq'] = zpath
      preds = self.eval(dataset, 'Generative:prediction',
                        key=key)
      R2 = compute_R2_from_sequences(obs[:,k:], preds[:,:-k])
      kR2[k] = R2
    
    return kR2
      
  