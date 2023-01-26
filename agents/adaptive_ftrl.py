import numpy as np
from open_spiel.python import policy
import pyspiel
from numba import njit

from agents.omd import OMDBase
from agents.utils import sample_from_weights
from agents.ixomd import IXOMD
from agents.utils import compute_log_sum_from_logit

from open_spiel.python.algorithms.exploitability import nash_conv
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class AdaptiveFTRL(OMDBase):
  
  """A class for AdaptiveFTRL algorithm,
  -adaptive learning rate is
      lr = lr_base / ( cf_scale*cf_plan+cf_prior )**lr_pow
  -adaptive implicit exploration is 
      ix = ix_base / ( cf_scale*cf_plan+cf_prior )**ix_pow
   """

  def __init__(
    self,
    game,
    budget,
    base_constant=1.0,
    lr_constant=1.0,
    lr_pow_H=0.0,
    lr_pow_A=-0.5,
    lr_pow_X=-0.5,
    lr_pow_T=0.5,
    lr_pow=-1.0,
    ix_constant=1.0,
    ix_pow_H=0.0,
    ix_pow_A=-0.5,
    ix_pow_X=-0.5,
    ix_pow_T=0.5,
    ix_pow=-1.0,
    cf_prior=1.0,
    name=None
  ):

    OMDBase.__init__(
      self,
      game,
      budget,
      base_constant=base_constant,
      lr_constant=lr_constant,
      lr_pow_H=lr_pow_H,
      lr_pow_A=lr_pow_A,
      lr_pow_X=lr_pow_X,
      lr_pow_T=lr_pow_T,
      ix_constant=ix_constant,
      ix_pow_H=ix_pow_H,
      ix_pow_A=ix_pow_A,
      ix_pow_X=ix_pow_X,
      ix_pow_T=ix_pow_T
      )

    self.name = 'AdaptiveFTRL'
    if name:
      self.name = name

    self.cf_prior = cf_prior
    
    #Set rates
    self.lr_pow=lr_pow
    self.learning_rates = self.base_learning_rate * np.ones(self.policy_shape[0])
    self.learning_rates *= cf_prior**self.lr_pow
    self.ix_pow=ix_pow
    self.implicit_explorations = self.base_implicit_exploration*np.ones(self.policy_shape)
    self.implicit_explorations *= cf_prior**self.ix_pow
    
    #Set policies
    self.current_policy.action_probability_array=self.uniform_policy
    self.current_logit = np.log(self.current_policy.action_probability_array, where=self.legal_actions_indicator)
    
    #Set the logit of the uniform policy for the adaptive updates
    self.initial_logit=self.current_logit.copy()
    
    #Initialize cumulative transition estimators
    self.cumulative_action_ptilde=np.zeros(self.policy_shape)+cf_prior
    self.cumulative_ptilde=np.zeros(self.policy_shape[0])+cf_prior

  def update(self, trajectory):
    #Initialize values
    values =  np.zeros(self.num_players)

    for transition in reversed(trajectory):
        player, state_idx, action_idx, plan, loss = transition.values()
        policy = self.current_policy.action_probability_array[state_idx,:]
        lr = self.learning_rates[state_idx]
        
        #Compute ix loss
        ix = self.implicit_explorations[state_idx, action_idx]
        ix_loss = loss/(plan+ix)
        
        #Compute new rates
        number_action=self.number_actions_from_idx[state_idx]
        ptilde_action_increment=1/(plan*policy[action_idx]+ix)
        ptilde_increment=ptilde_action_increment/number_action        
        self.cumulative_action_ptilde[state_idx,action_idx]+=ptilde_action_increment
        self.cumulative_ptilde[state_idx] += ptilde_increment
        new_lr = self.base_learning_rate*self.cumulative_ptilde[state_idx]**self.lr_pow
        new_ix = self.base_implicit_exploration*self.cumulative_action_ptilde[state_idx,action_idx]**self.ix_pow
        alpha=new_lr/lr
      
        #Set new rates
        self.learning_rates[state_idx]=new_lr
        self.implicit_explorations[state_idx,action_idx]=new_ix
    
        #Compute new policy 
        legal_actions=self.legal_actions_indicator[state_idx,:]
        lr = self.learning_rates[state_idx]
        adjusted_loss = ix_loss - values[player]
        self.current_logit[state_idx,:]=alpha*self.current_logit[state_idx,:]+(1-alpha)*self.initial_logit[state_idx,:]
        self.current_logit[state_idx,action_idx]-=lr*adjusted_loss
        logz=compute_log_sum_from_logit(self.current_logit[state_idx,:],legal_actions)
        self.current_logit[state_idx,:]-=logz*legal_actions
        values[player] = logz/lr
        new_policy=np.exp(self.current_logit[state_idx,:],where=legal_actions)*legal_actions

        #Update new policy 
        self.set_current_policy(state_idx, new_policy)
