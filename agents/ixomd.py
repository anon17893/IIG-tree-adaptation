import numpy as np
from open_spiel.python import policy
import pyspiel
from numba import njit

from agents.omd import OMDBase
from agents.utils import sample_from_weights

from open_spiel.python.algorithms.exploitability import nash_conv
#from torch.utils.tensorboard import SummaryWriter

class IXOMD(OMDBase):

  def __init__(
    self,
    game,
    budget,
    base_constant=1.0,
    lr_constant=1.0,
    lr_pow_H=0.0,
    lr_pow_A=0.0,
    lr_pow_X=0.0,
    lr_pow_T=-0.5,
    ix_constant=1.0,
    ix_pow_H=0.0,
    ix_pow_A=0.0,
    ix_pow_X=0.0,
    ix_pow_T=-0.5,
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
    
    self.name = 'IXOMD'
    if name:
      self.name = name
      
    
    #Set rates
    self.learning_rates = self.base_learning_rate * np.ones(self.policy_shape[0])
    self.implicit_explorations = self.base_implicit_exploration*np.ones(self.policy_shape)
    
    #Set policy
    self.current_policy.action_probability_array=self.uniform_policy
    self.current_logit=np.log(self.current_policy.action_probability_array,where=self.legal_actions_indicator)

  


  
      