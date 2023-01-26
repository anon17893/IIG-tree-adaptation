import numpy as np
from collections import defaultdict
from open_spiel.python import policy
import pyspiel
#from numba import njit

from agents.omd import OMDBase
from agents.utils import sample_from_weights
from agents.ixomd import IXOMD

from open_spiel.python.algorithms.exploitability import nash_conv
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class BalancedFTRL(OMDBase):
  """A class for Balanced FTRL algorithm,
  -base leaning rate is 
      lr_base = H**lr_pow_H* A**lr_pow_A * X**lr_pow_X * T**lr_pow_T
  -base implicit exploration is 
      ix_base = H**ix_pow_H* A**ix_pow_A * X**ix_pow_X * T**ix_pow_T
  -adaptive learning rate is
      lr = lr_base / ( total_actions_from_key )**lr_pow_bal
  -adaptive implicit exploration is 
      ix = ix_base / ( balanced_plan )**ix_pow_bal
   """

  def __init__(
    self,
    game,
    budget,
    base_constant=1.0,
    lr_constant=1.0,
    lr_pow_H=-0.5,
    lr_pow_A=-0.5,
    lr_pow_X=-0.5,
    lr_pow_T=-0.5,
    lr_pow_bal=-1.0,
    ix_constant=1.0,
    ix_pow_H=-0.5,
    ix_pow_A=-0.5,
    ix_pow_X=-0.5,
    ix_pow_T=-0.5,
    ix_pow_bal=-1.0,
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

    self.name = 'BalancedFTRL'
    if name:
      self.name = name

    #Balanced transitions and policy
    self.compute_balanced()
    
    #Set rates
    self.learning_rates = self.base_learning_rate * np.ones(self.policy_shape[0])
    self.learning_rates *= self.balanced_transition_plan**lr_pow_bal
    self.implicit_explorations = self.base_implicit_exploration*np.ones(self.policy_shape)
    self.implicit_explorations *=(self.balanced_transition_plan[:,np.newaxis])**ix_pow_bal

    
    self.current_policy.action_probability_array=self.balanced_policy
    self.current_logit=np.log(self.current_policy.action_probability_array,where=self.legal_actions_indicator)
    

  def compute_balanced(self):
      self.initial_keys=[]
      self.depth_from_key=np.zeros(self.policy_shape[0], dtype=int)
      self.current_player_from_key = np.zeros(self.policy_shape[0], dtype=int)
      self.total_actions_from_key = np.zeros(self.policy_shape[0], dtype=int)
      self.total_actions_from_action = np.zeros(self.policy_shape)
      self.legal_actions_from_key = [[] for i in range(self.policy_shape[0])]
      self.balanced_policy=np.zeros(self.policy_shape)
      self.balanced_transition_plan=np.zeros(self.policy_shape[0],dtype=float) 
      self.key_children = [defaultdict(list) for i in range(self.policy_shape[0])] #key_children gives for each state_key a dictionnary that associates to each action the list of children 
      
      self.compute_information_tree_from_state(self.game.new_initial_state(),[[],[]],[0,0])
      for initial_key in self.initial_keys:
        self.compute_balanced_policy_from_key(initial_key)
        self.balanced_transition_plan[initial_key]=1.0
        for initial_action in self.legal_actions_from_key[initial_key]:
          self.compute_balanced_transition_from_action(initial_key,initial_action,1.0)

  def compute_information_tree_from_state(self, state, trajectory, depth):
    if state.is_terminal():
        return
    if state.is_chance_node():
        for action, _ in state.chance_outcomes():
            self.compute_information_tree_from_state(state.child(action), trajectory, depth)
        return
    current_player = state.current_player()
    legal_actions = state.legal_actions(current_player)
    number_legal_actions=len(legal_actions)
    state_key = self.state_index(state)
    h=depth[current_player]
    if self.total_actions_from_key[state_key] == 0:
        self.current_player_from_key[state_key] = current_player
        self.legal_actions_from_key[state_key] = legal_actions
        self.depth_from_key[state_key]=h
        
        if len(trajectory[current_player]) == 0:
          self.initial_keys.append(state_key)
        else:
          self.key_children[trajectory[current_player][-1][0]][trajectory[current_player][-1][1]].append(state_key)
        
        self.total_actions_from_key[state_key]=number_legal_actions
        for action in legal_actions:
          self.total_actions_from_action[state_key, action]=1
        for parent_couple in trajectory[current_player]:
          self.total_actions_from_key[parent_couple[0]] += number_legal_actions
          self.total_actions_from_action[parent_couple[0],parent_couple[1]]+=number_legal_actions
          
    depth[current_player]=h+1
    for action in legal_actions:
      trajectory[current_player].append([state_key,action])
      self.compute_information_tree_from_state(state.child(action), trajectory, depth)
      trajectory[current_player].pop()
    depth[current_player]=h

  def compute_balanced_policy_from_key(self, state_key):
    for action in self.legal_actions_from_key[state_key]:
      self.balanced_policy[state_key,action]=self.total_actions_from_action[state_key,action]/self.total_actions_from_key[state_key]
      for state_key_child in self.key_children[state_key][action]:
        self.compute_balanced_policy_from_key(state_key_child)
      
  #Compute the product of the transitions
  def compute_balanced_transition_from_action(self, state_key, action, current_transition):
    #if the current action is not the last action of the episode
    if self.total_actions_from_action[state_key,action]>1:
      for state_key_child in self.key_children[state_key][action]:
        new_transition=current_transition*self.total_actions_from_key[state_key_child]/self.total_actions_from_action[state_key,action]
        self.balanced_transition_plan[state_key_child]=new_transition
        for new_action in self.legal_actions_from_key[state_key_child] :
          self.compute_balanced_transition_from_action(state_key_child,new_action,new_transition)
    
        
