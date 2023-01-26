import os
import importlib
from datetime import datetime
from collections import defaultdict
from unicodedata import name


import yaml
import pickle

import random
import math
import numpy as np

import pyspiel


import ray

def sample_from_weights(population, weights):
  '''Sample an element from population
   with probs proportional to weights
  '''
  return random.choices(population, weights=weights)[0]

#Return log(sum_x(exp(x)))
def compute_log_sum_from_logit(logit,mask):
  logit_max=logit.max(initial=-np.inf,where=mask)
  return math.log(np.sum(np.exp(logit-logit_max,where=mask),where=mask))+logit_max

def get_class(class_name):
  '''Import class from its name
  '''
  module_names = class_name.split('.')
  module = ".".join(module_names[:-1])          
  return getattr(importlib.import_module(module), module_names[-1])



class ExperimentGenerator(object):
  def __init__(
    self,
    description,
    game_names,
    agents,
    save_path,
    global_init_kwargs=None,
    global_training_kwargs=None,
    tuning_parameters=None,
    n_simulations=4,
  ):
    #Name of the exp
    self.description = description

    #Games
    self.game_names = game_names

    #Number of simulations
    self.n_simulations = n_simulations

    #Global init kwargs
    self.global_init_kwargs = {}
    if global_init_kwargs:
        self.global_init_kwargs = global_init_kwargs

    #Training kwargs
    self.training_kwargs = {}
    if global_training_kwargs:
        self.training_kwargs = global_training_kwargs
        
    #Tuning
    self.tuning_parameters = tuning_parameters
    self.tuned_rates = None
    
    #Path to save results
    self.save_path = os.path.join(save_path, description)


    #Build the agent constructors
    self.dict_agent_constructor = {}
    self.dict_agent_kwargs = {}
    self.agent_names = []
    for agent_config_path in agents:
      #Get agent config
      agent_config = yaml.load(open(agent_config_path, 'r'), Loader=yaml.FullLoader)
      #Get agent class
      agent_class_name = agent_config['agent_class']
      agent_class = get_class(agent_class_name)
      #Set agent parameter
      agent_kwargs = agent_config['init_kwargs']
      if self.global_init_kwargs:
        for key, value in self.global_init_kwargs.items():
          agent_kwargs[key] = value
      agent_name = agent_kwargs['name']
      #record name, kwargs, constructor
      self.agent_names.append(agent_name)
      self.dict_agent_kwargs[agent_name] = agent_kwargs
      self.dict_agent_constructor[agent_name] = agent_class

#   def build_agent(self, agent_name):
#       agent_kwargs = self.dict_agent_kwargs[agent_name].copy()
#       agent_kwargs['game'] =  pyspiel.load_game(self.game_name)
#       return self.dict_agent_constructor[agent_name](**agent_kwargs)

  def save_results(self, results, game_name, agent_name):
    #Buil path
    now = datetime.now().strftime("%d-%m__%H:%M")
    save_path = os.path.join(self.save_path, game_name, agent_name, now+'.pickle')
    #Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as _f:
      pickle.dump(results, _f)

  
  def load_results(self):
    #Load and return the last results
    dict_results = {}
    for game_name in self.game_names:
        dict_results[game_name]={}
        for agent_name in self.agent_names:
            save_path = os.path.join(self.save_path, game_name, agent_name)
            list_res = os.listdir(save_path)
            latest_res = max(list_res)
            save_path = os.path.join(save_path, latest_res)
            with open(save_path, 'rb') as _f:
                dict_results[game_name][agent_name] = pickle.load(_f)
    return dict_results


  def run(self):
    #Prepare tasks
    list_tasks = []
    for game_name in self.game_names:
        for agent_name in self.agent_names:
            for _ in range(self.n_simulations):
                if self.tuned_rates is None:
                    base_constant=1.0
                else:
                    base_constant=self.tuned_rates[game_name][agent_name]
                list_tasks.append([
                    self.dict_agent_constructor[agent_name],
                    self.dict_agent_kwargs[agent_name],
                    game_name,
                    base_constant,
                    self.training_kwargs
                    ])
    #Fit agents
    ray.init()
    result_ids = []
    for task in list_tasks:
       result_ids.append(fit_agent.remote(*task))
    results = ray.get(result_ids)
    ray.shutdown()
    print('Finished!')
    #Save results
    idx = 0
    for game_name in self.game_names:
        for agent_name in self.agent_names:
            final_results = defaultdict(list)
            for _ in range(self.n_simulations):
                res = results[idx]
                for key, value in res.items():
                    final_results[key].append(value)
                idx+=1
            for key in final_results.keys():
                if key == 'step':
                    final_results[key] = final_results[key][0]
                else:
                    final_results[key] = np.array(final_results[key])
            self.save_results(final_results, game_name, agent_name)
            
  def tune_rates(self):
    #Compute the bae_constant that will be tested
    lowest_multiplier=self.tuning_parameters['lowest_multiplier']
    highest_multiplier=self.tuning_parameters['highest_multiplier']
    size_grid_search=self.tuning_parameters['size_grid_search']
    log_step=(math.log(highest_multiplier)-math.log(lowest_multiplier))/(size_grid_search-1)
    base_constants=[lowest_multiplier*math.exp(i*log_step) for i in range(size_grid_search)]
    #Prepare tasks
    tuning_kwargs=self.training_kwargs.copy()
    tuning_kwargs['record_exploitabilities']=True
    tuning_kwargs['number_points']=None
    tuning_kwargs['log_interval']=self.global_init_kwargs['budget']
    tuning_kwargs['record_current']=False
    list_tasks = []
    for game_name in self.game_names:
        for agent_name in self.agent_names:
            for base_constant in base_constants:
                list_tasks.append([
                    self.dict_agent_constructor[agent_name],
                    self.dict_agent_kwargs[agent_name],
                    game_name,
                    base_constant,
                    tuning_kwargs
                    ])
    #Fit agents
    ray.init()
    result_ids = []
    for task in list_tasks:
       result_ids.append(fit_agent.remote(*task))
    results = ray.get(result_ids)
    ray.shutdown()
    print("Finished tuning!")
    #Save results
    idx = 0
    self.tuned_rates={}
    for game_name in self.game_names:
        self.tuned_rates[game_name]={}
        for agent_name in self.agent_names:
            for base_constant in base_constants:
                gap = results[idx].get('average')[0]
                if self.tuned_rates.get(game_name).get(agent_name) is None or best_gap > gap:
                    best_gap = gap
                    self.tuned_rates[game_name][agent_name]=base_constant
                idx+=1
    print("Best multipliers:")
    print(self.tuned_rates)

#I was out of memory, so I limited the number of workers
#@ray.remote(num_cpus=6)
@ray.remote
def fit_agent(agent_contstructor, agent_kwargs, game_name, base_constant, training_kwargs):
  agent_kwargs['game'] =  pyspiel.load_game(game_name)
  agent_kwargs['base_constant'] = base_constant
  agent = agent_contstructor(**agent_kwargs) 
  print(f'Train {agent.name} on {game_name}')
  return agent.fit(**training_kwargs)











