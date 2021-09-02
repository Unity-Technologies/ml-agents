import numpy as np
import torch
import random

np.random.seed(1336)
torch.manual_seed(1336)

from mlagents.trainers.torch.layers import linear_layer#, LinearEncoder
from mlagents.trainers.torch.attention import ResidualSelfAttention, EntityEmbedding#, LinearEncoder
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times']
import matplotlib as mpl
mpl.rc('font', family='serif', serif='Times New Roman')
mpl.style.use('classic')
mpl.use('TkAgg')
import pandas as pd

N_INPUT = 10
_colors =["b", "r", "g", "y"]
# 0 and 1 are absorbing states
# range between 0.4 and 0.6 for values
# compute the average
# run with 1, 2 and at most 2 absorbing state

seeds = 20

absorbing_states = [0]#, 1, -1, .4, 10]
LOW = 0.25
HIGH = 0.75
LOG = True



class AttenNetwork(torch.nn.Module):
    def __init__(self):
      super(AttenNetwork, self).__init__()
      hidden_size = 32
      self.entity_enc = EntityEmbedding(1, None, hidden_size)
      self.rsa = ResidualSelfAttention(hidden_size, num_heads=4)
      self.dense3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, data, masks):
      h = self.entity_enc(None, data.unsqueeze(-1))
      h = self.rsa(h, masks)
      return self.dense3(h)

class Network(torch.nn.Module):
    def __init__(self):
      super(Network, self).__init__()
      hidden_size = 32
      self.dense1 = torch.nn.Linear(N_INPUT, hidden_size)
      self.dense2 = torch.nn.Linear(hidden_size, hidden_size)
      self.dense3 = torch.nn.Linear(hidden_size, 1)
      self.relu = torch.nn.ReLU()
    def forward(self, data, mask):
      h = self.dense1(data)
      h = self.relu(h)
      h = self.dense2(h)
      h = self.relu(h)
      return self.dense3(h)


def generate_batch(batch, max_num_abs, abs_state, sample = False, atten=False):
  inputs = np.random.uniform(LOW, HIGH, (batch, N_INPUT))
  inputs = np.float32(inputs)
  #[numpy.random.shuffle(x) for x in a]
  for b in range(batch):
    if sample:
      nnn = random.choice(range(max_num_abs))
    else:
      nnn = max_num_abs
    for i in range(nnn):
      inputs[b, i] = abs_state

  [np.random.shuffle(x) for x in inputs]
  n_abs = (inputs == abs_state).astype(int).sum(axis = 1, keepdims = True)
  masks = (inputs == abs_state).astype(float)
  masks = np.float32(masks)
  target = (np.sum(inputs, axis=1, keepdims = True) - abs_state * n_abs) / (N_INPUT - n_abs)
  return inputs, masks, target 

_c = sns.color_palette("twilight_shifted_r")
for absorbing_state in absorbing_states:

    f = plt.figure(1, figsize=(5, 3), dpi=300)
    
    font_size = 10
    y = 2
    x = 2
    sns.set_style("white")
    sns.set_style("ticks", {'font.family':'serif', 'font.serif':'Times New Roman', 'lines.linewidth': 8})
    plt.title("Calculating the Average of Floats", fontsize=font_size)
    #plt.title("Computation of Average with Absorbing State " + str(absorbing_state), fontsize=font_size)

    if LOG:
      plt.ylabel("Log Mean Squared Error", fontsize=font_size)
    else:
      plt.ylabel("Mean Squared Error", fontsize=font_size)
    plt.xlabel("Epochs", fontsize=font_size)
    
    
    np.random.seed(1336)
    torch.manual_seed(1336)
    # _c =  ["blue", "green", "yellow", "m", "c", "orange", "red", "black"]
    # _c = ["blue", "orange", "green", "m", "red", "black","c", "yellow"]
    # _c= ['#1f77b4', '#ff7f0e',  '#d62728', '#9467bd','black', '#2ca02c', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    #_c = sns.color_palette("colorblind")
    #if attention:
    #    _title = "atten_abs_state"
    #else:
    #    _title = "abs_state_" + str(absorbing_state)
    #for num_absorb, color, sample in zip([0,2,4,6, 8], _c[:5], [False, True, True, True, True]):
    print(len(_c))
    for num_absorb, color, sample, attention in zip([0,2, 4, 6, 8], _c[:5], [False, False, False, False, False], [False,  False, False, False, False]):
        if attention:
            _title = "atten_abs_state"
        else:
            _title = "fixed_abs_state_" + str(absorbing_state)

        dfs = []
        print("  num_abs: ", num_absorb)

        if attention:
          if not sample:
            condition_name = "10 inputs (Attention)"
          else:
            condition_name = str(10 - num_absorb) + "-10 inputs (Attention)" 
        elif not sample:
          if num_absorb == 0:
            condition_name = str(10 - num_absorb) + " inputs (FC)"
          else:
            condition_name = str(10 - num_absorb) + " inputs (FC, Absorbing)"
        else:
          condition_name = str(10 - num_absorb) + "-10 inputs  (FC, Absorbing)" 
        # sns.tsplot(data=df["values"] , color=color, condition=condition_name, ci=95, time=df.index.values)
        values_per_seed = pd.read_csv("abs_plot_csvs/" + _title + str(num_absorb) + ".csv")
        sns.tsplot(data=[values_per_seed["seed" + str(_s)] for _s in range(seeds)], color=color, condition=condition_name, ci=95)#, time=values_per_seed["step"])
    
    plt.legend(handlelength=2, fontsize=font_size, labelspacing=0.25, borderpad=0.25, markerscale=0.75, frameon=False)
    plt.savefig("abs_state_plots/" + "fixed_.png", dpi=300, bbox_inches='tight')
    f.clear()
    plt.close(f)
