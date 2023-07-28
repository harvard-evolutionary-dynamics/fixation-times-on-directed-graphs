import matplotlib.pyplot as plt
import networkx as nx
import random
import seaborn as sns

from typing import Set, Optional

sns.set_theme(font_scale=2, rc={'text.usetex' : True})
sns.set_style("whitegrid", {'axes.grid' : False})

def sample(fn, times):
  count = 0
  while count < times:
    if (ans := fn()) is not None:
      yield ans
      count += 1

def trial_cftime(G: nx.DiGraph, S: Optional[Set], r: float):
  if S is None:
    S = {random.choice(list(G.nodes()))}

  N = len(G)
  V = G.nodes()
  mutants = set()
  mutants |= S
  steps = 0

  while V - mutants:
    if not mutants: return None
    k = len(mutants)
    if random.random() < r*k/(N + (r-1)*k):
      birther = random.choice(list(mutants))
    else:
      birther = random.choice(list(V - mutants))

    dier = random.choice([w for (_, w) in G.out_edges(birther)])
    assert birther != dier
    if birther in mutants:
      mutants.add(dier)
    elif dier in mutants:
      mutants.remove(dier)
    
    steps += 1
  return steps

def style(plot):
  fig = plt.gcf()
  # fig.patch.set_alpha(0)
  # Add a border around the plot
  # ax = plt.gca()
  for i, ax in enumerate(fig.get_axes()):
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Customize the border color and thickness
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.grid(False)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, [['$r = 1.1$', '$r = 2$'][i]], title='')

  # Remove legend title.
  # handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles=handles[1:], labels=labels[1:])