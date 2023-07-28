#!/usr/bin/env python3
import fractions
import functools
import heapq
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import operator

from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import DefaultDict, Dict, Generic, Iterable, List, Tuple, TypeVar, Set, Optional

from utils import yield_all_digraph6

from utils import networkx_to_pepa_format
from pepa import g2degs, g2mat, cftime

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

sns.set_theme(font_scale=2, rc={'text.usetex' : True})
sns.set_style("whitegrid", {'axes.grid' : False})


from utils import style, sample, trial_cftime

def monotonicity_of_ft(N):
  data = []
  example_graph = None
  Rs = list(np.linspace(start=1, stop=1.30, num=10))
  print(len(Rs))
  for graph_idx, G_nx in enumerate(yield_all_digraph6(Path(f"data/directed/direct{N}.d6"))):
    if not nx.is_strongly_connected(G_nx): continue
    G = networkx_to_pepa_format(G_nx)
    degs = g2degs(G)
    accs = []
    mat_1 = g2mat(G, degs, 1)
    cfts_1 = cftime(mat_1)
    for r in Rs:
      mat = g2mat(G, degs, r)
      cfts = cftime(mat)
      for node_idx in range(N):
        data.append(((graph_idx, node_idx), r, cfts[node_idx] / cfts_1[node_idx], node_idx, G, 0.5))

  # Find extreme values.
  extremes = (
    max(d[2] for d in data if d[1] == Rs[-1]),
    min(d[2] for d in data if d[1] == Rs[-1])
  )

  up = set()
  for d in data:
    if d[0] in up: continue
    if d[2] <= 1: continue
    up.add(d[0])
  
  print(f"num trajectories that increase at some point: {len(up)}")

  extreme_graph_idxs = set()
  for i in range(len(data)):
    if (r := data[i][1]) != Rs[-1]: continue
    normalized_ft = data[i][2]
    if normalized_ft not in extremes: continue
    graph_idx = data[i][0]
    extreme_graph_idxs.add(graph_idx)
    print(graph_idx, data[i][-2], normalized_ft, data[i][-3])
  
  for i in range(len(data)):
    graph_idx = data[i][0]
    if graph_idx not in extreme_graph_idxs: continue
    data[i] = data[i][:-1] + (1.0,)

  df = pd.DataFrame(data, columns=["graph idx", "Relative fitness (r)", "Normalized fixation time (t)", "node", "pepa", "color"])
  print(df)
  plot = sns.lineplot(
    # kind="line",
    # col="node",
    # col_wrap=2,
    data=df,
    x="Relative fitness (r)",
    y="Normalized fixation time (t)",
    # hue="counterexample",
    units="graph idx",
    estimator=None,
    palette="light:b",
    hue="color",
    # facet_kws={"ylim": (23.698, 23.705)},
    # style="counterexample",
    # markers=True,
    #, dashes=False,
    legend=False,
  )

  style(plot)
  ax = plt.gca()
  handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles, ['$r = 1.1$', '$r = 100$'], title='')
  ax.set(xlabel='Relative fitness, $r$', ylabel='Normalized fixation time, $T$')

  # height = df["normalized fixation time"].max()
  # plt.plot([1, 1], [0, height], linewidth=0.5, linestyle='--')
  plt.ticklabel_format(useOffset=False)
  plt.savefig(f'charts/normalized-r-vs-ft-{N}.png', dpi=300, bbox_inches="tight")
  plt.show()


if __name__ == '__main__':
  monotonicity_of_ft(4)