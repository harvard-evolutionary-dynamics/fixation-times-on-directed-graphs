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

def example_monotonicity_of_ft(N): 
  data = []
  example_graph = None
  Rs = list(np.linspace(start=1, stop=1.05, num=100))
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
        if graph_idx == 9 and node_idx == 0:
          example_graph = G_nx
          data.append(((graph_idx, node_idx), r, cfts[node_idx], node_idx, G, 0.5))

  df = pd.DataFrame(data, columns=["graph idx", "Relative fitness (r)", "Fixation time (t)", "node", "pepa", "color"])
  print(df)
  plot = sns.lineplot(
    # kind="line",
    # col="node",
    # col_wrap=2,
    data=df,
    x="Relative fitness (r)",
    y="Fixation time (t)",
    # hue="counterexample",
    units="graph idx",
    estimator=None,
    # palette="light:b",
    # hue="color",
    # ylim=(23.698, 23.705),
    # style="counterexample",
    # markers=True,
    #, dashes=False,
    legend=False,
  )
  plt.ylim(23.698, 23.705)

  # plot.fig.set_size_inches(15,15)
  style(plot)
  ax = plt.gca()
  handles, labels = ax.get_legend_handles_labels()
  ax.set(xlabel='Relative fitness, $r$', ylabel='Fixation time, $T$')

  # height = df["normalized fixation time"].max()
  # plt.plot([1, 1], [0, height], linewidth=0.5, linestyle='--')
  plt.ticklabel_format(useOffset=False)
  plt.savefig(f'charts/r-vs-ft-{N}.png', dpi=300, bbox_inches="tight")

  # plt.title(f"r vs expected fixation time, {N=}")
  # plt.xlabel("r")
  # plt.ylabel("expected fixation time")
  # for rs, fts in zip(rss, ftss):
  #   plt.plot(rs, fts, marker='o')
  # plt.show()
  plt.figure()
  nx.draw(
    example_graph,
    pos=nx.kamada_kawai_layout(example_graph),
    with_labels=False,
    connectionstyle="arc3,rad=0.1",
    node_color=["red"] + ["blue"] * 3,
  )
  plt.savefig(f'charts/example-graph-r-vs-ft-{N}.png', dpi=300, transparent=True, bbox_inches="tight")

if __name__ == '__main__':
  example_monotonicity_of_ft(4)