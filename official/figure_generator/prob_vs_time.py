#!/usr/bin/env python3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pathlib import Path

from utils import yield_all_digraph6

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib.backend_bases import PickEvent

sns.set_theme(font_scale=2, rc={'text.usetex' : True})
sns.set_style("whitegrid", {'axes.grid' : False})


from utils import style, is_oriented, is_undirected


def fp_vs_cftime(N: int):
  from utils import networkx_to_pepa_format
  from pepa import g2degs, g2mat, cftime, fprob
  data = []
  Gs = []
  count = 0

  def on_pick(event: PickEvent):
    print(event)
    print(event.artist)
    idx = event.ind[0]
    H = Gs[idx]
    plt.figure()
    nx.draw(H, pos=nx.kamada_kawai_layout(H), with_labels=True, connectionstyle="arc3,rad=0.1")
    plt.show()

  Rs = (1.1, 2.)
  noise = lambda: 0 # random.normalvariate(mu=0, sigma=.0001)
  for G_nx in yield_all_digraph6(Path(f"data/directed/direct{N}.d6")):
    if not nx.is_strongly_connected(G_nx): continue
    count += 1
    G = networkx_to_pepa_format(G_nx)
    degs = g2degs(G)
    undirected = is_undirected(G_nx)
    oriented = is_oriented(G_nx)
    for r in Rs:
      Gs.append(G_nx)
      temps = [1] * N
      mat = g2mat(G, degs, r)
      cft = np.average(cftime(mat), weights=temps)
      fp = np.average(fprob(mat), weights=temps)
      type_ = 'undirected' if undirected else 'oriented' if oriented else 'directed'
      data.append((fp + noise(), cft + noise(), r, type_, G_nx.number_of_edges()))

  df = pd.DataFrame(data, columns=["Fixation probability (p)", "Fixation time (t)", "r", "type", "num edges"])

  # Change z-order.
  df = df.sort_values(by="type", key=lambda s: s.apply(lambda t: ("oriented", "undirected", "directed").index(t)), ascending=False)
  
  print(df["type"])
  # plt.scatter(fps_, cftimes, s=2)
  plot = sns.relplot(
    data=df,
    x="Fixation probability (p)",
    y="Fixation time (t)",
    # aspect="num edges",
    # markers="undirected",
    # scatter_kws={'alpha': 0.5, "s": 8, 'linewidths': 0},
    facet_kws={'sharey': True, 'sharex': False},
    hue="type",
    hue_order=["undirected", "directed", "oriented"],
    col="r",
    col_wrap=2,
    # picker=4,
    # sharex=False,
    # sharey=True,
    # fit_reg=False,
  )
  for ax in plt.gcf().get_axes():
    ax.set(xlabel='Fixation probability, $\\rho$', ylabel='Fixation time, $T$')
    # ax.figure.canvas.mpl_connect("pick_event", on_pick)

  # Define the z-order for different hues
  zorder_dict = {'oriented': 1, 'undirected': 2, 'directed': 3}

  # Iterate over the scatter plot artists
  for artist in plot.legend.legendHandles:
    # Get the hue label
    hue_label = artist.get_label()
    
    # Get the corresponding z-order value from the dictionary
    zorder = zorder_dict.get(hue_label)
    
    # Set the z-order for the artist
    artist.set_zorder(zorder)

  style(plot)
  print(f"{count=}")
  # sns.relplot(
  #   data=tips,
  #   x="total_bill", y="tip", col="time",
  #   hue="smoker", style="smoker", size="size",
  # )
  plt.savefig(f'charts/fp-vs-ft-{N}.png', dpi=300)
  plt.show()

if __name__ == '__main__':
  fp_vs_cftime(5)