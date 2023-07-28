#!/usr/bin/env python3
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from utils import style, sample, trial_cftime

sns.set_theme(font_scale=2, rc={'text.usetex' : True})
sns.set_style("whitegrid", {'axes.grid' : False})

def vortex_graph(N):
  assert N > 3
  G = nx.DiGraph()

  down = N//2
  up = down + N%2
  # print(down, up)

  for b in range(2, 1+down):
    G.add_edge(-1, -b)
    G.add_edge(-b, +1)

  for b in range(2, 1+up):
    G.add_edge(+1, +b)
    G.add_edge(+b, -1)

  plt.show()
  return G

def vortex_fixation_time(N, samples=1000, overwrite=True, use_existing_file=True):
  Ns = list(range(4, N+1))
  file_name = f"data/vortex-graph-estimated-N-vs-ft-{N}.pkl"
  Rs = (1.1, 100)
  if use_existing_file:
    df = pd.read_pickle(file_name)
  else:
    data = []
    for N in Ns:
      print(N)
      G = vortex_graph(N)
      for r in Rs: 
        for time in sample(lambda: trial_cftime(G, None, r), samples):
          data.append((N, r, time))

    df = pd.DataFrame(data, columns=["Population size", "r", "Fixation time"])
    if overwrite:
      df.to_pickle(file_name)

  df = df[df['r'].isin(Rs)]
  plot = sns.lineplot(
    # kind="line",
    data=df,
    x="Population size",
    y="Fixation time",
    hue="r",
    palette="Paired",
    # units="graph idx",
    # estimator=None,
    # facet_kws={"ylim": (23.698, 23.705)},
    # style="counterexample",
    marker="o",
    # style="logic",
    linestyle="--",
    # dashes=True,
  )

  ax = plt.gca()
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, [f'${r = }$' for r in Rs], title='')
  ax.set(xlabel='Population size, $N$', ylabel='Fixation time, $T$')

  style(plot)

  # plt.ticklabel_format(useOffset=False)
  plt.xscale('log')
  plt.yscale('log')
  from matplotlib.ticker import ScalarFormatter
  ax.xaxis.set_major_formatter(ScalarFormatter()) 
  ax.xaxis.set_minor_formatter(ScalarFormatter())
  ax.set_xticks([n for n in Ns if n != 10 and n % 10 == 0])
  plt.savefig(f'charts/vortex-graph-estimated-N-vs-ft-{N}.png', dpi=300, bbox_inches="tight")
  plt.show()

if __name__ == '__main__':
  vortex_fixation_time(20)