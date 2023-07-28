#!/usr/bin/env python3
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

sns.set_theme(font_scale=2, rc={'text.usetex' : True})
sns.set_style("whitegrid", {'axes.grid' : False})


from utils import style, sample, trial_cftime

def two_layer_graph(N):
  G = nx.DiGraph()
  layer1 = [i+1 for i in range(N+1)]
  layer2 = [i+(N+1)+1 for i in range(N+1)]
  left = 0
  right = (2*N+4)-1

  G.add_edge(left, layer1[0])
  G.add_edge(layer2[0], right)
  G.add_edge(right, layer2[0])

  G.add_edge(layer1[-1], layer2[-1])
  G.add_edge(layer2[-1], layer1[-1])
  for i in range(N):
    G.add_edge(layer2[i], layer1[i])
    G.add_edge(layer1[i], layer1[i+1])
    G.add_edge(layer1[i+1], layer1[i])
    G.add_edge(layer2[i], layer2[i+1])
    G.add_edge(layer2[i+1], layer2[i])

  return G


def two_layer_fixation_time(N, samples=1000, overwrite=True, use_existing_file=True):
  file_name = f"data/two-layer-graph-estimated-N-vs-ft-{N}.pkl"
  Ns = list(range(1, N+1))
  if use_existing_file:
    df = pd.read_pickle(file_name)
  else:
    data = []
    for N in Ns:
      print(N)
      G = two_layer_graph(N)
      for r in ((1.1,) if N <= 10 else ()) + (100,):
        for time in sample(lambda: trial_cftime(G, {0}, r), samples):
          data.append((N, r, time))

    df = pd.DataFrame(data, columns=["Population size", "r", "Fixation time"])
    if overwrite:
      df.to_pickle(file_name)

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
  ax.legend(handles, ['$r = 1.1$', '$r = 100$'], title='')
  ax.set(xlabel='Population size, $N$', ylabel='Fixation time, $T$')
       

  style(plot)
  # plt.gca().set_xticks(Ns)

  # plt.ticklabel_format(useOffset=False)
  plt.yscale('log')
  plt.savefig(f'charts/two-layer-graph-estimated-N-vs-ft-{N}.png', dpi=300, bbox_inches="tight")
  plt.show()


if __name__ == '__main__':
  two_layer_fixation_time(20, samples=1000)