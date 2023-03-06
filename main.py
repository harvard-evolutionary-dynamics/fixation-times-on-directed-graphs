import networkx as nx
import random
from moran import Moran, Type
from stats import proportion_of_mutants, is_absorbed
from decimal import Decimal
import numpy as np

import matplotlib.pyplot as plt

N = 30
TRIALS = 1_000
MAX_NUMBER_STEPS = 100_000_000

def generate_graph(n: int) -> nx.Graph:
  G = nx.DiGraph()
  while not nx.is_connected(G := nx.gnp_random_graph(n=n, p=1/2)):
    ...
    # print("Gnp was unconnected. regenerating new graph")

  # for idx in range(N-1):
  #   G.add_edge(idx, idx+1)

  for idx in range(n):
    G.nodes()[idx][Moran.TYPE_ATTRIBUTE_NAME] = Type.WILD

  G.nodes()[random.randint(0, n-1)][Moran.TYPE_ATTRIBUTE_NAME] = Type.MUTANT

  return G

def num_steps_til_absorption(population: Moran) -> int:
  # print(f"mutant level: {proportion_of_mutants(population.graph)}")
  for step, _ in enumerate(population.evolve(steps=MAX_NUMBER_STEPS)):
    # print(f"mutant level {step=}: {proportion_of_mutants(population.graph)}")
    if is_absorbed(population.graph): return step
  else:
    raise ValueError(f"did not absorb after '{MAX_NUMBER_STEPS}'")

if __name__ == '__main__':
  avgs = []
  ns = list(range(2, N+1))
  for n in ns:
    all_steps = []
    for _ in range(TRIALS):
      G = generate_graph(n=n)
      population = Moran(graph=G, r=1)
      steps = num_steps_til_absorption(population)
      all_steps.append(steps)

    avgs.append(np.mean(all_steps))
    plt.scatter([n]*len(all_steps), all_steps)

  plt.plot(ns, avgs, 'x-')
  plt.show()

# print(all_steps)