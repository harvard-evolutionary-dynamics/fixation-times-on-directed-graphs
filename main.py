import networkx as nx
import random
from moran import Moran, Type
from stats import proportion_of_mutants, is_absorbed
from decimal import Decimal
import numpy as np

import matplotlib.pyplot as plt

N = 100
TRIALS = 1_000
MAX_NUMBER_STEPS = 100_000_000

def generate_undirected_gnp_graph(n: int, p: Decimal = 1/2) -> nx.Graph:
  G = nx.DiGraph()
  while not nx.is_connected(G := nx.gnp_random_graph(n=n, p=p)):
    ...
    # print("Gnp was unconnected. regenerating new graph")

  # for idx in range(N-1):
  #   G.add_edge(idx, idx+1)

  for idx in range(n):
    G.nodes()[idx][Moran.TYPE_ATTRIBUTE_NAME] = Type.WILD

  G.nodes()[random.randint(0, n-1)][Moran.TYPE_ATTRIBUTE_NAME] = Type.MUTANT

  return G

def generate_directed_gnp_graph(n: int, p: Decimal = 1/2) -> nx.Graph:
  while True:
    G = generate_undirected_gnp_graph(n, p=1-(1-p)**2)
    edges = list(G.edges())
    G = G.to_directed()
    for u, v in edges:
      random.choices(
        population=[
          lambda: ...,
          lambda: G.remove_edge(v, u),
          lambda: G.remove_edge(u, v),
        ],
        weights=[p*p, p*(1-p), (1-p)*p],
      )[0]()

    if nx.is_weakly_connected(G):
      break

  return G


def normalized_balance(G: nx.DiGraph) -> Decimal:
  """skew of out degree - in degree"""
  # print(G.out_degree(0), G.in_degree(0), G.nodes)
  std = Decimal(np.std([G.out_degree(node) - G.in_degree(node) for node in G.nodes]))
  return sum(
    ((G.out_degree(node) - G.in_degree(node)) / std)**3
    for node in G.nodes
  ) / Decimal(len(G))

def num_steps_til_absorption(population: Moran) -> int:
  # print(f"mutant level: {proportion_of_mutants(population.graph)}")
  for step, _ in enumerate(population.evolve(steps=MAX_NUMBER_STEPS)):
    # print(f"mutant level {step=}: {proportion_of_mutants(population.graph)}")
    if is_absorbed(population.graph): return step
  else:
    raise ValueError(f"did not absorb after '{MAX_NUMBER_STEPS}'")

if __name__ == '__main__':
  avg_steps = []
  avg_balances = []
  ns = [int(x) for x in np.linspace(start=10, stop=N, num=5)]
  print(ns)
  for n in ns:
    all_steps = []
    all_balances = []
    slow_outlier, outlier_step, outlier_balance = None, 0, 0
    for _ in range(TRIALS):
      G = generate_directed_gnp_graph(n=n)
      population = Moran(graph=G, r=1.5)
      initial_graph = G.copy()
      steps = num_steps_til_absorption(population)
      all_steps.append(steps)
      balance = normalized_balance(population.graph)
      all_balances.append(balance)
      if steps > outlier_step:
        slow_outlier = initial_graph
        outlier_step = steps
        outlier_balance = balance

    avg_step = np.mean(all_steps)
    std_step = np.std(all_steps)
    avg_steps.append(avg_steps)

    avg_balance = np.mean(all_balances)
    std_balance = np.std(all_balances)
    avg_balances.append(avg_balance)

    print(f"{avg_step=}, {std_step=}, {outlier_step=}")
    print(f"{avg_balance=}, {std_balance=}, {outlier_balance=}")

    nx.draw(
      G=slow_outlier,
      node_color=[['blue', 'red'][data['type'].value] for _, data in slow_outlier.nodes(data=True)],
      with_labels=True, 
      labels={
        node: f"{slow_outlier.in_degree(node)} --> {node} --> {slow_outlier.out_degree(node)}" 
        for node in slow_outlier.nodes
      },
    )
    plt.show()
    input()
    plt.scatter([n]*len(all_steps), all_steps)

  plt.plot(ns, avg_steps, 'x-')
  plt.plot(ns, avg_balances, 'x-')
  plt.show()

# print(all_steps)