import networkx as nx
import random
from moran import Moran, Type
from stats import proportion_of_mutants, is_absorbed
from decimal import Decimal
import numpy as np
from typing import List, Set, Tuple, Dict

import matplotlib.pyplot as plt

N = 100
TRIALS = 10_000
MAX_NUMBER_STEPS = 100_000_000

def random_node_label(n: int) -> int:
  return random.randint(0, n-1)

def initialize_types(G: nx.Graph) -> None:
  n = len(G)
  for idx in range(n):
    G.nodes()[idx][Moran.TYPE_ATTRIBUTE_NAME] = Type.WILD

  G.nodes()[random_node_label(n)][Moran.TYPE_ATTRIBUTE_NAME] = Type.MUTANT


def generate_undirected_gnp_graph(n: int, p: Decimal = 1/2) -> nx.Graph:
  G = nx.DiGraph()
  while not nx.is_connected(G := nx.gnp_random_graph(n=n, p=p)):
    ...
    # print("Gnp was unconnected. regenerating new graph")

  # for idx in range(N-1):
  #   G.add_edge(idx, idx+1)

  initialize_types(G)
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

def generate_random_cycle_labels(n: int) -> List[Tuple[int, int]]:
  cycle: List[Tuple[int, int]] = []

  u = random_node_label(n)
  seen: Dict[int, int] = {u: 0}
  while (v := random_node_label(n)) not in seen:
    cycle.append((u, v))
    seen[v] = len(cycle)
    u = v
  cycle.append((u, v))
  return cycle[seen[v]:]

def density(G: nx.DiGraph) -> Decimal:
  N = len(G)
  MAX_EDGES = N**2
  return G.number_of_edges() / Decimal(MAX_EDGES)

def generate_eulerian_graph(n: int, rho: Decimal = 1/2) -> nx.Graph:
  """`rho` is the lower bound of how dense the graph needs to be."""
  G = nx.DiGraph()
  for idx in range(n):
    G.add_node(idx)
  while not G or not nx.is_weakly_connected(G) or density(G) < rho:
    cycle_labels: List[int] = generate_random_cycle_labels(n)
    # print(cycle_labels)
    assert (cycle_labels[0][0] ==cycle_labels[-1][-1]), cycle_labels
    for u, v in cycle_labels:
      G.add_edge(u, v)
    
  # print("=======================")
  initialize_types(G)
  return G
    


def normalized_balance(G: nx.DiGraph) -> Decimal:
  """skew of out degree - in degree"""
  return 1
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
    slow_outlier, slow_outlier_step = None, 0
    fast_outlier, fast_outlier_step = None, 9999999999999999999
    for _ in range(TRIALS):
      # G = generate_directed_gnp_graph(n=n)
      G = generate_eulerian_graph(n, rho=1/10)
      population = Moran(graph=G, r=10)
      initial_graph = G.copy()
      steps = num_steps_til_absorption(population)
      all_steps.append(steps)
      balance = normalized_balance(population.graph)
      all_balances.append(balance)
      if steps > slow_outlier_step:
        slow_outlier = initial_graph
        slow_outlier_step = steps
        slow_outlier_balance = balance
      if proportion_of_mutants(population.graph) == 1 and steps < fast_outlier_step:
        fast_outlier = initial_graph
        fast_outlier_step = steps
        fast_outlier_balance = balance

    avg_step = np.mean(all_steps)
    std_step = np.std(all_steps)
    avg_steps.append(avg_steps)

    # avg_balance = np.mean(all_balances)
    # std_balance = np.std(all_balances)
    # avg_balances.append(avg_balance)

    print(f"{avg_step=}, {std_step=}, {fast_outlier_step=}, {slow_outlier_step=}")
    # print(f"{avg_balance=}, {std_balance=}, {slow_outlier_balance=}")

    for example in (slow_outlier, fast_outlier):
      nx.draw(
        G=example,
        # pos=nx.spectral_layout(example),
        pos=nx.kamada_kawai_layout(example),
        node_color=[['blue', 'red'][data['type'].value] for _, data in example.nodes(data=True)],
        with_labels=True, 
        font_size=6,
        labels={
          # node: f"{example.in_degree(node)} --> {node} --> {slow_outlier.out_degree(node)}" 
          node: f"d{example.in_degree(node)}" 
          # node: f"{node}"
          for node in example.nodes
        },
      )
      plt.show()
      plt.clf()
      plt.cla()
      input()
    # plt.scatter([n]*len(all_steps), all_steps)

  # plt.plot(ns, avg_steps, 'x-')
  # plt.plot(ns, avg_balances, 'x-')
  # plt.show()

# print(all_steps)