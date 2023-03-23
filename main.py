import networkx as nx
import random
from draw import save_animation
from moran import Moran, Type
from stats import proportion_of_mutants, is_absorbed
from decimal import Decimal
import numpy as np
from typing import List, Set, Tuple, Dict
import copy
from tqdm import tqdm

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

N = 100
TRIALS = 1
TRIALS_PER_GRAPH = 1_000
MAX_NUMBER_STEPS = 100_000_000

RNG = random.Random()

def random_node_label(n: int) -> int:
  return RNG.randint(0, n-1)

def initialize_types(G: nx.Graph) -> None:
  n = len(G)
  for idx in range(n):
    G.nodes()[idx][Moran.TYPE_ATTRIBUTE_NAME] = Type.WILD

  G.nodes()[random_node_label(n)][Moran.TYPE_ATTRIBUTE_NAME] = Type.MUTANT


def generate_undirected_gnp_graph(n: int, p: Decimal = 1/2) -> nx.Graph:
  G = nx.DiGraph()
  while not nx.is_connected(G := nx.gnp_random_graph(n=n, p=p, seed=RNG.getstate())):
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
      RNG.choices(
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
  R = 10
  avg_steps = []
  avg_balances = []
  ns = [int(x) for x in np.linspace(start=10, stop=N, num=5)]
  print(ns)
  for n in ns:
    all_steps = []
    all_balances = []
    slow_outlier, slow_outlier_step, slow_outlier_rng = None, 0, None
    for trial in tqdm(range(1, TRIALS+1), initial=1):
      # G = generate_directed_gnp_graph(n=n)
      G = generate_eulerian_graph(n, rho=1/10)
      initial_graph = G.copy()
      rng = random.Random()
      steps_for_G = []
      for _ in tqdm(range(TRIALS_PER_GRAPH), f'particular graph #{trial}', initial=1):
        original_graph = G.copy()
        population = Moran(graph=original_graph, r=R, rng=rng)
        steps_for_G.append(num_steps_til_absorption(population))

      # avg steps
      steps = np.mean(steps_for_G)
      all_steps.append(steps)

      if steps > slow_outlier_step:
        slow_outlier = initial_graph
        slow_outlier_step = steps
        slow_outlier_rng = rng

    avg_step = np.mean(all_steps)
    std_step = np.std(all_steps)
    avg_steps.append(avg_steps)

    # avg_balance = np.mean(all_balances)
    # std_balance = np.std(all_balances)
    # avg_balances.append(avg_balance)

    print(f"{avg_step=}, {std_step=}, {slow_outlier_step=}")
    # print(f"{avg_balance=}, {std_balance=}, {slow_outlier_balance=}")

    outliers = {
      'slow': Moran(graph=slow_outlier, r=R, rng=slow_outlier_rng),
    }
    for name, example in outliers.items():
      save_animation(population=copy.deepcopy(example), out_file=f'{name}.gif')
      input()
      nx.draw(
        G=example.graph,
        # pos=nx.spectral_layout(example),
        pos=nx.kamada_kawai_layout(example.graph),
        node_color=[['blue', 'red'][data['type'].value] for _, data in example.graph.nodes(data=True)],
        with_labels=True, 
        font_size=6,
        labels={
          # node: f"{example.in_degree(node)} --> {node} --> {slow_outlier.out_degree(node)}" 
          node: f"d{example.graph.in_degree(node)}" 
          # node: f"{node}"
          for node in example.graph.nodes
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