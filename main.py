import networkx as nx
import random
from draw import save_animation
from moran import Moran, Type
from stats import proportion_of_mutants, is_absorbed, is_fixated
from decimal import Decimal
import numpy as np
from typing import List, Set, Tuple, Dict, Optional
import copy
from tqdm import tqdm
import math
from halo import Halo

import matplotlib
#matplotlib.use("tkagg")
import matplotlib.pyplot as plt

N = 10
TRIALS = 10_000
TRIALS_PER_GRAPH = 1
MAX_NUMBER_STEPS = 10_000_000_000_000_000
MAX_LOOPS_WITHOUT_UPDATE = 100

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

def generate_random_cycle_labels(G: nx.DiGraph, n: int, num_edges_to_add: int) -> List[Tuple[int, int]]:
  # if num_edges_to_add == 0: return []

  first_node = u = random_node_label(n)
  v = None
  # if num_edges_to_add == 1:
  #   return [(u, u)]

  cycle: List[Tuple[int, int]] = []
  while True:
    possible_choices = list(set(range(len(G))) - set([out for _, out in G.out_edges(u)]))
    v = RNG.choice(possible_choices)
    G.add_edge(u, v)
    cycle.append((u, v))

    if v == first_node: break
    u = v
  return cycle

def density(G: nx.DiGraph) -> Decimal:
  N = len(G)
  MAX_EDGES = N**2
  return G.number_of_edges() / Decimal(MAX_EDGES)

def generate_eulerian_graph(n: int, rho: Decimal = 1/2) -> Optional[nx.Graph]:
  """`rho` is the lower bound of how dense the graph needs to be."""
  G = nx.DiGraph()
  for idx in range(n):
    G.add_node(idx)
  while not G or not nx.is_weakly_connected(G) or density(G) < rho:
    num_edges_to_add = int(math.ceil(len(G)**2 * rho - G.number_of_edges())) 
    # print(num_edges_to_add)
    if num_edges_to_add <= 0:
      break
    # assert num_edges_to_add >= 0, (G.number_of_edges(), rho, len(G))
    cycle_labels: List[int] = generate_random_cycle_labels(G, n, num_edges_to_add)
    if len(cycle_labels) == 0:
      return None
    # print(cycle_labels)
    assert (cycle_labels[0][0] == cycle_labels[-1][-1]), cycle_labels
    # for u, v in cycle_labels:
    #   G.add_edge(u, v)
    
  # print("=======================")
  for node in G.nodes():
    assert G.in_degree(node) == G.out_degree(node), f"{node=}, {G.in_degree(node)=} != {G.out_degree(node)=}"
  initialize_types(G)
  return G
    


def num_steps_til_fixation(population: Moran) -> int:
  # print(f"mutant level: {proportion_of_mutants(population.graph)}")
  for step, _ in enumerate(population.evolve(steps=MAX_NUMBER_STEPS)):
    # print(f"mutant level {step=}: {proportion_of_mutants(population.graph)}")
    if is_fixated(population.graph): return step
    if is_absorbed(population.graph):
      # print(f"did not fixate after '{step=}'")
      # print(population.graph.edges(data=True))
      # input()
      raise ValueError(f"did not fixate after '{step=}'")
  else:
    # print(f"did not fixate after '{MAX_NUMBER_STEPS=}'")
    # print(population.graph.edges(data=True))
    raise ValueError(f"did not fixate after '{MAX_NUMBER_STEPS=}'")


def num_steps_til_absorption(population: Moran) -> int:
  # print(f"mutant level: {proportion_of_mutants(population.graph)}")
  for step, _ in enumerate(population.evolve(steps=MAX_NUMBER_STEPS)):
    # print(f"mutant level {step=}: {proportion_of_mutants(population.graph)}")
    if is_absorbed(population.graph): return step
  else:
    raise ValueError(f"did not absorb after '{MAX_NUMBER_STEPS}'")


def generate_cycles(n: int):
  # Directed cycle.
  G: nx.DiGraph = nx.DiGraph()
  for idx in range(n):
    G.add_edge(idx, (idx+1) % n)

  Np = 6
  for idx in range(Np):
    G.add_edge(idx+(n-1), (n-1) + (idx+1) % Np)

  return G

if __name__ == '__main__':
  R = 1
  TRIALS = 1000
  ns = list(range(1, 21))
  all_steps = []
  avg_steps = []
  for n in ns:
    steps = []
    for _ in tqdm(range(TRIALS)):
      G = generate_cycles(n)
      initialize_types(G)
      population = Moran(graph=G, r=R)
      steps.append(num_steps_til_absorption(population))

    avg_step = np.mean(steps)
    std_step = np.std(steps)
    avg_steps.append(avg_step)

    print(f"{n=} {avg_step=}, {std_step=}")
    all_steps.append(steps)
  # plt.boxplot(all_steps, positions=ns)
  plt.plot(ns, avg_steps, marker="o")


  plt.show()
