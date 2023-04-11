#!/usr/bin/env python3
import functools
import heapq
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import DefaultDict, Dict, Generic, List, Tuple, TypeVar, Set, Optional

from utils import yield_all_digraph6

def eulerian(N):
  G = nx.DiGraph()
  G.add_nodes_from(range(N))
  yield from _eulerian(N, 0, 0, G, 0, N)

  # For regular graphs
  # for d in range(N+1):
  #   yield from _eulerian(N, 0, 0, G, 0, d)


def _is_eulerian(G: nx.DiGraph) -> bool:
  return all(
    G.in_degree(node) == G.out_degree(node)
    for node in G.nodes()
  )

def _num_unvisited_edges(N, i, j):
  return N**2 - N*i + j

def _eulerian(N, i, j, G: nx.DiGraph, acc_deg, last_deg):
  # Short circuit: We need our graph to be connected.
  if G.number_of_edges() + _num_unvisited_edges(N, i, j) < N:
    return 

  if i == N and j == 0:
    if _is_eulerian(G) and nx.is_weakly_connected(G): # and nx.is_regular(G):
    # if nx.is_strongly_connected(G): # and nx.is_regular(G):
      yield G.copy()
    return

  # Don't add self loops.
  # if i == j:
  #   yield from _eulerian(N, i, j+1, G, acc_deg, last_deg)
  #   return

  if j == N:
    # # Regularity check.
    # if last_deg == acc_deg:
    yield from _eulerian(N, i+1, 0, G, 0, acc_deg)
    return

  # Don't add edge.
  yield from _eulerian(N, i, j+1, G, acc_deg, last_deg)

  # Add edge.
  if acc_deg < last_deg:
    G.add_edge(i, j)
    yield from _eulerian(N, i, j+1, G, acc_deg+1, last_deg)
    G.remove_edge(i, j)


def deg_seq(G: nx.DiGraph):
  return tuple(
    (G.in_degree(node), G.out_degree(node))
    for node in G.nodes()
  )

def expected_absorption_time_systems(G: nx.DiGraph, r: float):
  N = len(G)
  config_to_index = lambda config: sum(b*2**i for b, i in zip(config, range(N)))
  prob: DefaultDict[Tuple[Tuple[int], Tuple[int]], float] = defaultdict(float)
  for config in itertools.product((0, 1), repeat=N):
    prob[config, config] = 0.0
  for u in G.nodes():
    for (_, v) in G.out_edges(u):
      for config in itertools.product((0, 1), repeat=N):
        k = sum(config)
        Wk = N + (r-1) * k
        prob_u_picked = ((r-1) * config[u] + 1) / Wk
        prob_u_to_v_picked = 1.0 / G.out_degree(u)
        next_config = config[:v] + (config[u],) + config[v+1:]

        prob[(config, next_config)] += prob_u_picked * prob_u_to_v_picked
  
  A = np.zeros(shape=(2**N, 2**N))
  for (curr_config, next_config), p in prob.items():
    # print(f"{curr_config} --> {next_config}: {p}")
    u = config_to_index(curr_config)
    v = config_to_index(next_config)
    A[u, v] = int(u == v) - p

  # Absorbing states
  A[0, 0] = A[-1+2**N, -1+2**N] = 1.0

  b = np.ones(shape=(2**N,))

  return (A, b)

def extreme_expected_absorption_times(G: nx.DiGraph, r: float, k: int = 1, keep_idx=lambda _: True):
  A, b = expected_absorption_time_systems(G, r)
  T = np.linalg.solve(A, b)
  idxs = [idx for idx in range(len(T)) if keep_idx(idx)]
  print(idxs)
  min_idxs = argmin_k(T, k, idxs)
  max_idxs = argmax_k(T, k, idxs)
  return {
    'min': [(mutants_from_idx(min_idx), T[min_idx]) for min_idx in min_idxs],
    'max': [(mutants_from_idx(max_idx), T[max_idx]) for max_idx in max_idxs],
  }

def expected_absorption_time_single_random_mutant(G: nx.DiGraph, r: float):
  """Under uniform single mutant initialization."""
  Ap, bp = expected_absorption_time_systems(G, r)

  A = np.zeros(shape=(1+2**N, 1+2**N))
  A[:-1, :-1] += Ap
  # Starting configuration.
  for i in range(N):
    A[2**N, 2**i] = -1.0/N
  A[2**N, 2**N] = 1

  b = np.zeros(shape=(1+2**N,))
  b[:-1] += bp
  # Starting configuration needs no time.
  b[2**N] = 0

  # Solve it!
  T = np.linalg.solve(A, b)

  # Return expected absorption time from the initial state.
  return T[-1]


Value = TypeVar("Value")
Example = TypeVar("Example")
class MaxExamples(Generic[Value, Example]):
  k: int
  heap: List[Tuple[Value, float, Example]]

  def __init__(self, k: int = 1) -> None:
    assert k > 0
    self.k = k
    # Min-heap
    self.heap = []

  def add(self, value: Value, example: Example) -> bool:
    """Returns true iff min value changed."""
    min_value, _, _ = self.heap[0] if len(self.heap) else (-np.inf, random.random(), None)
    if len(self.heap) == self.k:
      if min_value < value:
        _ = heapq.heappushpop(self.heap, (value, random.random(), example))
    else:
      heapq.heappush(self.heap, (value, random.random(), example))
    new_min_value, _, _ = self.heap[0]
    # print(new_min_value, min_value, np.isclose(new_min_value, min_value, rtol=1e-6))
    return new_min_value > min_value # and not np.isclose(new_min_value, min_value, rtol=1e-6)

  def get(self) -> None:
    return self.heap[::-1]
      
@functools.lru_cache(maxsize=None)
def make_binary(test_config, N):
  idxs = set(test_config)
  return tuple(int(idx in idxs) for idx in range(N))

@functools.lru_cache(maxsize=None)
def mutants_from_idx(idx):
  S = set()
  for loc, bit in enumerate(bin(idx)[2:][::-1]):
    if int(bit):
      S.add(loc)
  return S


@functools.lru_cache(maxsize=None)
def get_col(config) -> int:
  """TODO: optimize this"""
  N = len(config)
  r = sum(config)
  for idx, test_config in enumerate(itertools.combinations(range(N), r=r)):
    if make_binary(test_config, N) == config:
      return idx
  raise ValueError("no idx found")
  
@functools.lru_cache(maxsize=None)
def get_row(config) -> int:
  return sum(config)


def draw_stg(G: nx.DiGraph) -> None:
  """Draw the state transition graph"""
  N = len(G)
  STG = nx.MultiDiGraph()
  V = list(itertools.product((0, 1), repeat=N))
  STG.add_nodes_from(V)

  for u in G.nodes():
    for (_, v) in G.out_edges(u):
      for config in V:
        next_config = config[:v] + (config[u],) + config[v+1:]
        STG.add_edge(config, next_config)

  pos: Dict[Tuple[int], Tuple[int, int]] = {}
  for config in V:
    pos[config] = (get_row(config), get_col(config))

  edge_colors, node_colors = [], []
  edge_weights = []
  edgelist, nodelist = [], []
  seen = set()
  edge_cmap = LinearSegmentedColormap.from_list('rg', ["r", "y", "g"]) 
  node_cmap = plt.cm.Blues # LinearSegmentedColormap.from_list('rg', ["r", "y", "g"]) 
  for u, v in STG.edges():
    left, right = sorted((u, v), key=get_row)
    if (left, right) in seen: continue
    outgoing = STG.number_of_edges(left, right)
    incoming = STG.number_of_edges(right, left)
    weight = incoming + outgoing
    if weight == 0: continue
    seen.add((left, right))
    corr = (outgoing - incoming) / weight

    if left != right:
      edgelist.append((left, right))
      edge_weights.append(weight)
      edge_colors.append(corr)
    else:
      nodelist.append(left)
      node_colors.append(incoming)

  nx.draw_networkx_edges(STG, pos=pos, edgelist=edgelist, edge_color=edge_colors, width=edge_weights, edge_cmap=edge_cmap, edge_vmin=-1, edge_vmax=+1, arrows=False)
  nx.draw_networkx_nodes(STG, pos=pos, nodelist=nodelist, node_color=node_colors, cmap=node_cmap, vmin=0, vmax=G.number_of_edges())


def draw(
  G: nx.DiGraph,
  prefix: str,
  N: int,
  r: float,
  time: float,
  *G_draw_args,
  mutants: Optional[Set[int]] = None,
  with_stg: bool = True,
  **G_draw_kwargs,
) -> None:
  print(f"{prefix} {r=}, {time=}, {G.edges()=}")
  num_subplots = 1 + int(with_stg)
  fig = plt.figure()
  fig.set_figheight(7)
  fig.set_figwidth(7)
  plt.subplot(num_subplots, 1, 1)  
  plt.title(f"{prefix} {N=}, {r=}, {time=:.2f} steps")
  node_colors = [
    'blue'
    if not mutants or node not in mutants
    else 'red'
    for node in G.nodes()
  ]
  nx.draw(G, node_color=node_colors, with_labels=True, *G_draw_args, **G_draw_kwargs)
  if with_stg:
    plt.subplot(num_subplots, 1, 2)
    draw_stg(G)

  plt.show()

def generate_eulerians(N):
  by_degree_seq = defaultdict(list)
  for G in eulerian(N):
    if (d := deg_seq(G)) not in by_degree_seq or not any(nx.is_isomorphic(G, H) for H in by_degree_seq[d]):
      by_degree_seq[d].append(G)
      yield G


def tournament(eulerian_generator):
  count = 0
  K = 1
  SHOW = True
  INTERACTIVE = True
  # r -> (time, G)
  max_exp_abs_time_by_r: DefaultDict[float, MaxExamples[float, nx.DiGraph]] = defaultdict(lambda: MaxExamples(K))
  for G in eulerian_generator:
      for r in (200,):
        time = expected_absorption_time_single_random_mutant(G, r)
        if max_exp_abs_time_by_r[r].add(time, G) and SHOW and INTERACTIVE:
          draw(G, "Update!", len(G), r, time)

      count += 1
  
  print(f"{count=}")

  if SHOW:
    for r, examples in max_exp_abs_time_by_r.items():
      for (time, _, G) in examples.get():
        draw(G, "Winner!", N, r, time)

def argmin_k(a: np.array, k: int = 1, idxs = None):
  if idxs is None:
    idxs = range(len(a))
  return heapq.nsmallest(k, idxs, a.take)

def argmax_k(a: np.array, k: int = 1, idxs = None):
  if idxs is None:
    idxs = range(len(a))
  return heapq.nlargest(k, idxs, a.take)


def is_idx_lowest_circular_symmetry(idx, N):
  bits = tuple(int(bit) for bit in format(idx, f'0{N}b')[::-1])
  temp_bits = bits
  for _ in range(N):
    # circular shift left
    temp_bits = temp_bits[1:] + temp_bits[:1]
    if bits > temp_bits:
      return False
  return True

def num_mutants_eq(idx, k):
  bits = tuple(int(bit) for bit in format(idx, f'0{N}b')[::-1])
  return sum(bits) == k

if __name__ == '__main__':
  N = 7
  tournament(yield_all_digraph6(Path(f"data/euler{N}.d6")))
  # tournament(generate_eulerians(N))

def custom_graph_absorptions():
  N = 8
  R = 0.001

  # Directed cycle.
  G: nx.DiGraph = nx.DiGraph()
  for idx in range(N):
    G.add_edge(idx, (idx+1) % N)
    G.add_edge((idx+1) % N, idx)

  extremes = extreme_expected_absorption_times(
    G,
    r=R,
    k=30,
    keep_idx=lambda idx: is_idx_lowest_circular_symmetry(idx, N) and num_mutants_eq(idx, N // 2),
  )

  for extreme in ('max', 'min')[::+1]:
    for rank, (mutants, time) in enumerate(extremes[extreme], start=1):
      draw(G, f"{extreme} {rank=}", N, R, time=time, mutants=mutants, with_stg=False, pos=nx.circular_layout(G))