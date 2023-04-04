#!/usr/bin/env python3
import functools
import heapq
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from typing import DefaultDict, Dict, Generic, List, Tuple, TypeVar

def eulerian(N):
  G = nx.DiGraph()
  G.add_nodes_from(range(N))
  for d in range(N+1):
    yield from _eulerian(N, 0, 0, G, 0, d)


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
      yield G.copy()
    return

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

def expected_absorption_time(G: nx.DiGraph, r: float):
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
  
  A = np.zeros(shape=(1+2**N, 1+2**N))
  for (curr_config, next_config), p in prob.items():
    # print(f"{curr_config} --> {next_config}: {p}")
    u = config_to_index(curr_config)
    v = config_to_index(next_config)
    A[u, v] = int(u == v) - p

  # Absorbing states
  A[0, 0] = A[-1+2**N, -1+2**N] = 1.0

  # Starting configuration.
  for i in range(N):
    A[2**N, 2**i] = -1.0/N
  A[2**N, 2**N] = 1

  # print(A)
  # print(np.sum(A, axis=1))

  b = np.ones(shape=(1+2**N,1))

  # Starting configuration needs no time.
  b[2**N] = 0

  T = np.linalg.solve(A, b)
  return T[-1][0]


Value = TypeVar("Value")
Example = TypeVar("Example")
class MaxExamples(Generic[Value, Example]):
  k: int
  heap: List[Tuple[Value, Example]]

  def __init__(self, k: int = 1) -> None:
    assert k > 0
    self.k = k
    # Min-heap
    self.heap = []

  def add(self, value: Value, example: Example) -> bool:
    """Returns true iff min value changed."""
    min_value, _ = self.heap[0] if len(self.heap) else (np.inf, None)
    if len(self.heap) == self.k:
      if min_value < value:
        _ = heapq.heappushpop(self.heap, (value, example))
    else:
      heapq.heappush(self.heap, (value, example))
    new_min_value, _ = self.heap[0]
    return new_min_value > min_value and not np.isclose(new_min_value, min_value, rtol=1e-6)

  def get(self) -> None:
    return self.heap[::-1]
      
@functools.lru_cache(maxsize=None)
def make_binary(test_config, N):
  idxs = set(test_config)
  return tuple(int(idx in idxs) for idx in range(N))


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

  nx.draw_networkx_nodes(STG, pos)
  edge_colors = []
  edge_weights = []
  edgelist = []
  seen = set()
  cmap = LinearSegmentedColormap.from_list('rg', ["r", "y", "g"]) 
  # cmap=plt.cm.Blues
  print(cmap)
  for u, v in STG.edges():
    left, right = sorted((u, v), key=get_row)
    if (left, right) in seen: continue
    outgoing = STG.number_of_edges(left, right)
    incoming = STG.number_of_edges(right, left)
    weight = incoming + outgoing
    if weight == 0: continue
    seen.add((left, right))
    edgelist.append((left, right))
    edge_weights.append(weight)
    corr = (outgoing - incoming) / weight
    edge_colors.append(corr)

  print(edge_colors)
  nx.draw_networkx_edges(STG, pos=pos, edgelist=edgelist, edge_color=edge_colors, width=edge_weights, edge_cmap=cmap, edge_vmin=-1, edge_vmax=+1, arrows=False)


def draw(G: nx.DiGraph, prefix: str) -> None:
  print(f"{prefix} {r=}, {time=}, {G.edges()=}")
  fig = plt.figure()
  fig.set_figheight(7)
  fig.set_figwidth(7)
  plt.subplot(211)  
  plt.title(f"{prefix} {N=}, {r=}, {time=:.2f} steps")
  nx.draw(G)
  plt.subplot(212)
  draw_stg(G)
  plt.show()


if __name__ == '__main__':
  N = 4
  count = 0
  K = 1
  SHOW = True
  # r -> (time, G)
  max_exp_abs_time_by_r: DefaultDict[float, MaxExamples[float, nx.DiGraph]] = defaultdict(lambda: MaxExamples(K))
  by_degree_seq = defaultdict(list)
  for G in eulerian(N):
    if (d := deg_seq(G)) not in by_degree_seq or not any(nx.is_isomorphic(G, H) for H in by_degree_seq[d]):
      by_degree_seq[d].append(G)
      for r in (100,):
        time = expected_absorption_time(G, r)
        if max_exp_abs_time_by_r[r].add(time, G) and SHOW:
          draw(G, "Update!")

      count += 1
  
  print(f"{count=}")

  if SHOW:
    for r, examples in max_exp_abs_time_by_r.items():
      for (time, G) in examples.get():
        draw(G, "Winner!")