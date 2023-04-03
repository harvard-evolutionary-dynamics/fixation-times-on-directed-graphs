#!/usr/bin/env python3
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import heapq

from collections import defaultdict
from decimal import Decimal
from typing import DefaultDict, Dict, Generic, List, Tuple, TypeVar

def eulerian(N):
  G = nx.DiGraph()
  G.add_nodes_from(range(N))
  yield from _eulerian(N, 0, 0, G, 0, N)


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
    if _is_eulerian(G) and nx.is_weakly_connected(G):
      yield G.copy()
    return

  if j == N:
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
      

if __name__ == '__main__':
  N = 5
  count = 0
  K = 1
  # r -> (time, G)
  max_exp_abs_time_by_r: DefaultDict[float, MaxExamples[float, nx.DiGraph]] = defaultdict(lambda: MaxExamples(K))
  by_degree_seq = defaultdict(list)
  for G in eulerian(N):
    if (d := deg_seq(G)) not in by_degree_seq or not any(nx.is_isomorphic(G, H) for H in by_degree_seq[d]):
      by_degree_seq[d].append(G)
      for r in (0.1,):
        time = expected_absorption_time(G, r)
        if max_exp_abs_time_by_r[r].add(time, G):
          print(f"Update! {r=}, {time=}, {G.edges()=}")
          plt.figure()  
          plt.title(f"{N=}, {r=}, {time=:.2f} steps")
          nx.draw(G)
          plt.show()

      count += 1
  
  print(f"{count=}")
  for r, examples in max_exp_abs_time_by_r.items():
    for (time, G) in examples.get():
      print(f"Winner! {r=}, {time=}, {G.edges()=}")
      plt.figure()  
      plt.title(f"Winner: {N=}, {r=}, {time=:.2f} steps")
      nx.draw(G)
      plt.show()