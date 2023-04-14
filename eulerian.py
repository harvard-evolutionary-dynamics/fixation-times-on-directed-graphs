#!/usr/bin/env python3
import functools
import heapq
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import operator

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
  N = len(G)
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

  def __init__(self, k: int = 1, invert: bool = False) -> None:
    assert k > 0
    self.k = k
    self.invert = invert
    self.adjust = lambda value: (1-2*int(self.invert)) * value
    self._empty_extreme = self.adjust(-np.inf)
    # Min-heap
    self.heap = []

  def add(self, value: Value, example: Example) -> bool:
    """Returns true iff min value changed."""
    min_value, _, _ = self.heap[0] if len(self.heap) else (self._empty_extreme, random.random(), None)
    # cmp = operator.gt if self.invert else operator.lt
    if len(self.heap) == self.k:
      if min_value < self.adjust(value):
        _ = heapq.heappushpop(self.heap, (self.adjust(value), random.random(), example))
    else:
      heapq.heappush(self.heap, (self.adjust(value), random.random(), example))
    new_min_value, _, _ = self.heap[0]
    # print(new_min_value, min_value, np.isclose(new_min_value, min_value, rtol=1e-6))
    # final_cmp = operator.lt if self.invert else operator.gt
    return new_min_value != min_value # and not np.isclose(new_min_value, min_value, rtol=1e-6)

  def get(self) -> None:
    return sorted([(self.adjust(time), x, y) for time, x, y in self.heap], reverse=not self.invert)
      
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
  INTERACTIVE = False
  # r -> (time, G)
  max_exp_abs_time_by_r: DefaultDict[float, MaxExamples[float, nx.DiGraph]] = defaultdict(
    lambda: MaxExamples(K, invert=True)
  )
  for G in eulerian_generator:
      for r in (1.1,):
        time = expected_absorption_time_single_random_mutant(G, r)
        if max_exp_abs_time_by_r[r].add(time, G) and SHOW and INTERACTIVE:
          draw(G, "Update!", len(G), r, time, pos=nx.circular_layout(G))

      count += 1
  
  print(f"{count=}")

  if SHOW:
    for r, examples in max_exp_abs_time_by_r.items():
      for (time, _, G) in examples.get():
        draw(G, "Winner!", N, r, time, pos=nx.circular_layout(G))
        # draw(G, "Winner!", N, r, time, pos=nx.planar_layout(G))

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

def num_mutants_eq(idx, k, N):
  bits = tuple(int(bit) for bit in format(idx, f'0{N}b')[::-1])
  return sum(bits) == k

def fan_system(B: int, r: float):
  N = 2*B+1
  F = lambda b: 1 + (r-1) * b

  def W(core, b00, b01, b10, b11):
    return F(core) + b00 * (1 + 1) + (b01 + b10) * (1 + r) + b11 * (r + r)

  def is_valid_state(core, b00, b01, b10):
    return core in (0, 1) and all((bxx in range(B+1)) for bxx in (b00, b01, b10)) and b00 + b01 + b10 <= B

  @functools.lru_cache(maxsize=None)
  def state_to_idx(core, b00, b01, b10):
    assert is_valid_state(core, b00, b01, b10), (core, b00, b01, b10)
    
    count = 0
    for pcore in (0, 1):
      for pb00 in range(B+1):
        for pb01 in range(B-pb00+1):
          for pb10 in range(B-pb00-pb01+1):
            if (core, b00, b01, b10) == (pcore, pb00, pb01, pb10):
              return count
            count += 1
    
    raise ValueError(f"no known idx for state {(core, b00, b01, b10)}")

  # def idx_to_state(idx):
  #   return tuple(
  #     (idx // ((B+1)**i)) % (B+1)
  #     for i in range(4)
  #   )[::-1]

  num_indices = 1 + state_to_idx(1, B, 0, 0)
  A = np.zeros(shape=(1+num_indices, 1+num_indices))
  for core in (0, 1):
    for b00 in range(B+1):
      for b01 in range(B-b00+1):
        for b10 in range(B-b00-b01+1):
          b11 = B-b00-b01-b10
          w = W(core, b00, b01, b10, b11)


          # S = (core, b00, b01, b10)
          # T(S) = 0

          # # Another step.
          # T(S) += 1

          # # Event that core is selected.
          # core_weight = F(core)
          # T(S) +=  core_weight / w * (
          #     (b00/B) * T(core, b00-core, b01+core, b10)
          #   + (b01/B) * T(core, b00+(1-core), b01-(1-core), b10)
          #   + (b10/B) * T(core, b00, b01, b10-core)
          #   + (b11/B) * T(core, b00, b01, b10+(1-core))
          #   )

          # # Event that first node on blade is selected.
          # first_node_weight = b00 * 1 + b01 * r + b10 * 1 + b11 * r
          # T(S) += first_node_weight / w * (
          #     b00 * 1 / first_node_weight * T(core, b00, b01, b10)
          #   + b01 * r / first_node_weight * T(core, b00, b01-1, b10)
          #   + b10 * 1 / first_node_weight * T(core, b00+1, b01, b10-1)
          #   + b11 * r / first_node_weight * T(core, b00, b01, b10)
          # )

          # # Event that second node on blade is selected (pointing to core).
          # second_node_weight = b00 * 1 + b01 * 1 + b10 * r + b11 * r
          # T(S) += second_node_weight / w * (
          #     b00 * 1 / second_node_weight * T(0, b00, b01, b10)
          #   + b01 * 1 / second_node_weight * T(0, b00, b01, b10)
          #   + b10 * r / second_node_weight * T(1, b00, b01, b10)
          #   + b11 * r / second_node_weight * T(1, b00, b01, b10)
          # )

          # state -> coefficient
          coeffs = defaultdict(float)
          coeffs[(core, b00, b01, b10)] += 1

          prob_inc = 0
          prob_dec = 0
          # non-absorbing states.
          if (core, b00, b01, b10) not in ((1, 0, 0, 0), (0, B, 0, 0)):
            core_weight = F(core)
            coeffs[(core, b00-core, b01+core, b10)] -= (core_weight / w) * (b00/B)
            coeffs[(core, b00+(1-core), b01-(1-core), b10)] -= (core_weight / w) * (b01/B)
            coeffs[(core, b00, b01, b10-core)] -= (core_weight / w) * (b10/B)
            coeffs[(core, b00, b01, b10+(1-core))] -= (core_weight / w) * (b11/B)

            first_node_weight = b00 * 1 + b01 * r + b10 * 1 + b11 * r
            coeffs[(core, b00, b01, b10)] -= (first_node_weight / w) * (b00 * 1 / first_node_weight)
            coeffs[(core, b00, b01-1, b10)] -= (first_node_weight / w) * (b01 * r / first_node_weight)
            coeffs[(core, b00+1, b01, b10-1)] -= (first_node_weight / w) * (b10 * 1 / first_node_weight)
            coeffs[(core, b00, b01, b10)] -= (first_node_weight / w) * (b11 * r / first_node_weight)

            second_node_weight = b00 * 1 + b01 * 1 + b10 * r + b11 * r
            coeffs[(0, b00, b01, b10)] -= (second_node_weight / w) * (b00 * 1 / second_node_weight)
            coeffs[(0, b00, b01, b10)] -= (second_node_weight / w) * (b01 * 1 / second_node_weight)
            coeffs[(1, b00, b01, b10)] -= (second_node_weight / w) * (b10 * r / second_node_weight)
            coeffs[(1, b00, b01, b10)] -= (second_node_weight / w) * (b11 * r / second_node_weight)

            if core:
              prob_inc += (core_weight / w) * (b00/B)
              prob_inc += (core_weight / w) * (b10/B)
              prob_inc += (first_node_weight / w) * (b01 * r / first_node_weight)
              prob_dec += (core_weight / w) * (b10/B)
              prob_dec += (first_node_weight / w) * (b10 * 1 / first_node_weight)
              prob_dec += (second_node_weight / w) * (b00 * 1 / second_node_weight)
              prob_dec += (second_node_weight / w) * (b01 * 1 / second_node_weight)
            # else:
            #   prob_inc += 
            # print(f"{prob_inc=} vs {prob_dec=}")
            
              

          row = state_to_idx(core, b00, b01, b10)
          for state, coeff in coeffs.items():
            if not is_valid_state(*state): continue
            # print(row, state, coeff)
            col = state_to_idx(*state)
            A[row, col] = coeff

  # Start with a mutant placed randomly.
  A[num_indices, num_indices] = 1
  A[num_indices, state_to_idx(1, B, 0, 0)] = -1/N
  A[num_indices, state_to_idx(0, B-1, 1, 0)] = -(1-1/N)/2
  A[num_indices, state_to_idx(0, B-1, 0, 1)] = -(1-1/N)/2

  b = np.ones(shape=(1+num_indices,))

  # Absorbing states
  b[state_to_idx(1, 0, 0, 0)] = 0
  b[state_to_idx(0, B, 0, 0)] = 0

  # Starting state.
  b[num_indices] = 0

  return A, b

def fan_solution(B: int, r: float):
  A, b = fan_system(B, r)
  T = np.linalg.solve(A, b)
  return T[-1]
  

def potentials(G: nx.DiGraph, r: float = 1.0):
  N = len(G)
  V = set(range(N))
  W = lambda S: (r-1)*len(S) + N

  potentials = []
  for k in range(1, N):
    for Sp in itertools.combinations(G.nodes(), k):
      S = set(Sp)
      Sc = V - S


      ltr_sum = 0
      for x in S:
        for (_, y) in G.out_edges(x):
          if y in Sc:
            ltr_sum += (G.out_degree(x) * G.out_degree(y)) ** -1
      ltr_sum *= r / W(S)

      rtl_sum = 0
      for y in Sc:
        for (_, x) in G.out_edges(y):
          if x in S:
            rtl_sum += (G.out_degree(x) * G.out_degree(y)) ** -1
      rtl_sum *= 1 / W(S)

      potential = ltr_sum - rtl_sum
      potentials.append(potential)

  return potentials


def degree_variance(G: nx.DiGraph):
  ds = []
  for node in G.nodes():
    ind = G.in_degree(node)
    outd = G.out_degree(node)
    assert ind == outd
    ds.append(ind)
  return np.var(ds)

def is_undirected(G: nx.DiGraph):
  return all((v, u) in G.edges() for (u, v) in G.edges())


def is_good(ps: List[float]):
  return all(p >= 0 for p in ps)
  # if not nx.is_strongly_connected(G) and not nx.is_regular(G) and not is_undirected(G)

def plot_potentials():
  N = 5
  R = 1
  total_count = 0
  count = 0
  xs = list(range(2**N-2))
  # for G in yield_all_digraph6(Path(f"data/eulerian/euler{N}.d6")):
  for G in yield_all_digraph6(Path(f"data/directed/direct{N}.d6")):
    # if 0 < degree_variance(G):
    if not nx.is_strongly_connected(G) or nx.is_regular(G) or is_undirected(G): continue
    total_count += 1
    ps = potentials(G, R)
    if is_good(ps):
      print(sorted(((G.in_degree(node), G.out_degree(node)) for node in G.nodes()), reverse=True))
      # for z in nx.algorithms.isomorphism.DiGraphMatcher(G, G).isomorphisms_iter():
      #   print(z)
      nx.draw(G, pos=nx.circular_layout(G), with_labels=True, connectionstyle="arc3,rad=0.1")
      plt.show()
      # plt.plot(xs, ps, marker='o')
      count += 1

  print(f"{count=}")
  print(f"{total_count=}")
  plt.show()

def plot_fans():
  for r in (0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 100, 1_000, 10_000, 100_000):
    times = []
    for b in range(1, 10+1):
      time = fan_solution(B=b, r=r)
      times.append(time)
    plt.plot(times, marker="o", label=f"{r=}")

  plt.xlabel("# blades")
  plt.ylabel("time")
  plt.title("Fan size and r vs time")
  plt.legend()
  plt.show()

  # print(f"{time=}")

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
    keep_idx=lambda idx: is_idx_lowest_circular_symmetry(idx, N) and num_mutants_eq(idx, N // 2, N),
  )

  for extreme in ('max', 'min')[::+1]:
    for rank, (mutants, time) in enumerate(extremes[extreme], start=1):
      draw(G, f"{extreme} {rank=}", N, R, time=time, mutants=mutants, with_stg=False, pos=nx.circular_layout(G))
if __name__ == '__main__':

  # tournament()
  # N = 7
  # tournament((G for G in yield_all_digraph6(Path(f"data/directed-oriented/direct{N}.d6")) if nx.is_strongly_connected(G)))
  # tournament(yield_all_digraph6(Path(f"data/eulerian/euler{N}.d6")))
  # tournament(yield_all_digraph6(Path(f"data/eulerian-oriented/eulerian{N}.d6")))
  # tournament(generate_eulerians(N))
  # plot_potentials()
  plot_fans()
