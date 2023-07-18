#!/usr/bin/env python3
import fractions
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
from typing import DefaultDict, Dict, Generic, Iterable, List, Tuple, TypeVar, Set, Optional

from utils import yield_all_digraph6

from utils import networkx_to_pepa_format
from pepa import g2degs, g2mat, cftime

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
  node_idx = {node: idx for idx, node in enumerate(G.nodes())}
  for u in G.nodes():
    for (_, v) in G.out_edges(u):
      for config in itertools.product((0, 1), repeat=N):
        k = sum(config)
        Wk = N + (r-1) * k
        prob_u_picked = ((r-1) * config[node_idx[u]] + 1) / Wk
        prob_u_to_v_picked = 1.0 / G.out_degree(u)
        next_config = config[:node_idx[v]] + (config[node_idx[u]],) + config[node_idx[v]+1:]

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
  with_stg: bool = False,
  with_labels: bool = False,
  save: bool = True,
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
  ] if mutants else None
  nx.draw(G, node_color=node_colors, with_labels=with_labels, *G_draw_args, **G_draw_kwargs)
  if with_stg:
    plt.subplot(num_subplots, 1, 2)
    draw_stg(G)

  if save:
    plt.savefig(f'charts/{prefix}-N{N}-r{r}.png', dpi=300)

  plt.show()

def generate_eulerians(N):
  by_degree_seq = defaultdict(list)
  for G in eulerian(N):
    if (d := deg_seq(G)) not in by_degree_seq or not any(nx.is_isomorphic(G, H) for H in by_degree_seq[d]):
      by_degree_seq[d].append(G)
      yield G

def tournament(generator, type, *args, rs=(1.1,), k=1, show=True, interactive=False, invert=False, **kwargs):
  count = 0
  K = k
  SHOW = show
  INTERACTIVE = interactive
  # r -> (time, G)
  max_exp_abs_time_by_r: DefaultDict[float, MaxExamples[float, nx.DiGraph]] = defaultdict(
    lambda: MaxExamples(K, invert=invert)
  )
  for G in generator:
      for r in rs:
        time = expected_absorption_time_single_random_mutant(G, r)
        if max_exp_abs_time_by_r[r].add(time, G) and SHOW and INTERACTIVE:
          draw(G, "Update!", len(G), r, time, pos=nx.circular_layout(G), connectionstyle="arc3,rad=0.1", *args, **kwargs)

      count += 1
  
  print(f"{count=}")

  extreme = "fastest" if invert else "slowest"
  if SHOW:
    for r, examples in max_exp_abs_time_by_r.items():
      for (time, _, G) in examples.get():
        draw(G, f"{type}-{extreme}", len(G), r, time, pos=nx.kamada_kawai_layout(G), connectionstyle="arc3,rad=0.1", *args, **kwargs)

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
  
def cross_edges(G: nx.DiGraph, S, T):
  return list((u, v) for u in S for (_, v) in G.out_edges(u) if v in T)

def potentials_1(G: nx.DiGraph, r: float = 1.0):
  N = len(G)
  V = set(range(N))
  W = lambda S: (r-1)*len(S) + N

  def conductance(S_boundary, S, Sc):
    assert len(S|Sc) == N
    print(list(S_boundary), list(S), list(Sc), k)
    return ((1+flow_edge_set_eulerian(G, S_boundary)) / (1+min(flow_vertex_set_eulerian(G, S), flow_vertex_set_eulerian(G, Sc)))) ** -1

  potentials = []
  def phi(S):
    return conductance(cross_edges(G, S, V-S), S, V-S)

  for k in range(1, N):
    for Sp in itertools.combinations(G.nodes(), k):
      S = set(Sp)
      Sc = V - S

      c = conductance(cross_edges(G, S, Sc), S, Sc)
      ltr_sum = 0
      for x, y in cross_edges(G, S, Sc):
        ltr_sum += 1/G.out_degree(x) * (phi(S|{y}) - c)
      ltr_sum *= r / W(S)

      rtl_sum = 0
      for y, x in cross_edges(G, Sc, S):
        rtl_sum += 1/G.out_degree(y) * (c - phi(S-{x}))
      rtl_sum *= 1 / W(S)

      potential = ltr_sum + rtl_sum
      potentials.append(potential)

  return potentials

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

def is_oriented(G: nx.DiGraph):
  return all(
    not (
      (u, v) in G.edges() and (v, u) in G.edges()
    )
    for (u, v) in G.edges()
  )

def is_good(ps: List[float]):
  return all(p >= 0 for p in ps)

def h(G: nx.DiGraph):
  """edge expansion"""
  N = len(G)
  V = set(range(N))
  conductances = []
  for k in range(1, N):
    for Sp in itertools.combinations(G.nodes(), k):
      S = set(Sp)
      Sc = V - S
      boundary = set()
      for u in S:
        for (_, v) in G.out_edges(u):
          if v in Sc:
            boundary.add((u, v))
      conductance = flow_edge_set_eulerian(G, boundary) / min(flow_vertex_set_eulerian(G, S), flow_vertex_set_eulerian(G, Sc))
      # conductance = len(boundary) / k
      conductances.append(conductance)
  return np.mean(conductances), conductances

def flow_edge_set_eulerian(G: nx.DiGraph, S: Iterable[Tuple[int, int]]):
  return sum(
    flow_eulerian(G, u, v)
    for (u, v) in S
  )

def flow_vertex_set_eulerian(G: nx.DiGraph, S: Iterable[int]):
  return sum(
    flow_single_eulerian(G, v)
    for v in S
  )

def flow_single_eulerian(G: nx.DiGraph, v: int):
  return sum(
    flow_eulerian(G, u, v)
    for (u, _) in G.in_edges(v)
  )

def flow_eulerian(G: nx.DiGraph, u: int, v: int):
  if (u, v) not in G.edges(): return 0
  puv = 1/G.out_degree(u)
  phi = G.out_degree(u) / G.number_of_edges()
  return phi * puv

def find_r(G: nx.DiGraph, *, up: bool):
  SEARCH_ITERATIONS = 100
  multiple = 2 if up else 1/2
  agr = min if up else max
  # Find upper bound.
  advantageous_r_lower = advantageous_r_upper = 1
  while not is_good(potentials(G, advantageous_r_upper)) and advantageous_r_upper:
    advantageous_r_upper *= multiple
    # print(advantageous_r_upper)

  # Binary search.
  best_advantageous_r = np.inf if up else 0
  for _ in range(SEARCH_ITERATIONS):
    advantageous_r_mid = (advantageous_r_lower + advantageous_r_upper) / 2
    ps = potentials(G, advantageous_r_mid)
    if is_good(ps):
      best_advantageous_r = agr(advantageous_r_mid, best_advantageous_r)
      advantageous_r_lower = advantageous_r_mid
    else:
      advantageous_r_upper = advantageous_r_mid
  return best_advantageous_r

def drifts(G: nx.DiGraph, r: float):
  N = len(G)
  V = G.nodes()
  ds = []
  for k in range(1, N):
    for Sp in itertools.combinations(G.nodes(), k):
      S = set(Sp)
      Sc = V - S
      p_inc = sum(1/G.out_degree(u) for (u, _) in cross_edges(G, S, Sc))
      p_dec = sum(1/G.out_degree(v) for (v, _) in cross_edges(G, Sc, S))
      ds.append(r*p_dec/p_inc)
  return ds

def num_cross_edges(G: nx.DiGraph):
  N = len(G)
  V = G.nodes()
  count = 0
  for k in range(1, N):
    for Sp in itertools.combinations(G.nodes(), k):
      S = set(Sp)
      Sc = V - S
      count += sum(1 for _ in cross_edges(G, S, Sc))
  return count

def plot_drifts():
  N = 4
  R = 1
  total_count = 0
  count = 0
  mins = []
  maxs = []
  avgs = []
  stds = []
  total_cross_edges = []
  # for G in yield_all_digraph6(Path(f"data/eulerian/euler{N}.d6")):
  for G in yield_all_digraph6(Path(f"data/directed/direct{N}.d6")):
    # if 0 < degree_variance(G):
    # if not nx.is_strongly_connected(G) or nx.is_regular(G) or is_undirected(G) or _is_eulerian(G): continue
    if not nx.is_strongly_connected(G): continue # or nx.is_regular(G) or is_undirected(G): continue
    total_count += 1
    ds = drifts(G, R)
    # mins.append(min(ds))
    # maxs.append(max(ds))
    avgs.append(np.mean(ds))
    stds.append(np.std(ds))
    total_cross_edges.append(num_cross_edges(G) / G.number_of_edges())
    print(total_cross_edges)
    # plt.plot(ds, marker='o')

  print(f"{count=}")
  print(f"{total_count=}")
  plt.figure()
  plt.hist(total_cross_edges)
  # plt.figure()
  # plt.hist(avgs, bins=100)
  # plt.figure()
  # plt.hist(stds, bins=100)
  plt.show()

def plot_r_potential_boundaries():
  import matplotlib
  from matplotlib.backend_bases import PickEvent
  N = 6
  def on_pick(event: PickEvent):
    print(event)
    print(event.artist)
    idx = event.ind[0]
    H = undirected_graphs[idx]
    plt.figure()
    nx.draw(H, pos=nx.circular_layout(H), with_labels=True, connectionstyle="arc3,rad=0.1")
    plt.show()

  fig, axes = plt.subplots(3, 3)
  fig.tight_layout(rect=[0, 0.03, 1, 0.95])
  fig.supxlabel("average conductance")
  fig.supylabel("expected absorption time")
  fig.suptitle(f"average conductance vs expected absorption time for Eulerian {N=}")
  fig.canvas.callbacks.connect('pick_event', on_pick)
  for idx, r in enumerate((.1, .9, .99, 1, 1.01, 1.1, 1.5, 10, 100), start=0):
    avg_conductances_undirected = []
    abs_times_undirected = []
    avg_conductances_directed = []
    abs_times_directed = []
    undirected_graphs = []
    directed_graphs = []
    for G in (x for x in yield_all_digraph6(Path(f"data/eulerian/euler{N}.d6")) if random.random() <= 1.00005):
      if not nx.is_strongly_connected(G): continue #or not nx.is_regular(G): continue #  or is_undirected(G): continue
      avg_conductance, conductances = h(G)
      if is_undirected(G):
        abs_times_undirected.append(abs_time := expected_absorption_time_single_random_mutant(G, r))
        avg_conductances_undirected.append(avg_conductance)
        undirected_graphs.append(G)
      else:
        abs_times_directed.append(abs_time := expected_absorption_time_single_random_mutant(G, r))
        avg_conductances_directed.append(avg_conductance)
        directed_graphs.append(G)

      # if avg_conductance == 1:
      #   print(conductances)
      #   nx.draw(G)
      #   plt.show()
  
    axes[idx // 3, idx % 3].scatter(avg_conductances_directed, abs_times_directed, s=15, c='green', alpha=.2)
    axes[idx // 3, idx % 3].scatter(avg_conductances_undirected, abs_times_undirected, s=15, c='purple', alpha=.2, picker=5, data=undirected_graphs)
    axes[idx // 3, idx % 3].set_title(f"{r=}")
  plt.show()

def plot_potentials():
  N = 5
  R = 1
  total_count = 0
  count = 0
  for G in yield_all_digraph6(Path(f"data/eulerian/euler{N}.d6")):
  # for G in yield_all_digraph6(Path(f"data/directed-oriented/direct{N}.d6")):
    # if 0 < degree_variance(G):
    # if not nx.is_strongly_connected(G) or nx.is_regular(G) or is_undirected(G) or _is_eulerian(G): continue
    if not nx.is_strongly_connected(G): continue
    total_count += 1
    ps = potentials_1(G, R)
    if good := is_good(ps):
      # print(sorted(((G.in_degree(node), G.out_degree(node)) for node in G.nodes()), reverse=True))
      # for z in nx.algorithms.isomorphism.DiGraphMatcher(G, G).isomorphisms_iter():
      #   print(z)
      # nx.draw(G, pos=nx.circular_layout(G), with_labels=True, connectionstyle="arc3,rad=0.1")
      # plt.show()
      count += 1
    plt.plot(ps, marker='o', color=['red', 'green'][good])

  print(f"{count=}")
  print(f"{total_count=}")
  plt.show()

def plot_fans(N):
  Ns = list(range(1, N+1))
  data = []
  for blades in Ns:
    print(blades)
    for r in (1., 1.1, 1.5, 100.): # (0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 100, 1_000, 10_000, 100_000):
      time = fan_solution(B=blades, r=r)
      data.append((blades, time, r))
    # plt.plot(times, marker="o", label=f"{r=}")

  df = pd.DataFrame(data, columns=["Number of blades (b)", "Absorption time (t)", "Reproductive Fitness (r)"])
  plot = sns.lineplot(
    # kind="line",
    data=df,
    x="Number of blades (b)",
    y="Absorption time (t)",
    hue="Reproductive Fitness (r)",
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

  style(plot)

  plt.xscale('log')
  plt.yscale('log')
  ax = plt.gca()
  # ax.set_xticks(Ns)

  plt.xticks(rotation=-45) 
  plt.savefig(f'charts/fans-r-vs-at-{N}.png', dpi=300)
  plt.show()

  # print(f"{time=}")

def custom_graph_absorptions():
  N = 12
  R = 100

  G: nx.DiGraph = nx.DiGraph()
  for i in range(N):
    G.add_edge(i, (i+1)%N)
  G.add_edge(N-1, 0)

  # for graph in range(num_subgraphs):
  #   for i in range(N):
  #     for j in range(N):
  #       G.add_edge((graph, i), (graph, j))
  #       G.add_edge((graph, j), (graph, i))
  
  # for graph_a in range(num_subgraphs):
  #   for graph_b in range(graph_a+1, num_subgraphs):
  #     G.add_edge((graph_a, 0), ("connector", graph_a, graph_b))
  #     G.add_edge(("connector", graph_a, graph_b), (graph_a, 0))

  #     G.add_edge((graph_b, 0), ("connector", graph_a, graph_b))
  #     G.add_edge(("connector", graph_a, graph_b), (graph_b, 0))

  for num_mutants in range(1, N):
    extremes = extreme_expected_absorption_times(
      G,
      r=R,
      k=1,
      keep_idx=lambda idx: is_idx_lowest_circular_symmetry(idx, N) and num_mutants_eq(idx, num_mutants, N),
    )

    for extreme in ('max', 'min')[::+1]:
      for rank, (mutants, time) in enumerate(extremes[extreme], start=1):
        draw(G, f"{extreme} {rank=} {num_mutants=}", N, R, time=time, mutants=mutants, with_stg=False, pos=nx.circular_layout(G), connectionstyle="arc3,rad=0.1")

def stationary_probabilities(G: nx.DiGraph):
  return nx.pagerank(G)

def fixation_probabilities(G: nx.DiGraph):
  """Compute fixation probabilities under neutral evolution."""
  N = len(G)
  A = np.zeros((N, N))
  b = np.zeros((N,))

  V = list(G.nodes())
  subsets = {None} | {u for u, _ in zip(V, range(N-1))}
  idx = {u: i for i, u in enumerate(subsets)}

  loner_set = set(V) - subsets
  assert len(loner_set) == 1
  loner = list(loner_set)[0]
  idx[loner] = len(subsets)
  # label = {v: k for k, v in idx.items()}
  for u in subsets:
    if u is None:
      A[idx[u], idx[u]] = 1
      continue

    for (_, v) in G.out_edges(u):
      assert v != u, "No self loops allowed for this calculation."
      if idx[v] < len(subsets):
        A[idx[u], idx[v]] += 1 / G.out_degree(u)
      else:
        for w in subsets - {None}:
          A[idx[u], idx[w]] -= 1 / G.out_degree(u)
        b[idx[u]] -= 1 / G.out_degree(u)

    A[idx[u], idx[u]] -= sum(
      1 / G.out_degree(v)
      for (v, _) in G.in_edges(u)
    )

  fp = np.linalg.solve(A, b)
  fp_dict = {
    u: fractions.Fraction.from_float(fp[idx[u]]).limit_denominator(6**9)
    for u in subsets - {None}
  }
  fp_dict[loner] = 1-sum(fp_dict.values())
  return fp_dict

      
def two_cycles():
  N = 30
  G = nx.DiGraph()
  for i in range(N):
    G.add_edge(i, (i+1)%N)

  k = 5
  for i in range(N):
    G.add_edge(i+N-k, N-k+(i-1)%N)

  return G

def custom_graph():
  N = 6
  G = nx.DiGraph()
  for level in range(1, N):
    for node in itertools.product((0, 1), repeat=level):
      for i, pnode in enumerate(itertools.product((0, 1), repeat=level-1)):
        # if i % 2 == node[0]: continue
      # G.add_edge(node, node[:-1])
        G.add_edge(node, pnode)
  
  for node in itertools.product((0, 1), repeat=N-1):
    G.add_edge((), node)

  nx.draw(G, pos={v: (len(v), int(''.join(map(str, v)) or '0', 2)) for v in G.nodes()}, with_labels=True)
  plt.show()
  return G



import math
from statistics import mean
import seaborn as sns
import pandas as pd
sns.set_theme(font_scale=2, rc={'text.usetex' : True})
sns.set_style("whitegrid", {'axes.grid' : False})

def temperature(G: nx.DiGraph, v):
  return sum(1/G.out_degree(u) for (u, _) in G.in_edges(v))

from matplotlib.backend_bases import PickEvent

def fp_vs_cftime(N: int):
  from utils import networkx_to_pepa_format
  from pepa import g2degs, g2mat, cftime, fprob
  data = []
  Gs = []
  count = 0

  def on_pick(event: PickEvent):
    print(event)
    print(event.artist)
    idx = event.ind[0]
    H = Gs[idx]
    plt.figure()
    nx.draw(H, pos=nx.kamada_kawai_layout(H), with_labels=True, connectionstyle="arc3,rad=0.1")
    plt.show()

  for G_nx in yield_all_digraph6(Path(f"data/directed/direct{N}.d6")):
    if not nx.is_strongly_connected(G_nx): continue
    count += 1
    G = networkx_to_pepa_format(G_nx)
    degs = g2degs(G)
    undirected = is_undirected(G_nx)
    oriented = is_oriented(G_nx)
    for r in (1.1,):# , 2.): # np.linspace(start=0.1, stop=10, num=4):
      # temps = [1/temperature(G_nx, v) for v in G_nx.nodes()]
      Gs.append(G_nx)
      temps = [1] * N
      mat = g2mat(G, degs, r)
      cft = np.average(cftime(mat), weights=temps)
      fp = np.average(fprob(mat), weights=temps)
      type_ = 'undirected' if undirected else 'oriented' if oriented else 'directed'
      data.append((fp, cft, r, type_, G_nx.number_of_edges()))

  df = pd.DataFrame(data, columns=["Fixation probability (p)", "Fixation time (t)", "r", "type", "num edges"])
  # df.sort_values(by="type")
  df = df[df.type != 'directed']# .sort_values(by="type")
  # plt.scatter(fps_, cftimes, s=2)
  plot = sns.relplot(
    data=df,
    x="Fixation probability (p)",
    y="Fixation time (t)",
    # aspect="num edges",
    # markers="undirected",
    # scatter_kws={'alpha': 0.5, "s": 8, 'linewidths': 0},
    facet_kws={'sharey': True, 'sharex': False},
    hue="type",
    # hue_order=["oriented", "undirected", "directed"],
    col="r",
    col_wrap=2,
    picker=4,
    # sharex=False,
    # sharey=True,
    # fit_reg=False,
  )
  for ax in plt.gcf().get_axes():
    ax.set(xlabel='Fixation probability, $\\rho$', ylabel='Fixation time, $T$')
    ax.figure.canvas.mpl_connect("pick_event", on_pick)

  # Define the z-order for different hues
  zorder_dict = {'oriented': 1, 'undirected': 2, 'directed': 3}

  # Iterate over the scatter plot artists
  for artist in plot.legend.legendHandles:
    # Get the hue label
    hue_label = artist.get_label()
    
    # Get the corresponding z-order value from the dictionary
    zorder = zorder_dict.get(hue_label)
    
    # Set the z-order for the artist
    artist.set_zorder(zorder)

  style(plot)
  print(f"{count=}")
  # sns.relplot(
  #   data=tips,
  #   x="total_bill", y="tip", col="time",
  #   hue="smoker", style="smoker", size="size",
  # )
  plt.savefig(f'charts/fp-vs-ft-{N}.png', dpi=300)
  plt.show()

def fps():
  G = custom_graph()
  fp = fixation_probabilities(G)
  for u in G.nodes():
    # print(f"fp({u})={fp[u]}")
    nx.set_node_attributes(G, {u: fp[u]}, "fp")

  labels = nx.get_node_attributes(G, 'fp') 

  fig = plt.figure()
  ax = plt.gca()
  ax.set_title(f"N={len(G)}")
  str_values = ", ".join([f"{u} -> {str(fp[u])}" for u in G.nodes()])
  largest_denominator = functools.reduce(math.lcm, (f.denominator for f in fp.values()))
  scaled_fps = {v: fp[v] * largest_denominator for v in fp.keys()}
  print(f"{largest_denominator=}")
  for v in sorted(scaled_fps.keys(), key=lambda t: (len(t), t)):
    print(f"{v} --> {scaled_fps[v].numerator}")
  # print(f"edges={G.edges()}, values={{{str_values}}}")
  nx.draw(
    G,
    pos=nx.spectral_layout(G),
    ax=ax,
    labels=labels,
    connectionstyle="arc3,rad=0.1",
    font_size=5,
    node_size=20,
  )
  plt.show()

def symbolic_stuff():
  from sympy import symbols, solve, simplify
  N = 3
  V = tuple(range(N))
  vec = symbols(f'x:{N}', real=True, nonnegative=True)
  matrix = symbols(f"A:{N}:{N}", integer=True, nonnegative=True)

  x = vec
  A = tuple(
    matrix[i*N:i*N+N]
    for i in V
  )

  eqs = []
  eqs.append(sum(x)-1)
  eqs.extend([
    1/(sum(A[u][v]for v in V)) * sum(A[u][v]*x[v] for v in V)
    - x[u] * sum(A[v][u]/sum(A[v][w] for w in V) for v in V)
    for u in V
  ])

  solutions = solve(eqs, x, dict=True)
  for k, v in solutions[0].items():
    print(k, simplify(v))


  assumptions = []
  assumptions.extend([
    [
      sum(A[u][v] for v in V),
      sum(A[v][u] for v in V),
    ]
    for u in V
  ])

from statistics import mean


def monotonicity_of_ft(N):
  data = []
  example_graph = None
  Rs = list(np.linspace(start=1, stop=1.30, num=10))
  print(len(Rs))
  for graph_idx, G_nx in enumerate(yield_all_digraph6(Path(f"data/directed/direct{N}.d6"))):
    if not nx.is_strongly_connected(G_nx): continue
    G = networkx_to_pepa_format(G_nx)
    degs = g2degs(G)
    accs = []
    mat_1 = g2mat(G, degs, 1)
    cfts_1 = cftime(mat_1)
    for r in Rs:
      mat = g2mat(G, degs, r)
      cfts = cftime(mat)
      for node_idx in range(N):
        data.append(((graph_idx, node_idx), r, cfts[node_idx] / cfts_1[node_idx], node_idx, G, 0.5))

  # Find extreme values.
  extremes = (
    max(d[2] for d in data if d[1] == Rs[-1]),
    min(d[2] for d in data if d[1] == Rs[-1])
  )

  up = set()
  for d in data:
    if d[0] in up: continue
    if d[2] <= 1: continue
    up.add(d[0])
  
  print(f"num trajectories that increase at some point: {len(up)}")

  extreme_graph_idxs = set()
  for i in range(len(data)):
    if (r := data[i][1]) != Rs[-1]: continue
    normalized_ft = data[i][2]
    if normalized_ft not in extremes: continue
    graph_idx = data[i][0]
    extreme_graph_idxs.add(graph_idx)
    print(graph_idx, data[i][-2], normalized_ft, data[i][-3])
  
  for i in range(len(data)):
    graph_idx = data[i][0]
    if graph_idx not in extreme_graph_idxs: continue
    data[i] = data[i][:-1] + (1.0,)

  df = pd.DataFrame(data, columns=["graph idx", "Relative fitness (r)", "Normalized fixation time (t)", "node", "pepa", "color"])
  print(df)
  plot = sns.lineplot(
    # kind="line",
    # col="node",
    # col_wrap=2,
    data=df,
    x="Relative fitness (r)",
    y="Normalized fixation time (t)",
    # hue="counterexample",
    units="graph idx",
    estimator=None,
    palette="light:b",
    hue="color",
    # facet_kws={"ylim": (23.698, 23.705)},
    # style="counterexample",
    # markers=True,
    #, dashes=False,
    legend=False,
  )

  style(plot)
  ax = plt.gca()
  handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles, ['$r = 1.1$', '$r = 100$'], title='')
  ax.set(xlabel='Relative fitness, $r$', ylabel='Normalized fixation time, $T$')

  # height = df["normalized fixation time"].max()
  # plt.plot([1, 1], [0, height], linewidth=0.5, linestyle='--')
  plt.ticklabel_format(useOffset=False)
  plt.savefig(f'charts/normalized-r-vs-ft-{N}.png', dpi=300, bbox_inches="tight")
  plt.show()



def example_monotonicity_of_ft(N): 
  data = []
  example_graph = None
  Rs = list(np.linspace(start=1, stop=1.05, num=100))
  print(len(Rs))
  for graph_idx, G_nx in enumerate(yield_all_digraph6(Path(f"data/directed/direct{N}.d6"))):
    if not nx.is_strongly_connected(G_nx): continue
    G = networkx_to_pepa_format(G_nx)
    degs = g2degs(G)
    accs = []
    mat_1 = g2mat(G, degs, 1)
    cfts_1 = cftime(mat_1)
    for r in Rs:
      mat = g2mat(G, degs, r)
      cfts = cftime(mat)
      for node_idx in range(N):
        if graph_idx == 9 and node_idx == 0:
          example_graph = G_nx
          data.append(((graph_idx, node_idx), r, cfts[node_idx], node_idx, G, 0.5))

  df = pd.DataFrame(data, columns=["graph idx", "Relative fitness (r)", "Fixation time (t)", "node", "pepa", "color"])
  print(df)
  plot = sns.lineplot(
    # kind="line",
    # col="node",
    # col_wrap=2,
    data=df,
    x="Relative fitness (r)",
    y="Fixation time (t)",
    # hue="counterexample",
    units="graph idx",
    estimator=None,
    # palette="light:b",
    # hue="color",
    # ylim=(23.698, 23.705),
    # style="counterexample",
    # markers=True,
    #, dashes=False,
    legend=False,
  )
  plt.ylim(23.698, 23.705)

  # plot.fig.set_size_inches(15,15)
  style(plot)
  ax = plt.gca()
  handles, labels = ax.get_legend_handles_labels()
  ax.set(xlabel='Relative fitness, $r$', ylabel='Fixation time, $T$')

  # height = df["normalized fixation time"].max()
  # plt.plot([1, 1], [0, height], linewidth=0.5, linestyle='--')
  plt.ticklabel_format(useOffset=False)
  plt.savefig(f'charts/r-vs-ft-{N}.png', dpi=300, bbox_inches="tight")

  # plt.title(f"r vs expected fixation time, {N=}")
  # plt.xlabel("r")
  # plt.ylabel("expected fixation time")
  # for rs, fts in zip(rss, ftss):
  #   plt.plot(rs, fts, marker='o')
  # plt.show()
  plt.figure()
  nx.draw(
    example_graph,
    pos=nx.kamada_kawai_layout(example_graph),
    with_labels=False,
    connectionstyle="arc3,rad=0.1",
    node_color=["red"] + ["blue"] * 3,
  )
  plt.savefig(f'charts/example-graph-r-vs-ft-{N}.png', dpi=300, transparent=True, bbox_inches="tight")


def does_fp_only_depend_on_incoming_degrees():
  N = 4
  for G in yield_all_digraph6(Path(f"data/eulerian/euler{N}.d6")):
    fp = fixation_probabilities(G)
    for v, vp in itertools.product(G.nodes(), repeat=2):
      if G.in_degree(v) != G.in_degree(vp): continue
      if v == vp: continue
      v_degree = G.in_degree(v)
      vp_degree = G.in_degree(vp)
      v_ins = sorted(G.out_degree(u) for u, _ in G.in_edges(v))
      vp_ins = sorted(G.out_degree(up) for up, _ in G.in_edges(vp))
      v_fp = fp[v]
      vp_fp = fp[vp]

      if v_degree == vp_degree and v_ins == vp_ins:
        if v_fp == vp_fp:
          print("hurray!")
        else:
          print(":((((")
          print(f"{v=}, {vp=}, {v_ins=}, {vp_ins=}")
          print(fp)
          nx.draw(G, pos=nx.circular_layout(G), with_labels=True, connectionstyle="arc3,rad=0.1")
          plt.show()

def lol():
  xs = []
  fpls = []
  fpus = []
  fps_lowers = []
  fps_avgs = []
  fps_uppers = []
  for N in range(1, 50):
    G = nx.DiGraph()
    for i in range(N):
      for j in range(N):
        if i != j:
          G.add_edge(i, j)

    G.add_edge(N, 0)
    G.add_edge(N-1, N)

    leps = 0
    ueps = 0

    data = []
    davg_sum = 0
    for v in G.nodes():
      in_vertices = [G.out_degree(u) for (u, _) in G.in_edges(v)]
      davg = mean(in_vertices)
      davg_sum += davg
      dmin = min(in_vertices)
      dmax = max(in_vertices)
      # (1-eps)*davg <= dmin   ==>   1 - dmin/davg <= eps
      leps = max(leps, 1-dmin/davg)
      # (1+eps)*davg >= dmax   ==>   dmax/davg - 1 <= eps
      ueps = max(ueps, dmax/davg - 1)
      # local_eps = max(leps, ueps)
      data.append([v, leps, ueps, davg])
      # print(f"(1-{leps})*{davg} <= {dmin} <= {davg} <= {dmax} <= (1+{ueps})*{davg}")
    
    num_edges = G.number_of_edges()
    print(f"{davg_sum=}, {num_edges=}")
    for datum in data:
      v, leps, ueps, davg = datum
      ufpl = (1-leps)*davg
      ufpu = (1+ueps)*davg
      print(f"{ufpl=} {davg=} {ufpu=}")
      datum.extend([ufpl, ufpu])

    slb = sum(ufpu for _, _, _, _, _, ufpu in data)
    sub = sum(ufpl for _, _, _, _, ufpl, _ in data)
    min_fpl = np.inf
    max_fpu = 0
    for v, _, _, davg, ufpl, ufpu in data:
      fpl = ufpl / sub
      fpu = ufpu / slb
      min_fpl = min(min_fpl, fpl)
      max_fpu = max(max_fpu, fpu)

      print(f"{fpl} <= fp({v}) <= {fpu}, {davg=}")
    # print(f"so {(1-eps)/((1+eps)*N)} <= fp <= {(1+eps)/(1-eps)}")
    # input()
    xs.append(N)
    fpls.append(min_fpl)
    fpus.append(max_fpu)

    fp = fixation_probabilities(G)
    fps_lowers.append(min(fp.values()))
    fps_avgs.append(mean(fp.values()))
    fps_uppers.append(max(fp.values()))
    print()
    # nx.draw(G, pos=nx.circular_layout(G), with_labels=True, connectionstyle="arc3,rad=0.1")
    # plt.show()

  log = lambda ys: [-np.log(float(y)) for y in ys]
  # plt.plot(xs, log(fpls), label='lower bound')
  # plt.plot(xs, log(fpus), label='upper bound')
  plt.plot(xs, log(fps_lowers), label='actual lower fp')
  plt.plot(xs, log(fps_avgs), label='actual average fp')
  plt.plot(xs, log(fps_uppers), label='actual upper fp')
  # plt.yscale('log')
  plt.legend()
  plt.show()

def laplacian(G: nx.DiGraph):
  A = nx.adjacency_matrix(G)
  D = np.diag([
    G.out_degree(v) * sum(G.out_degree(u)**-1 for (u, _) in G.in_edges(v))
    for v in G.nodes()
  ])
  L = D-A
  return L

def laplacians():
  N = 3
  for G in yield_all_digraph6(Path(f"data/eulerian/euler{N}.d6")):
    if is_undirected(G): continue
    L = laplacian(G)
    fp = fixation_probabilities(G)
    ls = np.linalg.eig(L)
    print(ls)
    print(L)
    print(fp)
    nx.draw(G, with_labels=True, pos=nx.spectral_layout(G), connectionstyle="arc3,rad=0.1")
    plt.show()

def both_stationaries(G: nx.DiGraph):
  A = nx.adjacency_matrix(G)
  D = np.diag([G.out_degree(v) for v in G.nodes()])
  T = np.diag([temperature(G, v) for v in G.nodes()])
  zeros = np.zeros(shape=(len(G),))

  L = np.eye(len(G)) - np.invert(D)@A
  # Lt = T@D - A

  lams, pis = np.linalg.eig(L)
  # ltams, phis = np.linalg.eig(Lt)
  print(nx.is_regular(G))
  print(pis)
  print("-------")
  # print(phis)
  print(T.diagonal())
  print("=========")

def epsilon(G: nx.DiGraph):
  eps = 0
  for v in G.nodes():
    for (_, w) in G.out_edges(v):
      normalization = (1/G.out_degree(v)) * sum(1/temperature(G, z) for (_, z) in G.out_edges(v))
      b = (1/temperature(G, w))/normalization
      eps = max(abs(b-1), eps)
  return eps

def max_degree(G: nx.DiGraph):
  assert _is_eulerian(G)
  return max(G.out_degree(v) for v in G.nodes())

def min_degree(G: nx.DiGraph):
  assert _is_eulerian(G)
  return min(G.out_degree(v) for v in G.nodes())

def biased_stationary(G: nx.DiGraph):
  A = nx.adjacency_matrix(G)
  D = np.diag([G.out_degree(v) for v in G.nodes()])
  T = np.diag([temperature(G, v) for v in G.nodes()])
  print(A)
  print(D)
  print(T)

  B = np.linalg.inv(T)@np.linalg.inv(D)@A

  x_un = np.linalg.eig(B)
  print(x_un)
  input()
  norm = np.linalg.norm(x_un) ** 2
  x = x_un / norm
  return x

def normalize(dictionary):
  denom = sum(dictionary.values())
  return {
    k: v / denom
    for k, v in dictionary.items()
  }

def to_floats(dictionary):
  return {
    k: float(v)
    for k, v in dictionary.items()
  }

def derp():
  N = 8
  for G in yield_all_digraph6(Path(f"data/eulerian/euler{N}.d6")):
    if is_undirected(G) or nx.is_regular(G): continue
    xs = []
    ys = []
    fixations = fixation_probabilities(G)
    stationaries = {
      k: fractions.Fraction.from_float(v).limit_denominator(10**6)
      for k, v in stationary_probabilities(G).items()
    }
    custom_stationaries = normalize({
      # k: v * temperature(G, k)**-1 / sum(temperature(G, l)**-1 for (_, l) in G.out_edges(k))
      # k: temperature(G, k)**-1 * v**-1
      k: (1/v) # * temperature(G, k)**-1 * sum(1/G.out_degree(l) for (_, l) in G.out_edges(k))
      for k, v in stationaries.items()
    })
    print("edges:", G.edges())
    print("stationary probabilities:", to_floats(stationaries))
    print("???:", to_floats(fixations))
    print("==============")

    if True:
      for i in range(N):
        xs.append(float(fixations[i]))
        ys.append(float(custom_stationaries[i]))
    
      _, ax = plt.subplots(2, 2)
      ax[0, 0].bar(list(range(N)), xs, label='fp', color='green')
      ax[1, 0].bar(list(range(N)), ys, label='pi', color='red')
      plt.legend()
      nx.draw(
        G,
        pos=nx.kamada_kawai_layout(G),
        with_labels=True,
        connectionstyle="arc3,rad=0.1",
        node_size=[100*N*temperature(G, v) for v in G.nodes()],
        ax=ax[0, 1],
      )
      nx.draw(
        G,
        pos=nx.kamada_kawai_layout(G),
        with_labels=True,
        connectionstyle="arc3,rad=0.1",
        node_size=[100*N*sum(1/G.out_degree(w) for (_, w) in G.out_edges(v)) for v in G.nodes()],
        ax=ax[1, 1],
      )
      manager = plt.get_current_fig_manager()
      manager.full_screen_toggle()
      plt.show()
      # plt.clf()


def has_P_property(G: nx.DiGraph):
  for v in G.nodes():
    delta_v = None
    for (u, _) in G.in_edges(v):
      odeg_u = G.out_degree(u)
      if delta_v is not None and odeg_u != delta_v: return False
      delta_v = odeg_u
  return True

def symbolicc():
  from sympy import symbols, Eq, solve
  N = 5
  # for G in yield_all_digraph6(Path(f"data/eulerian/euler{N}.d6")):
  for G in yield_all_digraph6(Path(f"data/directed/direct{N}.d6")):
    if not nx.is_strongly_connected(G): continue
    if not has_P_property(G): continue
    print("============")
    print(G.edges())
    ys = symbols([f'y_{v}' for v in range(N)])
    odegs = symbols([f'do_{v}' for v in range(N)])
    idegs = symbols([f'di_{v}' for v in range(N)])
    Ap = nx.adjacency_matrix(G)
    # A = [
    #   symbols([f'A_{i}{j}' for j in range(N)])
    #   for i in range(N)
    # ]
    equations = [
      Eq(idegs[i], sum(Ap[j,i] for j in range(N)))
      for i in range(N)
    ] + [
      Eq(odegs[i], sum(Ap[i,j] for j in range(N)))
      for i in range(N)
    ] + [
      Eq(
        ys[v] * sum(Ap[u,v]/odegs[u] for u in range(N)),
        sum([Ap[v,w]*ys[w]/odegs[v] for w in range(N)]),
      )
      for v in range(N)
    ] + [
      Eq(sum(ys[v] for v in range(N)), 1)
    ]
    # [
    #   Eq(sum(A[i][k] for k in range(N)), sum(A[k][i] for k in range(N)))
    #   for i in range(N)
    # ]
    # [
    #   Eq(A[i][j]**2, A[i][j])
    #   for i in range(N)
    #   for j in range(N)
    # ] +
    #   Eq(A[i][j], Ap[i, j])
    #   for i in range(N)
    #   for j in range(N)
    # ] +
    # [
    #   Eq(Ap[i, i], 0) for i in range(N)
    # ] +
    #+ [
    #   Eq(
    #     ys[v] * sum(1/degs[u] for (u, _) in G.in_edges(v)),
    #     sum([ys[w]/degs[w] for (_, w) in G.out_edges(v)]),
    #   )
    #   for v in G.nodes()
    # ]
    # if not nx.is_strongly_connected(G): continue
    solution = solve(equations, ys)
    for k, v in solution.items():
      print(f"{k}: {v}")

    nx.draw(
      G,
      pos=nx.kamada_kawai_layout(G),
      with_labels=True,
      connectionstyle="arc3,rad=0.1",
    )
    plt.show()

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

def trial_cftime(G: nx.DiGraph, S: Optional[Set], r: float):
  if S is None:
    S = {random.choice(list(G.nodes()))}

  N = len(G)
  V = G.nodes()
  mutants = set()
  mutants |= S
  steps = 0

  while V - mutants:
    if not mutants: return None
    k = len(mutants)
    if random.random() < r*k/(N + (r-1)*k):
      birther = random.choice(list(mutants))
    else:
      birther = random.choice(list(V - mutants))

    dier = random.choice([w for (_, w) in G.out_edges(birther)])
    assert birther != dier
    if birther in mutants:
      mutants.add(dier)
    elif dier in mutants:
      mutants.remove(dier)
    
    steps += 1
  return steps


def sample(fn, times):
  count = 0
  while count < times:
    if (ans := fn()) is not None:
      yield ans
      count += 1

def style(plot):
  fig = plt.gcf()
  # fig.patch.set_alpha(0)
  # Add a border around the plot
  # ax = plt.gca()
  for i, ax in enumerate(fig.get_axes()):
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Customize the border color and thickness
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.grid(False)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, [['$r = 1.1$', '$r = 2$'][i]], title='')

  # Remove legend title.
  # handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles=handles[1:], labels=labels[1:])


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

def vortex_graph(N):
  assert N > 3
  G = nx.DiGraph()

  down = N//2
  up = down + N%2
  # print(down, up)

  for b in range(2, 1+down):
    G.add_edge(-1, -b)
    G.add_edge(-b, +1)

  for b in range(2, 1+up):
    G.add_edge(+1, +b)
    G.add_edge(+b, -1)

  # nx.draw(G)
  # nx.draw(
  #   G,
  #   pos=nx.circular_layout(G),
  #   with_labels=False,
  #   connectionstyle="arc3,rad=0.1",
  # )
  plt.show()
  return G

def vortex_fixation_time(N, samples=1000, overwrite=True, use_existing_file=True):
  Ns = list(range(4, N+1))
  file_name = f"data/vortex-graph-estimated-N-vs-ft-{N}.pkl"
  Rs = (1.1, 100)
  if use_existing_file:
    df = pd.read_pickle(file_name)
  else:
    data = []
    for N in Ns:
      print(N)
      G = vortex_graph(N)
      for r in Rs: 
        for time in sample(lambda: trial_cftime(G, None, r), samples):
          data.append((N, r, time))

    df = pd.DataFrame(data, columns=["Population size", "r", "Fixation time"])
    if overwrite:
      df.to_pickle(file_name)

  df = df[df['r'].isin(Rs)]
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
  ax.legend(handles, [f'${r = }$' for r in Rs], title='')
  ax.set(xlabel='Population size, $N$', ylabel='Fixation time, $T$')

  style(plot)

  # plt.ticklabel_format(useOffset=False)
  plt.xscale('log')
  plt.yscale('log')
  from matplotlib.ticker import ScalarFormatter
  ax.xaxis.set_major_formatter(ScalarFormatter()) 
  ax.xaxis.set_minor_formatter(ScalarFormatter())
  ax.set_xticks([n for n in Ns if n != 10 and n % 10 == 0])
  plt.savefig(f'charts/vortex-graph-estimated-N-vs-ft-{N}.png', dpi=300, bbox_inches="tight")
  plt.show()

def fan_graph(B):
  G = nx.DiGraph()

  for b in range(1, B+1):
    G.add_edge(0, -b)
    G.add_edge(-b, +b)
    G.add_edge(+b, 0)

  return G

def fan_fixation_time(B, samples=1000, overwrite=True, use_existing_file=True):
  Bs = list(range(1, B+1))
  file_name = f"data/fan-graph-estimated-B-vs-ft-{B}.pkl"
  Rs = (1.1, 100)
  if use_existing_file:
    df = pd.read_pickle(file_name)
  else:
    data = []
    for B in Bs:
      print(B)
      G = fan_graph(B)
      for r in Rs:
        for time in sample(lambda: trial_cftime(G, None, r), samples):
          data.append((B, r, time))

    df = pd.DataFrame(data, columns=["Blades", "r", "Fixation time"])
    if overwrite:
      df.to_pickle(file_name)

  df = df[df['r'].isin(Rs)]
  df['N'] = 2*df['Blades'] + 1
  plot = sns.lineplot(
    # kind="line",
    data=df,
    x="N",
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
  ax.legend(handles, [f'${r = }$' for r in Rs], title='')
  ax.set(xlabel='Population size, $N$', ylabel='Fixation time, $T$')

  style(plot)
  ax = plt.gca()
  # plt.gca().set_xticks([2*B+1 for B in Bs])

  plt.xscale('log')
  plt.yscale('log')
  # ax.ticklabel_format(useOffset=False, style='plain')
  from matplotlib.ticker import ScalarFormatter
  plt.gca().xaxis.set_major_formatter(ScalarFormatter()) 
  plt.gca().xaxis.set_minor_formatter(ScalarFormatter())
  ax.set_xticks([2*B+1 for B in Bs if (2*B+1) % 10 == 0 and ()])
  plt.savefig(f'charts/fan-graph-estimated-B-vs-ft-{B}.png', dpi=300, bbox_inches="tight")
  plt.show()

  # G = networkx_to_pepa_format(G_nx)
  # degs = g2degs(G)
  # mat = g2mat(G, degs, 1)
  # cfts = cftime(mat)
  # print(cfts)

  # ys = []
  # N = 4
  # for G in yield_all_digraph6(Path(f"data/directed/direct{N}.d6")):
  #   if not nx.is_strongly_connected(G): continue
  #   biased_stationary(G)
    # stationary_probabilities()
  #   # if max_degree(G) / min_degree(G) > 5/4: continue
  #   # eps = epsilon(G)
  #   # print(eps)
  #   print(biased_stationary(G))
  #   print(fixation_probabilities(G))
  #   # ys.append(eps)
  
  # # plt.hist(ys, bins=10)
  # # plt.show()

  # does_fp_only_depend_on_incoming_degrees()
  # monotonicity_of_ft(4)
  # fps()
  # G = custom_graph()
  # tournament()
  # N = 7
  # tournament((
  #   G
  #   for G in yield_all_digraph6(Path(f"data/eulerian/euler{N}.d6"))
  #   if nx.is_strongly_connected(G)
  # ), type="eulerian", invert=False)
  # tournament(yield_all_digraph6(Path(f"data/eulerian/euler{N}.d6")))
  # tournament(yield_all_digraph6(Path(f"data/eulerian-oriented/eulerian{N}.d6")))
  # tournament(generate_eulerians(N))
  # plot_potentials()
  # plot_fans()
  # plot_r_potential_boundaries()
  # custom_graph_absorptions()
  # plot_drifts()
  # monotonicity_of_ft(4)

if __name__ == '__main__':
  # fp_vs_cftime(5)
  # plot_fans(20)
  # two_layer_fixation_time(20, samples=1000)
  # example_monotonicity_of_ft(4)
  # monotonicity_of_ft(4)
  # fan_fixation_time(20)
  # vortex_graph(3)
  vortex_fixation_time(20)

  ...
  # Plotting with Seaborn