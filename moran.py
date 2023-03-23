from __future__ import annotations
import random
from decimal import Decimal
from networkx import DiGraph, Graph
import itertools

import networkx as nx

from sortedcontainers import SortedList
from dataclasses import dataclass

from enum import Enum
from typing import Any, Final, Generator, Dict, Set, List, Optional, Tuple, overload


class Type(Enum):
  WILD = 0
  MUTANT = 1

Edge = Tuple[int, int]
Node = int

@dataclass
class Action:
  birth_node: int
  death_node: int

@dataclass
class ActionState:
  state: Moran
  action: Optional[Action] = None

class Moran:
  """
  Birth-Death Moran Process in structured population.
  Constant reproductive fitness: wild type has relative fitness 1 and mutants
  have relative fitness r.

  Note: evolved graph must be accessed from this object rather than trying to hold
  a reference to it.
  """
  TYPE_ATTRIBUTE_NAME: Final[str] = "type"
  IS_BIRTHING_ATTRIBUTE_NAME: Final[str] = 'is_birthing'
  IS_DYING_ATTRIBUTE_NAME: Final[str] = 'is_dying'
  IS_ACTIVE_EDGE_ATTRIBUTE_NAME: Final[str] = 'is_active_edge'
  graph: DiGraph
  r: Decimal
  rng: random.Random
  _type_to_nodes: Dict[Type, SortedList[int]]
  _out_neighbors: Final[Dict[int, List[int]]]
  dying_node: Optional[int]
  birthing_node: Optional[int]

  def __init__(self, graph: Graph, r: Decimal, rng: Optional[random.Random] = None) -> None:
    Moran._check_population_types(graph)
    Moran._check_reproductive_fitness(r)
    self.graph = graph.to_directed()
    self.r = r
    self.rng = rng or random.Random()
    self.birthing_node = None
    self.dying_node = None
    self._type_to_nodes = {
      type: SortedList([
        node 
        for node, data in self.graph.nodes(data=True)
        if data[self.TYPE_ATTRIBUTE_NAME] == type
      ])
      for type in Type
    }
    self._out_neighbors = {
      node: [neighbor for neighbor in self.graph[node].keys()]
      for node in self.graph.nodes()
    }

  def action_state(self, action: Optional[Action] = None) -> ActionState:
    return ActionState(state=self, action=action)

  def evolve(self, steps: Optional[int] = None, include_initial: bool = False) -> Generator[ActionState]:
    if include_initial:
      yield self.action_state()

    for _ in (range(steps) if steps else itertools.count()):
      action = self._step()
      yield self.action_state(action)


  @overload
  def _set_graph_attribute(self, article: Edge, name: str, value: Any) -> None: ...
  @overload
  def _set_graph_attribute(self, article: Node, name: str, value: Any) -> None: ...
  @overload
  def _set_graph_attribute(self, article: None, name: str, value: Any) -> None: ...
  def _set_graph_attribute(self, article: Optional[Node | Edge], name: str, value: Any) -> None:
    if article is None: return
    fn = nx.set_node_attributes if isinstance(article, Node) else nx.set_edge_attributes
    fn(G=self.graph, name=name, values={ article: value })


  def _reset_birth_death_attributes(self) -> None:
    self._set_graph_attribute(self.birthing_node, self.IS_BIRTHING_ATTRIBUTE_NAME, False)
    self._set_graph_attribute(self.dying_node, self.IS_DYING_ATTRIBUTE_NAME, False)
    self._set_graph_attribute((self.birthing_node, self.dying_node), self.IS_ACTIVE_EDGE_ATTRIBUTE_NAME, False)
    self.birthing_node = None
    self.dying_node = None


  def _step(self) -> Optional[Action]:
    self._reset_birth_death_attributes()

    birth_node_type: Type = self.rng.choices(
      population=list(Type),
      weights=[
        1*len(self._type_to_nodes[Type.WILD]),
        self.r*len(self._type_to_nodes[Type.MUTANT]),
      ]
    )[0]
    possible_birthing_nodes = self._type_to_nodes[birth_node_type]
    birth_idx = self.rng.randint(0, len(possible_birthing_nodes)-1)
    birth_node = possible_birthing_nodes[birth_idx]
    self.birthing_node = birth_node
    self._set_graph_attribute(self.birthing_node, self.IS_BIRTHING_ATTRIBUTE_NAME, True)

    if not (possible_locations_to_birth := self._out_neighbors[birth_node]):
      return None

    death_node = self.rng.choice(possible_locations_to_birth)
    original_death_node_type: Type = self.graph.nodes()[death_node][self.TYPE_ATTRIBUTE_NAME]
    new_death_node_type: Type = birth_node_type
    self.dying_node = death_node
    self._set_graph_attribute(self.dying_node, self.IS_DYING_ATTRIBUTE_NAME, True)
    self._set_graph_attribute((self.birthing_node, self.dying_node), self.IS_ACTIVE_EDGE_ATTRIBUTE_NAME, True)

    if new_death_node_type != original_death_node_type:
      self._set_graph_attribute(death_node, self.TYPE_ATTRIBUTE_NAME, new_death_node_type)
      # Manual tracking
      self._type_to_nodes[original_death_node_type].remove(value=death_node)
      self._type_to_nodes[new_death_node_type].add(value=death_node)

    return Action(birth_node=birth_node, death_node=death_node)


  def _unoptimized_step(self) -> None:
    """Note: This is unoptimized."""
    birth_node, *_ = self.rng.choices(
      population=list(self.graph.nodes().keys()),
      weights=(
        self._type_to_weight(data[self.TYPE_ATTRIBUTE_NAME])
        for _, data in self.graph.nodes(data=True)
      )
    )

    birth_node_type = self.graph.nodes()[birth_node][self.TYPE_ATTRIBUTE_NAME]

    if not (possible_locations_to_birth := list(self.graph[birth_node].keys())):
      return

    death_node = self.rng.choice(possible_locations_to_birth)
    nx.set_node_attributes(G=self.graph, values={ death_node: birth_node_type }, name=self.TYPE_ATTRIBUTE_NAME)



  def _type_to_weight(self, type: Type) -> Decimal:
    if type == Type.WILD: return 1
    if type == Type.MUTANT: return self.r
    raise ValueError(f"Unknown type: '{type}'")

  @classmethod
  def _check_reproductive_fitness(cls, r: Decimal) -> None:
    assert r >= 0

  @classmethod
  def _check_population_types(cls, graph: Graph) -> None:
    for _, data in graph.nodes(data=True):
      assert "type" in data
      assert isinstance(data[cls.TYPE_ATTRIBUTE_NAME], Type)