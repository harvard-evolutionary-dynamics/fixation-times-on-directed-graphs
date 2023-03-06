from __future__ import annotations
import random
from decimal import Decimal
from networkx import DiGraph, Graph

import networkx as nx

from sortedcontainers import SortedList

from enum import Enum
from typing import Final, Generator, Dict, Set, List


class Type(Enum):
  WILD = "wild"
  MUTANT = "mutant"


class Moran:
  """
  Birth-Death Moran Process in structured population.
  Constant reproductive fitness: wild type has relative fitness 1 and mutants
  have relative fitness r.

  Note: evolved graph must be accessed from this object rather than trying to hold
  a reference to it.
  """
  TYPE_ATTRIBUTE_NAME: Final[str] = "type"
  graph: DiGraph
  r: Decimal
  _type_to_nodes: Dict[Type, SortedList[int]]
  _out_neighbors: Final[Dict[int, List[int]]]

  def __init__(self, graph: Graph, r: Decimal) -> None:
    Moran._check_population_types(graph)
    Moran._check_reproductive_fitness(r)
    self.graph = graph.to_directed()
    self.r = r
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

  def evolve(self, steps: int) -> Generator[Moran]:
    for _ in range(steps):
      # self._step()
      self._unoptimized_step()
      yield self

  def _step(self) -> None:
    birth_node_type: Type = random.choices(
      population=list(Type),
      weights=[
        1*len(self._type_to_nodes[Type.WILD]),
        self.r*len(self._type_to_nodes[Type.MUTANT]),
      ]
    )[0]
    possible_birthing_nodes = self._type_to_nodes[birth_node_type]
    birth_idx = random.randint(0, len(possible_birthing_nodes)-1)
    birth_node = possible_birthing_nodes[birth_idx]

    if not (possible_locations_to_birth := self._out_neighbors[birth_node]):
      return

    death_node = random.choice(possible_locations_to_birth)
    original_death_node_type: Type = self.graph.nodes()[death_node][self.TYPE_ATTRIBUTE_NAME]
    new_death_node_type: Type = birth_node_type
    if new_death_node_type == original_death_node_type:
      return

    nx.set_node_attributes(G=self.graph, values={ death_node: new_death_node_type }, name=self.TYPE_ATTRIBUTE_NAME)

    # Manual tracking
    self._type_to_nodes[original_death_node_type].remove(value=death_node)
    self._type_to_nodes[new_death_node_type].add(value=death_node)


  def _unoptimized_step(self) -> None:
    """Note: This is unoptimized."""
    birth_node, *_ = random.choices(
      population=list(self.graph.nodes().keys()),
      weights=(
        self._type_to_weight(data[self.TYPE_ATTRIBUTE_NAME])
        for _, data in self.graph.nodes(data=True)
      )
    )

    birth_node_type = self.graph.nodes()[birth_node][self.TYPE_ATTRIBUTE_NAME]

    if not (possible_locations_to_birth := list(self.graph[birth_node].keys())):
      return

    death_node = random.choice(possible_locations_to_birth)
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