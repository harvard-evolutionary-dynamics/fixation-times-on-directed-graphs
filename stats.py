from networkx import Graph
from decimal import Decimal
from moran import Moran, Type

def proportion_of_mutants(graph: Graph) -> Decimal:
  return sum(
    int(data.get(Moran.TYPE_ATTRIBUTE_NAME) == Type.MUTANT)
    for _, data in graph.nodes(data=True)
  ) / Decimal(graph.number_of_nodes())

def is_absorbed(graph: Graph) -> bool:
  return proportion_of_mutants(graph) in (0, 1)