import networkx as nx
from typing import Callable, Iterable, Generator, TypeVar
from pathlib import Path

from networkx.readwrite.graph6 import data_to_n

_T = TypeVar("_T")
_Predicate = Callable[[_T], object]

def takewhile_inclusive(predicate: _Predicate[_T], it: Iterable[_T]) -> Generator[_T, None, None]:
  for x in it:
    yield x
    if not predicate(x): break

def digraph6_to_data(string):
  """Convert digraph6 character sequence to 6-bit integers."""
  v = [ord(c)-63 for c in string]
  if len(v) > 0 and (min(v) < 0 or max(v) > 63):
    return None
  return v

# def data_to_n(data):
#   """Read initial one-, four- or eight-unit value from graph6
#   integer sequence.
# 
#   Return (value, rest of seq.)"""
#   if data[0] <= 62:
#     return data[0], data[1:]
#   if data[1] <= 62:
#     return (data[1]<<12) + (data[2]<<6) + data[3], data[4:]
#   return ((data[2]<<30) + (data[3]<<24) + (data[4]<<18) +
#         (data[5]<<12) + (data[6]<<6) + data[7], data[8:])

def parse_digraph6(string):
  """Read a simple directed graph in digraph6 format from string.

  Parameters
  ----------
  string : string
    Data in digraph6 format

  Returns
  -------
  G : Graph

  Raises
  ------
  NetworkXError
      If the string is unable to be parsed in digraph6 format

  References
  ----------
  Graph6 specification:
  http://cs.anu.edu.au/~bdm/data/formats.txt for details.
  """
  def bits():
      """Return sequence of individual bits from 6-bit-per-value
      list of data values."""
      for d in data:
          for i in [5,4,3,2,1,0]:
              yield (d>>i)&1

  HEADER = '>>digraph6<<'
  if string.startswith(HEADER):
    string = string[len(HEADER):]
  
  assert string[0] == "&", string
  string = string[1:]
  data = digraph6_to_data(string)
  n, data = data_to_n(data)
  nd = (n**2 + 5) // 6
  if len(data) != nd:
    raise nx.NetworkXError(\
      'Expected %d bits but got %d in digraph6' % (n**2, len(data)*6))

  G=nx.DiGraph()
  G.add_nodes_from(range(n))
  for (i,j),b in zip([(i,j) for i in range(n) for j in range(n)], bits()):
    if b:
      G.add_edge(i,j)

  return G


def yield_all_digraph6(path: Path):
    """Read simple directed graphs in digraph6 format from path.

    Parameters
    ----------
    path : file or string
       File or filename to write.

    Returns
    -------
    G : generator of nx.DiGraphs

    Raises
    ------
    NetworkXError
        If the string is unable to be parsed in digraph6 format

    References
    ----------
    Digraph6 specification:
    http://cs.anu.edu.au/~bdm/data/formats.txt for details.
    """
    with path.open() as f:
      for line in f.readlines():
        line = line.strip()
        if not len(line):
          continue
        yield parse_digraph6(line)

def networkx_to_pepa_format(G: nx.Graph):
  N, M = G.number_of_nodes(), G.number_of_edges()
  return [(N, M), *G.edges()]