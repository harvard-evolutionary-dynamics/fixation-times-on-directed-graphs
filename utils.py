from typing import Callable, Iterable, Generator, TypeVar

_T = TypeVar("_T")
_Predicate = Callable[[_T], object]

def takewhile_inclusive(predicate: _Predicate[_T], it: Iterable[_T]) -> Generator[_T, None, None]:
  for x in it:
    yield x
    if not predicate(x): break