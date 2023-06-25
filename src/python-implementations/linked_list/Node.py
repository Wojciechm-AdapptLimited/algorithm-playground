from __future__ import annotations

from typing import TypeVar, Generic

T = TypeVar('T')


class Node(Generic[T]):
    value: T
    prev_node: Node[T] | None
    next_node: Node[T] | None

    def __init__(self, value: T) -> None:
        self.value = value
        self.prev_node = None
        self.next_node = None

    def __repr__(self) -> str:
        return str(self.value)

    def __eq__(self, other: Node[T]) -> bool:
        return self.value == other.value
