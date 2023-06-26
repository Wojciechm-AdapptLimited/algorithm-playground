from . import Node, T
from typing import Generic


class LinkedList(Generic[T]):
    head: Node[T] | None

    def get_last_node(self) -> Node[T] | None:
        if self.head is None:
            return None

        if self.head.next_node is None:
            return self.head

        prev_node: Node[T] | None = None
        cur_node: Node[T] | None = self.head

        while cur_node is not None:
            prev_node = cur_node
            cur_node = cur_node.next_node

        return prev_node

    def push_back(self, value: T) -> None:
        new_node: Node[T] = Node(value)

        if self.head is None:
            self.head = new_node
            return

        prev_node: Node[T] = self.get_last_node()

        prev_node.next_node = new_node
        new_node.prev_node = prev_node

    def push_front(self, value: T) -> None:
        new_node: Node[T] = Node(value)

        if self.head is None:
            self.head = new_node
            return

        self.head.prev_node = new_node
        new_node.next_node = self.head
        self.head = new_node


