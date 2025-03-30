from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Generator

import typer

app = typer.Typer()


def value2int(value: int | float | None) -> int | None:
    """Converts a float to an int if it's a whole number.
    This is to ensure we only allow integers as a result of division even
    if the ultimate result is an int.
    """
    if isinstance(value, (float, int)) and value.is_integer():
        return int(value)
    return None


# Define possible operations
OPS = (
    (lambda x, y: x + y, "({x} + {y})"),
    (lambda x, y: x * y, "({x} * {y})"),
    (lambda x, y: x - y, "({x} - {y})"),
    (lambda x, y: y - x, "({y} - {x})"),
    (lambda x, y: value2int(x / y if y != 0 else None), "({x} / {y})"),
    (lambda x, y: value2int(y / x if x != 0 else None), "({y} / {x})"),
)


def generate_numbers() -> list[int]:
    # Generate six random numbers: just like Countdown.
    n_large = random.randint(0, 4)
    small = list(range(1, 11)) + list(range(1, 11))
    return random.sample([25, 50, 75, 100], n_large) + random.sample(small, 6 - n_large)


def generate_target() -> int:
    # Generate a random target number between 100 and 999
    return random.randint(100, 999)


@dataclass(order=True)
class State:
    expr: str = field(compare=False, default="")
    value: int = field(compare=False, default=0)
    diff: int = field(compare=True, default=10**10)
    remaining: list[int] = field(compare=False, default_factory=lambda: [])


class Solver:
    def __init__(self, numbers: list[int], target: int) -> None:
        self.target = target
        self.heap: list[State] = []
        node = State(
            expr=str(numbers[0]),
            value=numbers[0],
            diff=abs(self.target - numbers[0]),
            remaining=numbers[1:],
        )
        heapq.heappush(self.heap, node)

    def _apply_operations(self, node: State) -> Generator[State, None, None]:
        for index, number in enumerate(node.remaining):
            remaining = node.remaining[:index] + node.remaining[index + 1 :]
            for func, s in OPS:
                value = func(node.value, number)
                if value is not None:
                    yield State(
                        expr=s.format(x=node.expr, y=number),
                        value=value,
                        diff=abs(self.target - value),
                        remaining=remaining,
                    )

    def solve(self) -> State:
        best_node = State()
        while self.heap:
            node = heapq.heappop(self.heap)
            if node.diff == 0:
                return node
            for new_node in self._apply_operations(node):
                heapq.heappush(self.heap, new_node)
            best_node = min(best_node, node)
        return best_node


@app.command()
def demo() -> None:
    """Runs a demo of the number puzzle solver."""
    numbers = generate_numbers()
    target = generate_target()

    solver = Solver(numbers, target)
    node = solver.solve()

    print(f"Target: {target}")
    print(f"Numbers: {', '.join(map(str, sorted(numbers, reverse=True)))}")
    print(f"Best expression: {node.expr} = {node.value}")
    print(f"Difference from target: {node.diff}")
    print(
        f"Remaining numbers: [{', '.join(map(str, node.remaining)) if node.remaining else ' '}]"
    )


@app.command()
def play() -> None:
    """Starts a playable version of the number puzzle."""
    numbers = generate_numbers()
    target = generate_target()

    print(f"Target: {target}")
    print(f"Numbers: {', '.join(map(str, sorted(numbers, reverse=True)))}")

    while True:
        expression = input("Enter your expression (or type 'quit'): ").strip()
        if expression.lower() == "quit":
            break
        try:
            # Basic evaluation - be cautious with arbitrary user input in real applications
            result = eval(expression)
            if isinstance(result, int):
                print(f"Your expression evaluates to: {result}")
                difference = abs(target - result)
                if difference == 0:
                    print("Congratulations! You hit the target!")
                    break
                else:
                    print(f"You are {difference} away from the target.")
            else:
                print("Result must be an integer.")
        except (SyntaxError, NameError, ZeroDivisionError, TypeError) as e:
            print(f"Invalid expression: {e}")


@app.command()
def solve(
    target: int, numbers: list[int] = typer.Argument(..., help="The list of numbers")
) -> None:
    """Solves the number puzzle for a given target and set of numbers."""
    solver = Solver(numbers, target)
    node = solver.solve()

    print(f"Target: {target}")
    print(f"Numbers: {', '.join(map(str, sorted(numbers, reverse=True)))}")
    print(f"Best expression: {node.expr} = {node.value}")
    print(f"Difference from target: {node.diff}")
    print(
        f"Remaining numbers: [{', '.join(map(str, node.remaining)) if node.remaining else ' '}]"
    )


if __name__ == "__main__":
    app()
