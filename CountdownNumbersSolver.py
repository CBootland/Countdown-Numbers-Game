"""
Module Docstring:
Example usage of the countdown_solve function
"""

import random

from CountdownNumbersGame import countdown_solve


# Example usage with solutions printed:
countdown_solve([100, 50, 25, 7, 6, 1], 827)

# Example of using brute force (not recommended):
countdown_solve([100, 25, 7, 6, 1], 851, brute_force=True)

# If the solutions are to be used and not printed:
solutions = countdown_solve([100, 50, 25, 7, 6, 1], 827, print_results=False)

# For a random instance of the game and any solutions:
number_of_large = random.randrange(5)
number_of_small = 6 - number_of_large
large = [100, 75, 50, 25]
small = [10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
numbers = []
for _ in range(number_of_large):
    numbers.append(large.pop(random.randrange(len(large))))
for _ in range(number_of_small):
    numbers.append(small.pop(random.randrange(len(small))))
result = random.randrange(1, 1000)

print(f'Random instance: {numbers} {result}')
solutions = countdown_solve(numbers, result)
