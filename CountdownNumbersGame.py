"""A set of methods to solve the Countdown numbers game.

Functions
---------
countdown_solve
    Solves an instance of the Countdown numbers game

Examples
--------
Example usage with solutions printed:
countdown_solve([100, 50, 25, 7, 6, 1], 827)

Example of using brute force (not recommended):
countdown_solve([100, 25, 7, 6, 1], 851, brute_force=True)

If the solutions are to be used and not printed:
solutions = countdown_solve([100, 50, 25, 7, 6, 1], 827,
                            print_results=False)

For a random instance of the game and any solutions:
import random
number_of_large = random.randrange(5)
number_of_small = 6 - number_of_large
large = [100, 75, 50, 25]
small = [10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
large_numbers = [large.pop(random.randrange(len(large)))
                 for _ in range(number_of_large)]
small_numbers = [small.pop(random.randrange(len(small)))
                 for _ in range(number_of_small)]
numbers = large_numbers + small_numbers
result = random.randrange(1, 1000)
print(f'Random instance: {numbers} {result}')
countdown_solve(numbers, result)
"""

# Author: Carl Bootland


import math
from fractions import Fraction


INVERSE = ({'+': '-', '-': '+', '*': '/', '/': '*', '!': '!'},
           {'+': '-', '-': '-', '*': '/', '/': '/', '!': '!'},)
ORDER = {'+': [1, 0], '-': [0, 1], '*': [1, 0], '/': [0, 1], '!': [1, 0]}


def countdown_solve(numbers: list, result: int, brute_force: bool = False,
                    print_results: bool = True) -> list:
    """Find all solutions to the countdown numbers game.

    Parameters
    ----------
    numbers : list of numerical types
        The numbers that can each be used at most once
    result : int
        The target to be reached
    brute_force : bool, optional
        A flag used to perform a brute force search (default is False)
    print_results : bool, optional
        A flag used to print any solutions (default is True)

    Returns
    -------
    list
        A list of strings detailing each distinct solution

    Two solutions are only considered distinct if one cannot be
    transformed into the other by reordering the operations performed.

    Note: we allow intermediate results to be rationals and not
    necessarily integers.

    This method uses a meet-in-the middle arrpoach which works when 6 or
    fewer numbers are given as in the gameshow but may not show all
    solutions with more numbers; this is indicated if it occurs.
    """
    if result < 0 or any([number < 0 for number in numbers]):
        raise ValueError("Must use non-negative values.")
    if brute_force:
        print("Warning: This is very slow!")
        return _brute_force_search(numbers, result, print_results)
    n = len(numbers)
    if n > 6:
        print("This may not give all solutions.")
    solutions = _meet_in_the_middle(numbers, result, n, n // 2, [],
                                    print_results)
    if n > 1:
        solutions.extend(_meet_in_the_middle(numbers, result, n, n // 2 - 1,
                                             solutions, print_results))
    return solutions


def _meet_in_the_middle(numbers: list, result: int, n: int, m: int,
                        solutions: list, print_results: bool) -> list:
    """Apply the meet-in-the-middle approach to find solutions.

    Parameters
    ----------
    numbers : list of numerical types
        The numbers that can each be used at most once
    result : int
        The target to be reached
    n : int
        Length of `numbers`
    m : int
        Number of elements from `numbers` to be used with `result`
    solutions : list
        List of previously found solutions
    print_results : bool
        A flag used to print any solutions

    Returns
    -------
    list
        A list of all solutions found

    Compute all intermediate values reached by using any `m` elements of
    `numbers` together with `result` and look for a collision with those
    values reached by using the remaining `n`-`m` elements from
    `numbers`.
    If a collision is found which uses `result`, determine the
    corresponding method to compute `result` using the elements of
    `numbers`.
    """

    for c in range(math.comb(n, m)):
        first_indices = _binomial_subset(c, n, m)
        first_set = [numbers[i] for i in first_indices] + [result]
        second_set = [numbers[i] for i in range(n) if i not in first_indices]
        firsts = _compute_all(first_set, result)
        seconds = _compute_all(second_set, result)
        for key in firsts.keys():
            if key in seconds.keys():
                for method in firsts[key]:
                    if method[1]:
                        first_method = _rewrite_method(first_set, method[0],
                                                       result)
                        for other_method in seconds[key]:
                            second_method = _rewrite_method(second_set,
                                                            other_method[0],
                                                            result)
                            new_method = _merge_methods(first_set,
                                                        first_method,
                                                        second_set,
                                                        second_method,
                                                        result)
                            if new_method and new_method not in solutions:
                                solutions.append(new_method)
                                if print_results:
                                    print(new_method)
    return solutions


def _binomial_subset(c: int, n: int, k: int) -> list:
    """Return the subset of {0, 1, ..., n-1} of size k indexed by c."""

    if c < 0 or c >= math.comb(n, k):
        raise ValueError("c is out of range")
    if k == 0 or n == 0:
        return []
    elif c < math.comb(n - 1, k):
        return _binomial_subset(c, n - 1, k)
    return _binomial_subset(c - math.comb(n - 1, k), n - 1, k - 1) + [n - 1]


def _compute_expression(numbers: list, expression: list, result: int,
                        separate_sign: bool = False):
    """Compute the value of the given expression.

    Parameters
    ----------
    numbers : list of numerical types
        The numbers to be computed with
    expression : list of 3-tuples
        The encoding of the expression to be computed
    result : int
        The target to be reached
    separate_sign : bool:
        A flag used for splitting the sign from value (defaul is true)

    Returns
    -------
    int or Fraction, [optional bool if sepearate_sign,] bool
        value computed, [optional True if negative else False,]
        whether `result` was used in the computation

    Compute the value of `expression` using `numbers` and return this
    value [optional with its sign separated] along with whether `result`
    was used in the calculation or not.
    """

    if result in numbers:
        status = True
    else:
        status = False
    for expr in expression:
        numbers, status, used = _compute_step(numbers, expr[0], expr[1],
                                              expr[2], result, status)
        if used:
            result = numbers[-1]
        if numbers == 'DivisionError':
            if not separate_sign:
                return numbers, status
            return numbers, False, status
    if not separate_sign:
        return numbers[0], status
    return abs(numbers[0]), numbers[0] < 0, status


def _compute_step(numbers: list, index1: int, index2: int, operation: list,
                  result: int, status: bool):
    """Compute one step of an expression.

    Parameters
    ----------
    numbers : list of numerical types
        The numbers to be computed with
    index1 : int
        The index of the first operand in `numbers`
    index2 : int
        The index of the second operand in `numbers`
    operation : str
        The operation to be performed
    result : int
        The target to be reached
    status : bool
        Whether `result` has already been used in the computation

    Returns
    -------
    list, bool, bool
        The new set of numbers that can be used, whether `result` has
        been used at any point in the current calculation, whether
        `result` has been used in this step of the calculation

    Compute the operation using the indices of `numbers` to be used.
    The `!` operation simply drops the second operand so that we do not
    have to use all the numbers.
    """

    new_numbers = numbers[:]
    number1 = new_numbers.pop(index1)
    number2 = new_numbers.pop(index2)
    used = False
    if result in {number1, number2}:
        used = True
    if operation == '+':
        new_numbers.append(number1 + number2)
    elif operation == '-':
        new_numbers.append(number1 - number2)
    elif operation == '*':
        if {number1, number2} == {0, result}:
            status = False
        new_numbers.append(number1 * number2)
    elif operation == '/':
        if {number1, number2} == {0, result}:
            status = False
        if number2 == 0:
            return 'DivisionError', status, used
        new_numbers.append(Fraction(number1, number2))
    elif operation == '!':
        if number2 == result:
            status = False
        new_numbers.append(number1)
    return new_numbers, status, used


def _rewrite_method(numbers: list, expression: list, result: int) -> list:
    """Rewrite the expression to use actual numbers.

    Parameters
    ----------
    numbers : list of numerical types
    expression : list of 3-tuples
        Descriction of method used in the computation in terms of the
        indices of `numbers`
    result : int
        The target to be reached

    Returns
    -------
    list of 4-tuples
        The new description of `expression`

    Rewrite the expression so that the actual numbers being computed
    as well as their indices are given.

    Furthermore, we ensure all intermediate values encountered are
    positive.
    """

    method = []
    negatives = [False] * len(numbers)
    status = True
    for expr in expression:
        index1 = expr[0]
        index2 = expr[1]
        negative1 = negatives.pop(index1)
        negative2 = negatives.pop(index2)
        if index2 >= index1:
            index2 += 1
        negative = False
        operation = expr[2]
        if negative1 ^ negative2:
            if operation == '-':
                operation = '+'
            elif operation == '+':
                operation = '-'
            if (operation in '+-' and negative1) or operation in '*/':
                negative = True
        if negative1 and negative2 and operation in '+-':
            negative = True
        step = [[numbers[index1], expr[0]],
                [numbers[index2], expr[1]],
                operation]
        numbers, status, used = _compute_step(numbers, expr[0], expr[1],
                                              operation, result, status)
        if numbers[-1] < 0:
            numbers[-1] *= -1
            negative = not negative
            if step[0][1] <= step[1][1]:
                step[1][1] += 1
            else:
                step[0][1] -= 1
            step[0], step[1] = step[1], step[0]
        negatives.append(negative)
        method.append(step + [[numbers[-1], len(numbers) - 1]])
    return method


def _div_and_rem(number: int, divisor: int) -> tuple[int, int]:
    """Return the quotient and remainder"""
    rem = number % divisor
    div = number // divisor
    return div, rem


def _compute_part1(counter: int, num_exc: int,
                   n: int) -> tuple[list, int, list]:
    """Subroutine of _compute_all"""
    ops = ['+', '-', '*', '/']
    operations = ['!'] * num_exc
    tot = math.comb(n, num_exc)
    parts = [tot]
    y = counter
    for j in range(n - 1 - num_exc):
        y, t = _div_and_rem(y, 4)
        operations.append(ops[t])
        if t == 3:
            tot *= (n - num_exc - j) * (n - 1 - num_exc - j)
            parts.append((n - num_exc - j) * (n - 1 - num_exc - j))
        else:
            tot *= math.comb(n - num_exc - j, 2)
            parts.append(math.comb(n - num_exc - j, 2))
    return operations, tot, parts


def _compute_part2(counter: int, parts: list, operations: list, num_exc: int,
                   n: int, numbers: list, result: int, D: dict) -> dict:
    """Subroutine of _compute_all"""
    x = counter
    x, s = _div_and_rem(x, parts[0])
    elements = _binomial_subset(s, n, num_exc)
    if num_exc == 0:
        expression = []
    elif num_exc == 1:
        if elements[0] == n - 1:
            expression = [(n - 2, n - 2, '!')]
        else:
            expression = [(n - 1, elements[0], '!')]
    else:
        expression = [(elements[0], elements[-1] - 1, '!')]
        for k in range(1, num_exc - 1):
            expression.append((n - 1 - k, elements[k] - k, '!'))
        expression.append((n - num_exc - 1, n - num_exc - 1, '!'))
    for i in range(n - 1 - num_exc):
        x, r = _div_and_rem(x, parts[i + 1])
        if operations[num_exc + i] == '/':
            v, u = _div_and_rem(r, n - num_exc - i)
        else:
            pair = _binomial_subset(r, n - i, 2)
            u = pair[0]
            v = pair[1] - 1
        expression.append((u, v, operations[num_exc + i]))
    c, sgn, status = _compute_expression(numbers, expression, result, True)
    if c != 'DivisionError':
        if c in D.keys():
            if [expression, status] not in D[c]:
                D[c].append([expression, status])
        else:
            D[c] = [[expression, status]]
    return D


def _compute_all(numbers: list, result: int) -> dict[int or Fraction, str]:
    """Compute all possible values made from numbers.

    Parameters
    ----------
    numbers : list of numerical types
        The numbers that can each be used at most once
    result : int
        The target to be reached

    Returns
    -------
    dict

    Compute all possible values that can be obtained from the elements
    of `numbers` and place them in a dictionary whose keys are the
    computed values and whose data is the methods which gives that value
    and whether `result` was used in the computation.

    The allowed operations are +, -, *, / and ! where the operator !
    denotes simply dropping the second operand, allowing not all numbers
    to be used in the computation.

    If there are 3 or 4 elements in `numbers` then we use a pre-computed
    list of distinct methods otherwise we try every combination of
    operations which will include duplicates.
    """

    D = {}
    n = len(numbers)
    if n == 1:
        numbers.append(0)
        D[numbers[0]] = [[[(0, 0, '!')], numbers[0] == result]]
        return D
    for num_exc in range(n):
        for counter1 in range(4 ** (n - 1 - num_exc)):
            operations, tot, parts = _compute_part1(counter1, num_exc, n)
            for counter2 in range(tot):
                D = _compute_part2(counter2, parts, operations,
                                   num_exc, n, numbers, result, D)
    return D


def _brute_force_search(numbers: list, result: int, print_results: bool) -> list:
    """Perform a brute force search for result."""
    D = _compute_all(numbers, result)
    solutions = []
    if result in D.keys():
        for method in D[result]:
            new_method = _rewrite_method(numbers, method[0], result)
            new_method = [[step[0][0], step[1][0], step[2], step[3][0]]
                          for step in new_method]
            solution = _canonical_form(new_method, numbers)
            if solution and solution not in solutions:
                solutions.append(solution)
                if print_results:
                    print(solution)
    return solutions


def _merge_methods(numbers1: list, method1_original: list, numbers2: list,
                   method2: list, result: int) -> str:
    """Merge two methods with the same output.

    Parameters
    ----------
    numbers1 : list of numerical types
        The numbers `method1_original` computes with
    method1_original : list of 4-tuples
        The first method to compute the intermediate value
    numbers2 : list of numerical types
        The numbers `method2` computes with
    method2 : list of 4-tuples
        The second method to compute the intermediate value
    result : int
        The final target to be reached

    Returns
    -------
    str
        Human-readable method to compute the target `result`

    Merge two methods which compute the same intermediate value,
    `method1_original`  using the elements of `numbers1` and `method2`
    using the elements of `numbers2`, into a single method which
    computes `result` using the elements from both `numbers1` and
    `numbers2`.
    """

    method1 = method1_original[:]
    method = []
    length = len(numbers1)
    m = length + len(numbers2) - 1
    for step in method2:
        m -= 1
        method.append([[step[0][0], step[0][1] + length],
                       [step[1][0], step[1][1] + length],
                       step[2],
                       [step[3][0], m]])
    used_count = 0
    using_result = [[result, length - 1, used_count]]
    used_result = []
    method1_remove = []
    for step in method1:
        delta = 0
        if step[0][1] <= step[1][1]:
            delta = 1
        triple1 = step[0] + [used_count]
        triple2 = [step[1][0], step[1][1] + delta] + [used_count]
        if triple1 in using_result or triple2 in using_result:
            to_remove = []
            for index, triple in enumerate(using_result):
                if triple in [step[0] + [used_count],
                              [step[1][0], step[1][1] + delta] + [used_count]]:
                    to_remove.append(triple)
                else:
                    shift = 0
                    if step[0][1] < triple[1]:
                        shift += 1
                    if step[1][1] + delta < triple[1]:
                        shift += 1
                    using_result[index][1] -= shift
            for item in to_remove:
                using_result.remove(item)
                used_result.append(item)
            used_count += 1
            using_result.append(step[3] + [used_count])
        else:
            method1_remove.append(step)
            for index, triple in enumerate(using_result):
                shift = 0
                if step[0][1] < triple[1]:
                    shift += 1
                if step[1][1] + delta < triple[1]:
                    shift += 1
                using_result[index][1] -= shift
            method.append([step[0], step[1], step[2],
                           [step[3][0], step[3][1] + 1]])
    for item in method1_remove:
        method1.remove(item)
    used_count -= 1
    final_method = _invert_operations(method, method1, used_result, used_count)
    return _canonical_form(final_method, numbers1[:] + numbers2[:])


def _invert_operations(method: list, method1: list, used_result: list,
                       used_count: int) -> list:
    """Invert the opertions to recover the target.

    Parameters
    ----------
    method : list
        The method to be appended to
    method1 : list
        The method whose operations are to be inverted
    used_result : list
        List of the intermediate values that used `result` to be reached
    used_count : int
        How many operations have been applied to `result` that need inverting

    Returns
    -------
    list
        The final method which computes the target

    Invert the operations from `method1` so that we have a method for
    computing the target `result`.

    """

    while method1:
        step = method1.pop()
        delta = 0
        if step[0][1] <= step[1][1]:
            delta = 1
        if step[0] + [used_count] in used_result:
            inputs = [step[1], step[3]]
            method.append([inputs[ORDER[step[2]][0]],
                           inputs[ORDER[step[2]][1]],
                           INVERSE[0][step[2]],
                           step[0]])
        elif [step[1][0], step[1][1] + delta] + [used_count] in used_result:
            inputs = [step[0], step[3]]
            method.append([inputs[ORDER[step[2]][0]],
                           inputs[ORDER[step[2]][1]],
                           INVERSE[1][step[2]],
                           step[1]])
        used_count -= 1
    return [[step[0][0], step[1][0], step[2], step[3][0]] for step in method]


def _canonical_form(method: list, numbers: list) -> str:
    """Write the method as a string in canonical form.

    Parameters
    ----------
    method : list of 4-tuples
        Description of the method to compute the target
    numbers: list of numerical types
        The numbers to be computed with

    Returns
    -------
    str
        Human-readable method to compute the target using `numbers`

    Write the method in canonical form (larger values before smaller,
    + before -, * before /) and return this as a readable string or the
    empty string if the expression 0/0 occurs.
    """

    index = 0
    while index < len(method):
        if method[index][2] == '!' and len(method) > 1:
            del method[index]
        else:
            index += 1
    final_op = method.pop()
    if final_op[2] == '!':
        return str(final_op[3])
    if final_op[2] in '+*' and final_op[0] < final_op[1]:
        final_op[0], final_op[1] = final_op[1], final_op[0]
    new_method = [final_op + [0, final_op[2] in '+-']]
    max_depth = 0
    for operation in reversed(method):
        if operation[2] in '+*' and operation[0] < operation[1]:
            operation[0], operation[1] = operation[1], operation[0]
        for new_operation in new_method:
            if operation[3] in new_operation[:2]:
                depth = new_operation[4]
                addition = new_operation[5]
        if (operation[2] in '+-') ^ addition:
            depth += 1
        new_method.append(operation + [depth, operation[2] in '+-'])
        if depth > max_depth:
            max_depth = depth
    new_method.sort(key=lambda operation: operation[4])
    new_method.append([None, None, None, None, max_depth + 1, None])
    final_method = _group_operations(new_method, max_depth)
    return _method_to_string(final_method, numbers)


def _group_operations(method: list, max_depth: int) -> list:
    """Group and order consecutive operations which have the same depth.

    Parameters
    ----------
    method : list
        The method whose operators are to be grouped and ordered
    max_depth : int
        The maximum bracket depth of an operation in `method`

    Returns
    -------
    list
        The transformed method with grouped operations

    Rewrite the method by grouping consecutive operations and sorting them in
    size order. For example, 5 - 3 + 9 would be rewritten as 9 + 5 - 3,
    (10 - 7) * 8 would be rewritten as 8 * (10 - 7), 10 - (2 + 3) would be
    rewritten as 10 - 3 - 2, 4 / (20 / 100) would be rewritten as 100 * 4 / 20
    and (100 / 4) / 5 would be rewritten as 100 / (5 * 4)
    """

    final_method = []
    parts = []
    depth = 0
    for operation in method:
        if operation[4] > depth:
            for part in parts:
                part[1].sort()
                part[2].sort()
                final_method.append(part)
            parts = []
            depth = operation[4]
            if depth > max_depth:
                break
        if not parts:
            if operation[2] in '+-':
                part = ['+', [operation[0]], []]
                if operation[2] == '+':
                    part[1].append(operation[1])
                else:
                    part[2].append(operation[1])
            else:
                part = ['*', [operation[0]], []]
                if operation[2] == '*':
                    part[1].append(operation[1])
                else:
                    part[2].append(operation[1])
            parts.append(part)
        else:
            added = False
            for index, part in enumerate(parts):
                if operation[3] in part[1]:
                    parts[index][1].remove(operation[3])
                    parts[index][1].append(operation[0])
                    if operation[2] in '+*':
                        parts[index][1].append(operation[1])
                    else:
                        parts[index][2].append(operation[1])
                    added = True
                    break
                if operation[3] in part[2]:
                    parts[index][2].remove(operation[3])
                    parts[index][2].append(operation[0])
                    if operation[2] in '+*':
                        parts[index][2].append(operation[1])
                    else:
                        parts[index][1].append(operation[1])
                    added = True
                    break
            if not added:
                if operation[2] in '+-':
                    part = ['+', [operation[0]], []]
                    if operation[2] == '+':
                        part[1].append(operation[1])
                    else:
                        part[2].append(operation[1])
                else:
                    part = ['*', [operation[0]], []]
                    if operation[2] == '*':
                        part[1].append(operation[1])
                    else:
                        part[2].append(operation[1])
                parts.append(part)
    return final_method


def _method_to_string(method: list, numbers: list) -> str:
    strings = {}
    for number in numbers:
        strings.setdefault(number, []).append(str(number))
    while method:
        part = method.pop()
        value = part[1].pop()
        value_string = strings[value].pop()
        string = [value_string]
        while part[1]:
            other_value = part[1].pop()
            other_value_string = strings[other_value].pop()
            string.append(other_value_string)
            if part[0] == '+':
                value += other_value
            else:
                value *= other_value
        if part[0] == '+':
            string = [' + '.join(string)]
        else:
            numerator = ' * '.join(string)
            string = []
        while part[2]:
            other_value = part[2].pop()
            other_value_string = strings[other_value].pop()
            string.append(other_value_string)
            if part[0] == '+':
                value -= other_value
            else:
                if value == 0 and other_value == 0:
                    return ""
                value = Fraction(value, other_value)
        if part[0] == '+':
            string = ' - '.join(string)
            strings.setdefault(value, []).append(f'({string})')
        else:
            denominator = ' * '.join(string)
            if len(string) > 1:
                strings.setdefault(value, []).append(
                    f'{numerator} / ({denominator})')
            elif denominator:
                strings.setdefault(value, []).append(
                    f'{numerator} / {denominator}')
            else:
                strings.setdefault(value, []).append(numerator)
    if part[0] == '+':
        return strings[value][-1][1:-1]
    else:
        return strings[value][-1]


if __name__ == '__main__':
    input_numbers = input("Please enter the numbers to be used separated "
                          + "by a comma and press enter: ")
    numbers = [int(number.strip()) for number in input_numbers.split(',')]
    input_result = input("Please enter the target number and press enter: ")
    result = int(input_result)
    solutions = countdown_solve(numbers, result)
    if solutions == []:
        print("No solutions found")
