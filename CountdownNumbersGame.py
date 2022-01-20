"""
Module Doctring:
A set of methods to solve the Countdown numbers game
"""

import math
import json
from fractions import Fraction


INVERSE = ({'+': '-', '-': '+', '*': '/', '/': '*', '!': '!'},
           {'+': '-', '-': '-', '*': '/', '/': '/', '!': '!'},)
ORDER = {'+': [1,0], '-': [0,1], '*': [1,0], '/': [0,1], '!': [1,0]}


def countdown_solve(numbers, result, brute_force=False, print_results=True):
    """ A method for finding all solutions to the countdown numbers game
    using a meet-in-the-middle approach.

    Two solutions are only considered distinct if one cannot be
    transformed into the other by reordering the operations performed.

    Note: we allow intermediate results to be rationals and not
    necessarily integers.

    This works when 6 or fewer numbers are given as in the gameshow but
    may not show all solutions with more numbers.
    """

    if brute_force:
        print("Warning: This is very slow!")
        return brute_force_search(numbers, result, print_results)
    n = len(numbers)
    if n > 6:
        print("This may not give all solutions.")
    out = meet_in_the_middle(numbers, result, n, n//2, [], print_results)
    out.extend(
        meet_in_the_middle(numbers, result, n, n//2-1, out, print_results))
    return out


def meet_in_the_middle(numbers, result, n, m, collisions, print_results):
    """Use the meet-in-the-middle approach to finding a collision
    between what numbers can be generated from a set of numbers from
    `numbers` of size m together with `result` and those that can be
    generated from the remaining elements of `numbers`
    """

    for c in range(math.comb(n, m)):
        first_indices = BinRel(c, n, m)
        first_set = [numbers[i] for i in first_indices] + [result]
        second_set = [numbers[i] for i in range(n) if i not in first_indices]
        firsts = compute_all(first_set, result)
        seconds = compute_all(second_set, result)
        for key in firsts.keys():
            if key in seconds.keys():
                for method in firsts[key]:
                    if method[1]:
                        first_method = rewrite_method(first_set, method[0],
                                                      result)
                        for other_method in seconds[key]:
                            second_method = rewrite_method(second_set,
                                                           other_method[0],
                                                           result)
                            new_method = merge_methods(first_set, first_method,
                                                       second_set,
                                                       second_method, result)
                            if new_method and new_method not in collisions:
                                collisions.append(new_method)
                                if print_results:
                                    print(new_method)
    return collisions

def BinRel(c, n, k):
    """Return the subset of {0, 1, ..., n-1} of size k indexed by c
    """

    if c < 0 or c >= math.comb(n, k):
        raise ValueError("c is out of range")
    if k == 0 or n == 0:
        return []
    elif c < math.comb(n-1, k):
        return BinRel(c, n-1, k)
    else:
        return BinRel(c-math.comb(n-1, k), n-1, k-1) + [n-1]

def compute_expression(numbers, expression, result, separate_sign=False):
    """Compute `expression` using `numbers` and `result` together with a
    status stating whether `result` has been used in the computation
    If `separate_sign` is set return the sign separately from the output
    """

    if result in numbers:
        status = True
    else:
        status = False
    for expr in expression:
        numbers, status, used = compute_step(numbers, expr[0], expr[1],
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

def compute_step(numbers, index1, index2, operation, result, status):
    """Compute one step of an expression using the given indices
    (relative to `numbers`) and the given `operation`
    The `!` operation simply drops the second number so that it does not
    get used
    """

    new_numbers = numbers[:]
    number1 = new_numbers.pop(index1)
    number2 = new_numbers.pop(index2)
    used = False
    if number1 == result or number2 == result:
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

def rewrite_method(numbers, expression, result):
    """Rewrite the `expression` in terms of the actual numbers being
    calculated with as well as their indices, and ensure that all
    numbers encountered are positive
    """

    method = []
    negatives = [False]*len(numbers)
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
        numbers, status, used = compute_step(numbers, expr[0], expr[1],
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

def div_and_rem(number, divisor):
    rem = number % divisor
    div = number // divisor
    return div, rem

def compute_part1(counter, num_exc, n):
    ops = ['+', '-', '*', '/']
    operations = ['!']*num_exc
    tot = math.comb(n, num_exc)
    parts = [tot]
    y = counter
    for j in range(n - 1 - num_exc):
        y, t = div_and_rem(y, 4)
        operations.append(ops[t])
        if t == 3:
            tot *= (n-num_exc-j) * (n-1-num_exc-j)
            parts.append((n-num_exc-j) * (n-1-num_exc-j))
        else:
            tot *= math.comb(n - num_exc - j, 2)
            parts.append(math.comb(n - num_exc - j, 2))
    return operations, tot, parts

def compute_part2(counter, parts, operations, num_exc, n, numbers, result, D):
    x = counter
    x, s = div_and_rem(x, parts[0])
    elements = BinRel(s, n, num_exc)
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
    for i in range(n - 1 -num_exc):
        x, r = div_and_rem(x, parts[i+1])
        if operations[num_exc+i] == '/':
            v, u = div_and_rem(r, n - num_exc - i)
        else:
            pair = BinRel(r, n - i, 2)
            u = pair[0]
            v = pair[1] - 1
        expression.append((u, v, operations[num_exc+i]))
    c, sgn, status = compute_expression(numbers, expression, result, True)
    if c != 'DivisionError':
        if c in D.keys():
            if [expression,status] not in D[c]:
                D[c].append([expression, status])
        else:
            D[c] = [[expression, status]]
    return D

def compute_all(numbers, result):
    """Compute all possible numbers that can be created from numbers
    using +, -, *, / and !
    The operator ! denotes simply dropping the second argument,
    allowing for not all numbers to be used
    """

    D = {}
    n = len(numbers)
    if n == 1:
        c, sgn = compute_expression(numbers, (0, 0, '!'), True)
        status = False
        if numbers[0] == result:
            status = True
        D[c] = [[(0, 0, '!'), status]]
        return D

        if n == 3:
            with open('3_numbers_indices.json', 'r') as f:
                all_str_indices = f.read()
        else:
            with open('4_numbers_indices.json', 'r') as f:
                all_str_indices = f.read()

        all_indices = json.loads(all_str_indices)

        for indices in all_indices:
            num_exc, counter1, counter2 = indices
            operations, tot, parts = compute_part1(counter1, num_exc, n)
            D = compute_part2(counter2, parts, operations,
                              num_exc, n, numbers, result, D)
        return D

    for num_exc in range(n):
        for counter1 in range(4 ** (n-1-num_exc)):
            operations, tot, parts = compute_part1(counter1, num_exc, n)
            for counter2 in range(tot):
                D = compute_part2(counter2, parts, operations,
                                  num_exc, n, numbers, result, D)
    return D

def brute_force_search(numbers, result, print_results):
    D = compute_all(numbers, result)
    solutions = []
    if result in D.keys():
        for method in D[result]:
            new_method = rewrite_method(numbers, method[0], result)
            new_method = [[step[0][0], step[1][0], step[2],step[3][0]]
                          for step in new_method]
            solution = canonical_form(new_method, numbers)
            if solution and solution not in solutions:
                solutions.append(solution)
                if print_results:
                    print(solution)
    return solutions

def merge_methods(numbers1, method1_original, numbers2, method2, result):
    """Merge two methods which have the same output to a single method
    for computing `result`
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
        if (step[0] + [used_count] in using_result
            or [step[1][0], step[1][1]+delta] + [used_count] in using_result):
            to_remove = []
            for index, triple in enumerate(using_result):
                if triple in [step[0] + [used_count], 
                              [step[1][0], step[1][1]+delta] + [used_count]]:
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
    while method1:
        step = method1.pop()
        delta = 0
        if step[0][1] <= step[1][1]:
            delta = 1
        if step[0]+[used_count] in used_result:
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
    final_method = [[step[0][0], step[1][0], step[2], step[3][0]]
                    for step in method]
    return canonical_form(final_method, numbers1[:] + numbers2[:])

def canonical_form(method, numbers):
    """Write the method in a canonical form (larger values before
    smaller, + before -, * before /)
    Output the method as a readable string
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
    final_method = []
    parts = []
    depth = 0
    for operation in new_method:
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
    strings = {}
    for number in numbers:
        strings.setdefault(number, []).append(str(number))
    while final_method:
        part = final_method.pop()
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
            string = ['+'.join(string)]
        else:
            numerator = '*'.join(string)
            string = []
        while part[2]:
            other_value = part[2].pop()
            other_value_string = strings[other_value].pop()
            string.append(other_value_string)
            if part[0] == '+':
                value -= other_value
            else:
                if other_value == 0:
                    return ""
                value = Fraction(value, other_value)
        if part[0] == '+':
            string = '-'.join(string)
            strings.setdefault(value, []).append(f'({string})')
        else:
            denominator = '*'.join(string)
            if len(string) > 1:
                strings.setdefault(value, []).append(
                    f'{numerator}/({denominator})')
            elif denominator:
                strings.setdefault(value, []).append(
                    f'{numerator}/{denominator}')
            else:
                strings.setdefault(value, []).append(numerator)
    if part[0] == '+':
        return strings[value][-1][1:-1]
    else:
        return strings[value][-1]
