"""
Module Docstring:
Iteractive solver for the Countdown numbers game
"""

from CountdownNumbersGame import countdown_solve

if __name__ == '__main__':
	input_numbers = input("Please enter the numbers to be used separated by a comma and press enter: ")
	numbers = [int(number.strip()) for number in input_numbers.split(',')]
	input_result = input("Please enter the target number and press enter: ")
	result = int(input_result)
	solutions = countdown_solve(numbers, result)
	if solutions == []:
		print("No solutions found")
	_ = input("Press enter to exit.")
