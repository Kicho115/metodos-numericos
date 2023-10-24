import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime


def parse_date(date_str):
    # Parse the date string into a numerical value
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return (date_obj - datetime.datetime(2022, 10, 13)).days  # Days since the starting date


def error(b, m, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent(points, starting_b, starting_m, learning_rate, iterations):
    b = starting_b
    m = starting_m
    for i in range(iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]



data = []

# Read data from "data.csv" and convert the dates
with open("data.csv", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row if it exists
    for row in csv_reader:
        date_str, value_str = row
        date_numeric = parse_date(date_str)
        value = float(value_str)
        if not np.isnan(value):
            data.append([date_numeric, value])

learning_rate = 0.00001
initial_b = 18
initial_m = 18
iterations = 1000

print(f'Starting gradient descent at b = {initial_b}, m = {initial_m}, and error {error(initial_b, initial_m, np.array(data))}')
[b, m] = gradient_descent(data, initial_b, initial_m, learning_rate, iterations)
print(f'Ending gradient descent at b = {b}, m = {m}, and error {error(b, m, np.array(data))} after {iterations} iterations')

# Run gradient descent to get the 'b' and 'm' values
[b, m] = gradient_descent(data, initial_b, initial_m, learning_rate, iterations)

# Calculate the regression line
regression_line = [m * x + b for x in np.array(data)[:, 0]]

# Plot the data points and regression line
plt.scatter(np.array(data)[:, 0], np.array(data)[:, 1], color='blue', label='Data Points')
plt.plot(np.array(data)[:, 0], regression_line, color='red', label='Regression Line')
plt.xlabel('Days Since Starting Date')
plt.ylabel('Value')
plt.legend()
plt.show()
