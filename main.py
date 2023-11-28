""""
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
"""

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import csv
import datetime


def parse_date(date_str):
    # Parse the date string into a numerical value
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return (date_obj - datetime.datetime(2022, 10, 13)).days  # Days since the starting date


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

data = np.array(data)
n = data.shape[0]

xi = data[:, 0]
yi = data[:,1]

xi2 = xi ** 2
xi3 = xi ** 3
xi4 = xi ** 4
xiyi = xi * yi
xi2yi = xi2 * yi

tabla = np.transpose([xi, yi, xi2, xi3, xi4, xiyi, xi2yi])
sum = np.sum(tabla, axis=0)
tabla = np.vstack([tabla, sum])
print(tabulate(tabla, headers=['xi', 'yi', 'xi^2', 'xi^3', 'xi^4', 'xi * yi', 'xi^2 * yi'],tablefmt='fancy_grid'))

A = np.array( [
    [n, sum[0], sum[2]],
    [sum[0], sum[2], sum[3]],
    [sum[2], sum[3], sum[4]]
])

b = np.array([[sum[1]], [sum[5]], [sum[6]]])

print(A)
print(b)
print(np.append(A,b,1))

def gauss_jordan(y):
    x = y.copy()
    filas = x.shape[0]

    for i in range(filas):
        # Hacer que el elemento diagonal actual sea igual a 1
        x[i] /= x[i, i]

        for j in range(filas):
            if i != j:
                factor = x[j, i]
                x[j] -= factor * x[i]
    return x

matriz = np.append(A, b, 1)
A = gauss_jordan(matriz)
print(A)

a = A[:, 3]
print(a)


y_prom = np.mean(yi)
st = (yi - y_prom) ** 2

sr = (yi - a[0] - a[1] * xi - a[2] * xi ** 2) ** 2

tabla = np.transpose([xi, yi, xi2, xi3, xi4, xiyi, xi2yi, st, sr])
sum = np.sum(tabla, axis=0)
tabla = np.vstack([tabla, sum])
print(tabulate(tabla, headers=['xi', 'yi', 'xi^2', 'xi^3', 'xi^4', 'xi * yi', 'xi^2 * yi', '(yi - yi)^2', '(yi - a0 - a1 * xi - a2 * xi^2)^2'],tablefmt='fancy_grid'))

coeficiente_determinacion = (sum[7] - sum[8]) / sum[7]
coeficiente_correlacion = np.sqrt(coeficiente_determinacion)

print(f'Coeficiente de determinacion = {coeficiente_determinacion} \nCoeficiente de relacion = {coeficiente_correlacion}')

y = a[0] + a[1] * xi + a[2] * xi ** 2

plt.title('Regresion polinomial')
plt.plot(xi, y, color='black', label='Ajuste polinómico', zorder=1)
plt.scatter(xi, yi, color='red', label='Datos originales', zorder=2)
plt.xlabel('xi')
plt.ylabel('yi')
plt.legend()
plt.show()