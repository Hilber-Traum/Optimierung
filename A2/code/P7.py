import numpy as np
import matplotlib.pyplot as plt

## TODO: Aufgabenteil 7a. 1D-Funktion plotten und Minimierer rot markieren
x = np.linspace(-10, 12, 500)
f = lambda x: (x - 1)**2 - 0.5
# Hinweis: x kann hier sowohl als einzelner Wert als auch als ein ganzes Array übergeben werden!

# Plotten und markieren
# BEGIN SOLUTION
...
# END SOLUTION


## TODO: Aufgabenteil 7b. Quadratische Funktion
print('### 7b ###')
def quadratic_function(x, A, b, c):
    """
    Quadratische Funktion
    Input:
        x: Variable, shape (n,)  | Zusatz: shape (n,) oder shape (m,n)
        A: Matrix, shape (n, n)
        b: Vektor, shape (n,)
        c: Skalar.

    Output:
        val: Skalar (g(x)).
    """
    # BEGIN SOLUTION
    ...
    # END SOLUTION


# Test
A = np.array([[2, 5], [1, 4]])
b = np.array([1, 1])
c = 5
x = np.array([1, 2])

result = quadratic_function(x, A, b, c)
print(f'Result quadratic function = {result}')


## TODO: Aufgabenteil 7c. Implementierung spezifischer quadratischer Funktionen
print('\n### 7c ###')
# BEGIN SOLUTION
...
# END SOLUTION


## TODO: Aufgabenteil 7d. Visualisierung quadratischer 2D-Funktionen
# 2D-Gitter-Punkte anlegen
# BEGIN SOLUTION
...
# END SOLUTION

# zugehörige Funktionswerte bestimmen
# BEGIN SOLUTION
...
# END SOLUTION

# Visualisirung als Höhenlinien
fig, ax = plt.subplots(1, 3)
fig.suptitle('Visualisierung als Höhelinien')
fig.set_size_inches(10, 4)
fig.tight_layout(pad=3)
# BEGIN SOLUTION
...
# END SOLUTION
plt.show()

# Visualisierung als Surface-Plot
fig = plt.figure()
fig.suptitle('Visualisierung als Surface-Plot')
fig.set_size_inches(12, 4)
fig.tight_layout(pad=1.5)
# BEGIN SOLUTION
...
# END SOLUTION
plt.show()