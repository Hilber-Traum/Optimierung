import numpy as np
import matplotlib.pyplot as plt

## TODO: Aufgabenteil 7a. 1D-Funktion plotten und Minimierer rot markieren
x = np.linspace(-10, 12, 500)
f = lambda x: (x - 1)**2 - 0.5
# Hinweis: x kann hier sowohl als einzelner Wert als auch als ein ganzes Array übergeben werden!

# Plotten und markieren
# BEGIN SOLUTION
# Plotte f(x) = (x - 1)² - 0.5
plt.plot(x, f(x), 'b-', label='f(x)')
# Plotten des Minimums
x_min = 1
y_min = (x_min - 1)**2 - 0.5
min_label = f'Minimum: f({x_min}) = {y_min}'
plt.scatter(x_min, y_min, color='red', label=min_label)
# Kosmetische Anpassungen
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = (x - 1)² - 0.5')
plt.legend()
plt.grid(True)
plt.show()
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
    return 0.5*np.dot(x.T, np.dot(A, x)) - np.dot(b.T, x) + c
    # END SOLUTION


# Test
A = np.array([[2, 5], [1, 4]])
b = np.array([1, 4]) # Im Blatt steht (1,4)
c = 5
x = np.array([1, 2])

result = quadratic_function(x, A, b, c)
print(f'Result quadratic function = {result}')


## TODO: Aufgabenteil 7c. Implementierung spezifischer quadratischer Funktionen
print('\n### 7c ###')
# BEGIN SOLUTION
# Erstelle Matrizen und Vektoren
A1 =  np.array([[2,0],[0,1]])
A2 =  np.array([[-2,0],[0,1]])
A3 =  np.array([[-2,1],[1,-1]])
b2 = np.array([1,3])
c2 = -1.5
x2 = np.array([1, 1])

# Erstelle die g1, g2, g3 als lambda-Funktion
g1 = lambda x: quadratic_function(x, A1, b2, c2)
g2 = lambda x: quadratic_function(x, A2, b2, c2)
g3 = lambda x: quadratic_function(x, A3, b2, c2)

# Teste die Implementierung
print(f'g1({x2}) = {g1(x2)} \n' 
      f'g2({x2}) = {g2(x2)} \n'
      f'g3({x2}) = {g3(x2)}')
# Ausgabe nicht ganz sauber, x müsste in der Darstellung ein Spaltenvektor sein :D
# END SOLUTION


## TODO: Aufgabenteil 7d. Visualisierung quadratischer 2D-Funktionen
# 2D-Gitter-Punkte anlegen
# BEGIN SOLUTION
x_werte = np.linspace(-10, 10, 1000)
y_werte = np.linspace(-10, 10, 1000)
xx, yy = np.meshgrid(x_werte, y_werte)
# END SOLUTION

# zugehörige Funktionswerte bestimmen
# BEGIN SOLUTION
# Geht bestimmt einfacher:
# werte_funktion_aus wertet für eine gegebene Funktion g Stellenweise die Funktion g aus
def werte_funktion_aus(funktion):
    bestimmte_werte = np.zeros_like(xx)
    # doppelte Indizierung weil 2d
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            # setze bestimmten Punkt in Funktion ein
            bestimmte_werte[i, j] = funktion(np.array([xx[i, j], yy[i, j]]))
    return bestimmte_werte

g1_werte = werte_funktion_aus(g1)
g2_werte = werte_funktion_aus(g2)
g3_werte = werte_funktion_aus(g3)
# END SOLUTION

# Visualisirung als Höhenlinien
fig, ax = plt.subplots(1, 3)
fig.suptitle('Visualisierung als Höhelinien')
fig.set_size_inches(10, 4)
fig.tight_layout(pad=3)
# BEGIN SOLUTION
# Plotten der Funktionen als Contour Plot
ax[0].contourf(xx,yy,g1_werte)
ax[1].contourf(xx,yy,g2_werte)
ax[2].contourf(xx,yy,g3_werte)

# Kosmetische Anpassungen
funktionen = ['g1', 'g2', 'g3']
for i in range(3):
    ax[i].set_title(funktionen[i])
    ax[i].set_xlabel("x")
    ax[i].set_ylabel("y")
# END SOLUTION
plt.show()

# Visualisierung als Surface-Plot
fig = plt.figure()
fig.suptitle('Visualisierung als Surface-Plot')
fig.set_size_inches(12, 4)
fig.tight_layout(pad=1.5)
# BEGIN SOLUTION
# Erstellen 3 Subplots mit jeweils 3 Achsen in 3d
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

# Plotten der Funktionen
ax1.plot_surface(xx, yy, g1_werte, cmap='viridis')
ax1.set_title("g1")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("g1(x)")

ax2.plot_surface(xx, yy, g2_werte, cmap='viridis')
ax2.set_title("g2")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("g2(x)")

ax3.plot_surface(xx, yy, g3_werte, cmap='viridis')
ax3.set_title("g3")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("g3(x)")
# END SOLUTION
plt.show()