import numpy as np
from utils import *

# TODO: Aufgabenteil 2. Implementieren Sie die quadratische Funktion.
def quadratic_function(x, A, b, c):
    """
    Quadratische Funktion

    Input:
        x: Variable,
        A: Matrix,
        b: Vektor,
        c: Skalar.

    Output:
        val: Skalar (f(x)),
        grad: Vektor ( \nabla f(x)),
        hess: Matrix ( \nablaˆ2 f(x)).
    """

    # BEGIN SOLUTION
    val = 1/2 * x.T @ A @ x - b.T @ x + c
    grad = 1/2 * (A + A.T) @ x - b
    hess = 1/2 * (A + A.T)
    return val, grad, hess
    # END SOLUTION


# TODO: Aufgabenteil 3. Implementieren Sie das Gradientenabstiegsverfahren mit optimaler Schrittweitenbestimmung.
def gradient_descent(f, x0, tol=1e-10, kmax=100, xstar=None):
    """
    Gradientenabstiegsverfahren

    Input:
        f: Funktion, wobei f(x) = (val, grad, hess),
        x0: Vektor (Startwert),
        tol: Skalar (Toleranz),
        kmax: Skalar (Maximale Anzahl an Iterationen),
        xstar: Vektor (Minimierer von f), optional.

    Output:
        log: Dictionary vom Verlauf der Optimierung.
    """
    # Dictionary zum Abspeichern der Ergebnisse.
    log = {
        'xstar': xstar,           # Tatsächlicher Minimierer
        'val_star': None if xstar is None else f(xstar)[0],  # Tatsächliches Minimum
        'x0': x0,                 # Startwert
        'xgd': None,              # Lösung des Gradientenabstiegverfahrens
        'x_list': [],             # Liste der Iterierten
        'val_list': [],           # Liste der Funktionswerte in den Iterierten
        'norm_grad_list': [],     # Liste der Norm der Gradienten in den Iterierten
    }

    # BEGIN SOLUTION
    xk=x0
    for k in range(kmax):
        # Werte berechnen
        val, grad, hess = f(xk)
        norm_gradf = np.linalg.norm(grad)
        # logging
        log['x_list'].append(xk)
        log['val_list'].append(val)
        log['norm_grad_list'].append(norm_gradf)

        # Abbruchbedingung
        if norm_gradf <= tol:
            break

        # Suchrichtung: p = -nabla_f(xk)
        pk = -grad

        # Optimale Schrittweite alpha_k
        alpha_k = (grad.T @ grad) / (grad.T @ hess @ grad)

        # Update xk
        xk = xk + alpha_k * pk
    # END SOLUTION
    return log



if __name__ == '__main__':
    kmax = 200
    tol = 1e-10
    b = np.array([3, 6])
    c = np.pi
    x0 = np.array([0, 0])

    A1 = np.array([[13, 0], [0, 13]])
    A2 = np.array([[1, 1/2], [1/2, 1]])
    A3 = np.array([[100, 0], [0, 1]])

    # TODO: Aufgabenteil 4 und 5. Teste Implementierung auf gegebenen Werten, visualisiere Optimierungsverlauf.
    # Hinweis: Die plot-Funktion aus utils.py soll für publish.py nur die Plots erstellen und die Figure zurückgeben.
    # Stellen Sie die Abbildung bitte dar, indem Sie HIER plt.show() aufrufen.

    # BEGIN SOLUTION

    f = lambda x : quadratic_function(x, A1, b, c)
    log = gradient_descent(f, x0, tol=tol, kmax=kmax, xstar=None)
    number_iterations = len(log['x_list'])
    x_solution = log['x_list'][-1]
    xstar = np.linalg.solve(A1, b) # +c ändert das Minimum nicht
    print(f"\nErgebnis für {A1}\n"
          "Berechnete Lösung x = ",x_solution,
          "\nAnzahl Iterationen: ",number_iterations,
          "\nTatsächliche Lösung x* = ",xstar
          )

    f = lambda x : quadratic_function(x, A2, b, c)
    log = gradient_descent(f, x0, tol=tol, kmax=kmax, xstar=None)
    number_iterations = len(log['x_list'])
    x_solution = log['x_list'][-1]
    xstar = np.linalg.solve(A2, b) # +c ändert das Minimum nicht
    print(f"\nErgebnis für {A2}\n"
          "Berechnete Lösung x = ",x_solution,
          "\nAnzahl Iterationen: ",number_iterations,
          "\nTatsächliche Lösung x* = ",xstar
          )

    f = lambda x : quadratic_function(x, A3, b, c)
    log = gradient_descent(f, x0, tol=tol, kmax=kmax, xstar=None)
    number_iterations = len(log['x_list'])
    x_solution = log['x_list'][-1]
    xstar = np.linalg.solve(A3, b) # +c ändert das Minimum nicht
    print(f"\nErgebnis für {A3}\n"
          "Berechnete Lösung x = ",x_solution,
          "\nAnzahl Iterationen: ",number_iterations,
          "\nTatsächliche Lösung x* = ",xstar
          )
    # END SOLUTION
