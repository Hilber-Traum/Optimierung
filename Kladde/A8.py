import matplotlib.pyplot as plt
import numpy as np
from utils import *

# weitere Module importieren
# BEGIN SOLUTION
from matplotlib import cm
import time
# END SOLUTION

def armijo(f, p, x, rho, c1, alpha0=1):
    """
    Armijo Backtracking Liniensuche

    Input:
        f: Funktion
        p: Vektor (Abstiegsrichtung)
        x: Vektor (Iterierte)
        rho: Skalar (Reduktionsfaktor für alpha)
        c1: Skalar (Parameter für die Armijo-Bedingung)
        alpha0: Skalar (Initial getesteter Startwert)

    Output:
        alpha: Skalar (gesuchte Schrittweite)
    """

    # TODO:  Aufgabenteil 1.i: Armijo Backtracking Liniensuche implementieren.
    # BEGIN SOLUTION
    alpha = alpha0
    fx , grad_fx , _ = f(x)
    for i in range(100):
        fx_alpha_p , _ , _ = f(x + alpha * p)

        if fx_alpha_p <= fx + c1 * alpha * np.inner(grad_fx.T, p):
            break
        else:
            alpha = rho * alpha
    else:
        print("Nach 100 Iterationen wurde keine Schrittweite gefunden, welche die Armijo-Bedingung erfüllt.\n")
    return alpha
    # END SOLUTION


def gill_murray_wright(xk, xkplus1, valk, valkplus1, gradk, k, eps, tau, kmax):
    """
    Abbruchkriterien nach Gill, Murray und Wright

    Input:
        xk: Vektor (aktuelle Iterierte)
        xkp1: Vektor (nächste Iterierte)
        valk: Skalar (f(xk))
        valkp1: Skalar (f(xkp1))
        gradk: Vektor (\nabla f(xk))
        k: Integer (Iteration-Nummer)
        tau: Skalar (Tolerenz für Kriterium 1)
        eps: Skalar (Tolerenz für Gradient -- Kriterium 2)
        kmax: Integer (maximale Anzahl der Iterationen)

    Output:
        test: Boolescher Wert (Ergebnis der Abbruchkriterien nach Gill, Murray, Wright)
    """

    # TODO:  Aufgabenteil 1.ii: Gill-Murray-Wright-Abbruchkriterien ueberpruefen.
    # BEGIN SOLUTION
    test = False #Initialisiere Boolean

    #Pruefe erste Bedingung
    eins_a_1 = valk - valkplus1
    eins_a_2 = tau * (1 + abs(valk))

    eins_b_1 = np.linalg.norm(xkplus1 - xk)
    eins_b_2 = tau**(1/2) * (1 + np.linalg.norm(xk))

    eins_c_1 = np.linalg.norm(gradk)
    eins_c_2 = tau**(1/3) * (1 + abs(valk))

    eins_a_ungl = eins_a_1 < eins_a_2
    eins_b_ungl = eins_b_1 < eins_b_2
    eins_c_ungl = eins_c_1 < eins_c_2

    if eins_a_ungl and eins_b_ungl and eins_c_ungl:
        test = True
        print("Die Verbesserung der Werte, der xk und der Norm des Gradienten sind sehr klein.\n")

    #Pruefe 2. Bedingung
    if np.linalg.norm(gradk) < eps:
        test = True
        print("Die Norm des Gradienten ist kleiner epsilon.\n")

    #Pruefe 3. Bedingung
    if k > kmax:
        test = True
        print("DIe maximale Anzahl an Iterationen ist erreicht.\n")

    return test
    # END SOLUTION


def gradient_descent_erweitert(f, x0, rho=0.5, c1=1e-3, eps=1e-10, tau=1e-10, kmax=100, xstar=None):
    """
    Gradientenabstiegsverfahren

    Input:
        f: Funktion, wobei f(x) = (val, grad, hess),
        x0: Vektor (Startwert),
        rho: Skalar (Armijo Parameter für die Anpassung der Schrittweite)
        c1: Skalar (Armijo Parameter für das Testen der Bedinungung),
        tau: Skalar (Tolerenz für Kriterium 1),
        eps: Skalar (Tolerenz für Gradient in GMW-Kriterium 2),
        kmax: Skalar (Maximale Anzahl an Interationen),
        xstar: Vector (Minimierer von f), optional


    Output:
        log: Dictionary vom Verlauf der Optimierung.
    """
    # Dictionary zum Abspeichern der Ergebnisse.
    log = {
        'xstar': xstar,           # Tatsächlicher Minimierer
        'val_star': None if xstar is None else f(xstar)[0],  # Tatsächliches Minimum
        'x0': x0,                 # Startwert
        'xgd': None,              # Lösung des Gradientenabstiegverfahrens
        'kgd': None,              # Anzahl benötigter Iterationen
        'x_list': [],             # Liste der Iterierten
        'val_list': [],           # Liste des Funktionswerts in den Iterierten
        'norm_grad_list': [],     # Liste der Norm des Gradienten in den Iterierten
    }

    # TODO:  Aufgabenteil 1.iii: Funktion ergaenzen.
    # BEGIN SOLUTION
    ...
    # END SOLUTION
    return log


def newton_erweitert(f, x0, rho=0.5, c1=1e-3, eps=1e-10, tau=1e-10, kmax=100, xstar=None):
    """
    Newton-Verfahren zur Minimierung der Funktion f.

    Input:
        f: Funktion, wobei f(x) = (val, grad, hess),
        x0: Vektor (Startwert),
        rho: Skalar (Armijo Parameter für die Anpassung der Schrittweite)
        c1: Skalar (Armijo Parameter für das Testen der Bedinungung),
        tau: Skalar (Tolerenz für Kriterium 1),
        eps: Skalar (Tolerenz für Gradient in GMW-Kriterium 2),
        kmax: Skalar (Maximale Anzahl an Interationen),
        xstar: Vector (Minimierer von f), optional


    Output:
        log: Dictionary vom Verlauf der Optimierung.
    """

    # Dictionary zum Abspeichern der Ergebnisse.
    log = {
        'xstar': xstar,           # Tatsächlicher Minimierer
        'val_star': None if xstar is None else f(xstar)[0],  # Tatsächliches Minimum
        'x0': x0,                 # Startwert
        'xnv': None,              # Lösung des Newton-Verfahrens
        'knv': None,              # Anzahl benötigter Iterationen
        'x_list': [],             # Liste der Iterierten
        'val_list': [],           # Liste des Funktionswerts in den Iterierten
        'norm_grad_list': [],     # Liste der Norm des Gradienten in den Iterierten
    }

    # TODO:  Aufgabenteil 1.iii: Funktion ergaenzen.
    # BEGIN SOLUTION
    ...
    # END SOLUTION
    return log


def rosenbrock(x, y):
    """
    Implementierung der Rosenbrock-Funktion f(x, y) = 100(y-x**2)**2 + (1-x)**2

    Input:
        x, y: Skalare bzw. Arrays zur Auswertung der Funktion.

    Output:
        val: Zielfunktionswert(e),
        grad: Gradient(en),
        hess: Hessematri(x/zen).
    """
    val, grad, hess = None, None, None

    # TODO. Aufgabenteil 2.ii. Rosenbrock implementieren.
    # BEGIN SOLUTION
    # Berechne Wert
    x = np.asarray(x)
    y = np.asarray(y)
    val = 100 * (y - x**2)**2 + (1 - x)**2

    # Berechne Gradienten
    # Differenzieren nach x: 2 * 100 * (y - x**2) * (-2) * x - 2 * (1 - x)
    gradx = -400 * x * (y - x**2) - 2 * (1 - x)
    # Differenzieren nach y: 2 * 100 * (y - x**2) * 1
    grady = 200 * (y - x**2)
    grad = np.array([gradx,grady])

    # Berechne Hesse-Matrix
    hesse_xx = -400 * (y - 3 * x**2) + 2
    hesse_yy = 200 * np.ones_like(x)
    hesse_xy = -400 * x

    hess = np.stack([[[hesse_xx, hesse_xy],[hesse_xy, hesse_yy]]], axis=0)
    # END SOLUTION

    return val, grad, hess


if __name__ == '__main__':
    # TODO. Aufgabenteil 2.iii. Rosenbrock visualisieren.
    # BEGIN SOLUTION
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)
    X, Y = np.meshgrid(x, y)
    Z, _, _ = rosenbrock(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.hsv, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=.5, aspect=10, pad=.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('$f(x, y)$')
    ax.set_title('Rosenbrock Funktion')

    plt.show()
    # END SOLUTION

    xstar = np.array([1, 1])
    rho = .5
    c1 = 1e-3
    eps = 1e-10
    kmax = 1000

    f = lambda x: rosenbrock(*x)

    for x0 in [np.array([0, -0.71]), np.array([-1.2, 1])]:
        for which_method in ['Gradientenabstieg', 'Newton']:
            for tau in [1e-2, 1e-10]:
                print('----')
                # TODO: Aufgabenteil 3: which_method Verfahren testen und Ergebnisse in Konsole ausgeben
                # BEGIN SOLUTION
                start_time = time.time()
                if which_method == 'Gradientenabstieg':
                    log = gradient_descent_erweitert(f,x0,rho,c1,eps,tau,kmax,xstar)
                    anzahl_iter = log['kgd']
                    loesung = log['xgd']
                else:
                    log = newton_erweitert(f,x0,rho,c1,eps,tau,kmax,xstar)
                    anzahl_iter = log['knv']
                    loesung = log['xnv']

                gesamt_time = time.time() - start_time
                zeit_iter = gesamt_time/anzahl_iter

                print(f"Verfahren: {which_method}")
                print(f"x0: {x0}")
                print(f"Tau: {tau}")
                print(f"Berechnete Lösung: {loesung}")
                print(f"Anzahl Iterationen: {anzahl_iter}")
                print(f"Gesamte Zeit: {gesamt_time}")
                print(f"Durchschnittliche Zeit pro Iteration: {zeit_iter}")
                print()
                # END SOLUTION

                # TODO:  Aufgabenteil 4: Iterationsverlauf plotten
                # BEGIN SOLUTION
                title = f"Plot für das {which_method}-Verfahren mit Startwert ${x0}$ und Tau = ${tau}$"
                figure = plot_iteration_rosenbrock(log, title)
                plt.show()
                # END SOLUTION

    # TODO:  Aufgabenteil 4: Diskussion
    # BEGIN SOLUTION
    ...
    # END SOLUTION
