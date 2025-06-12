import matplotlib.pyplot as plt
import numpy as np
import warnings
from utils import *

#Armijo Backtracking Liniensuche von Blatt 8
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
        x_alpha_p = (x + alpha * p).ravel()
        fx_alpha_p , _ , _ = f(x_alpha_p)

        if fx_alpha_p <= fx + c1 * alpha * np.inner(grad_fx.T, p):
            break
        else:
            alpha = rho * alpha
    else:
        print("Nach 100 Iterationen wurde keine Schrittweite gefunden, welche die Armijo-Bedingung erfüllt.\n")
    return alpha
    # END SOLUTION


def quasi_newton(f, x0, B0, rho=0.5, c1=1e-3, eps=1e-10, tau=1e-10, kmax=100, plot=True, display=True):
    """
    BFGS-basiertes Quasi-Newton-Verfahren zur Minimierung der Funktion f.

    Input:
        f: Funktion, wobei f(x) = (val, grad, hess),
        x0: Vektor (Startwert),
        B0: Matrix (Initiale Schätzung für die inverse Hesse-Matrix),
        rho, c1: Skalare (Armijo Parameter),
        tau, eps, kmax: Skalare (Abbruchkriterien nach Gill-Murray-Wright),
        plot: Boolescher Wert (gibt an, ob Daten zur Ploterstellung gespeichert werden sollen),
        display: Boolescher Wert (gibt an, ob Zwischenwerte aus den Iterationen in der Konsole ausgegeben werden sollen)

    Output:
        log: Dictionary vom Verlauf der Optimierung.
    """

    # Dictionary zum Abspeichern der Ergebnisse.
    log = {
        'x0': x0,                 # Startwert
        'xqn': None,              # Lösung des CG-Verfahrens
        'kqn': None,              # Anzahl benötigter Iterationen
        'x_list': [],             # Liste der Iterierten
        'val_list': [],           # Liste des Funktionswerts in den Iterierten
        'norm_grad_list': [],     # Liste der Norm des Gradienten in den Iterierten
    }

    # TODO: Aufgabenteil 1. Quasi-Newton-Verfahren implementieren.
    # BEGIN SOLUTION
    Bk = B0
    sk = 0
    xk = np.array([0,0])
    valk = 0
    xkplus1 = x0
    valkplus1, gradfxkplus1, hess = f(xkplus1)
    k = 0

        #speichere alle Einträge im log
    log['x_list'].append(xkplus1)
    log['val_list'].append(valkplus1)
    log['norm_grad_list'].append(np.linalg.norm(gradfxkplus1))

    #a) Überprüfe Abbruchkriterien
    while(not gill_murray_wright(xk, xkplus1,valk, valkplus1, gradfxkplus1, k, eps, tau, kmax)):

        #b) Berechne Suchrichtung
        pk = np.matmul(Bk, (-gradfxkplus1))

        #c) Bestimme Schrittweite, die die Armijo-Bed. erfüllt
        alphak = armijo(f, pk, xkplus1, rho, c1)


        #d) Bestimme neues x^k
        xk = xkplus1
        xkplus1 = xkplus1 + alphak * pk
        print(f'xk ist: {xk} und xk+1 ist: {xkplus1}')
        print(f'alphak ist: {alphak} und pk ist: {pk}')

        #Berechne Gradienten an x^k, x^(k+1)
        valk, gradfxk, hess = f(xk)
        valkplus1, gradfxkplus1, hess = f(xkplus1)

        #printe alle geforderten Informationen
        if(display):
            print(f'Ausgabe der Werte in der {k}-ten Iteration:')
            print(f'alpha^k ist: {alphak}')
            print(f'x^k ist: {xk}')
            print(f'f(x^k) ist: {valkplus1}')

        #e) Berechne s^k, y^k und überprüfe, ob deren Produkt >0 ist
        sk = xkplus1 - xk
        yk = gradfxkplus1 - gradfxk

        if(np.dot(yk, sk) == 0):
            warnings.warn("s^k * y^k = 0 - das Programm wird abgebrochen")
            return log


        #f) Berechne B_(k+1)
        rohk = 1/(np.dot(yk, sk))
        Bk = np.identity(len(yk)) - rohk*np.outer(yk,sk) * Bk * np.identity(len(yk)) - rohk*np.outer(yk, sk) + rohk * np.dot(sk, sk)

        k+=1

        #gebe alle Einträge aus, sofern display=1 ist

        #speichere alle Einträge im log
        log['x_list'].append(xkplus1)
        log['val_list'].append(valkplus1)
        log['norm_grad_list'].append(np.linalg.norm(gradfxkplus1))
        log['kqn'] = k

    # END SOLUTION
    return log

#Rosenbrock-Funktion von Blatt 8
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
    hesse_xx = 1200 * x**2 - 400 * y + 2
    hesse_yy = 200 * np.ones_like(x)
    hesse_xy = -400 * x

    hess = np.stack([[[hesse_xx, hesse_xy],[hesse_xy, hesse_yy]]], axis=0)
    # END SOLUTION

    return val, grad, hess



if __name__ == '__main__':
    # TODO: Aufgabenteil 3. Test-Werte anlegen, Verfahren für tau = 1e-3 und tau = 1e-10 testen und Verlauf plotten
    # BEGIN SOLUTION
    f = lambda x: rosenbrock(*x.T)

    #Für tau=1e-10
    print('----------------Testen Verlauf für tau = 1e-10 und x0 = (-1, 1)----------------------')
    result_tau_big = quasi_newton(f, np.array([-1,1]), np.identity(2), rho = 0.5, c1 = 1e-3, eps = 1e-10, kmax = 100)
    iterationcount = result_tau_big['kqn']
    minimizer = result_tau_big['x_list'][-1]
    print(iterationcount)
    print(f'Der gefundene Minimierer ist: {minimizer}')


    #Für tau=1e-3
    print('----------------Testen Verlauf für tau = 1e-2 und x0 = (-1, 1)----------------------')
    result_tau_small = quasi_newton(f, np.array([-1,1]), np.identity(2), rho = 0.5, c1 = 1e-3, eps = 1e-2, kmax = 100)
    iterationcount = result_tau_small['kqn']
    minimizer = result_tau_small['x_list'][-1]
    print(iterationcount)
    print(f'Der gefundene Minimierer ist: {minimizer}\n')

    #Plotte die Ergebnisse
    fig = plot_iteration_rosenbrock(result_tau_big, "Quasi-Newton mit tau = 1e-10")
    plt.show()

    fig = plot_iteration_rosenbrock(result_tau_big, "Quasi-Newton mit tau = 1e-2")
    plt.show()

    # END SOLUTION

    # TODO: Aufgabenteil 3. Verfahren für x0 = (0, -0.75) testen und Verlauf plotten
    # BEGIN SOLUTION

    #Für tau=1e-10
    print('----------------Testen Verlauf für tau = 1e-10 und x0 = (0, -0.75)----------------------')
    result_tau_big = quasi_newton(f, np.array([0, -0.75]), np.identity(2), rho = 0.5, c1 = 1e-3, eps = 1e-10, kmax = 100)
    iterationcount = result_tau_big['kqn']
    minimizer = result_tau_big['x_list'][-1]
    print(iterationcount)
    print(f'Der gefundene Minimierer ist: {minimizer}')


    #Für tau=1e-3
    print('----------------Testen Verlauf für tau = 1e-2 und x0 = (0, -0.75)----------------------')
    result_tau_small = quasi_newton(f, np.array([0, -0.75]), np.identity(2), rho = 0.5, c1 = 1e-3, eps = 1e-2, kmax = 100)
    iterationcount = result_tau_small['kqn']
    minimizer = result_tau_small['x_list'][-1]
    print(iterationcount)
    print(f'Der gefundene Minimierer ist: {minimizer}\n')

    #Plotte die Ergebnisse
    fig = plot_iteration_rosenbrock(result_tau_big, "Quasi-Newton mit tau = 1e-10")
    plt.show()

    fig = plot_iteration_rosenbrock(result_tau_big, "Quasi-Newton mit tau = 1e-2")
    plt.show()

    # END SOLUTION

    # TODO: Aufgabenteil 3. Diskussion
    # BEGIN SOLUTION
    # END SOLUTION
