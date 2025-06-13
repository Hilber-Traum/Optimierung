import matplotlib.pyplot as plt
import numpy as np
import warnings
from utils import *

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
    xk = x0
    valk, gradfxk, _ = f(xk)
    pk = np.dot(Bk, - gradfxk)
    alphak = armijo(f, pk, xk, rho, c1)
    xkplus1 = xk + alphak * pk
    valkplus1, gradfxkplus1, _ = f(xkplus1)
    k = 0

    #speichere alle Einträge im log
    if plot:
        log['x_list'].append(xk)
        log['val_list'].append(valk)
        log['norm_grad_list'].append(np.linalg.norm(gradfxk))
        log['x_list'].append(xkplus1)
        log['val_list'].append(valkplus1)
        log['norm_grad_list'].append(np.linalg.norm(gradfxkplus1))

    if (display):
        print(f'Ausgabe der Werte in der {k}-ten Iteration:')
        print(f'alpha^k ist: {alphak}')
        print(f'x^k ist: {xk}')
        print(f'f(x^k) ist: {valkplus1}')

    #a) Überprüfe Abbruchkriterien
    while (not gill_murray_wright(xk, xkplus1, valk, valkplus1, gradfxkplus1, k, eps, tau, kmax)):

        #e) Berechne s^k, y^k und überprüfe, ob deren Produkt >0 ist
        sk = xkplus1 - xk
        yk = gradfxkplus1 - gradfxk
        if(np.dot(yk.T, sk) <= 0):
            warnung = f"s^k * y^k <= 0 in Iteration: {k}"
            warnings.warn(warnung)
            break


        #f) Berechne B_(k+1)
        rohk = 1/(np.dot(yk, sk))
        Bk = (np.eye(len(yk)) - rohk * np.dot(sk,yk.T)) @ Bk @ (np.eye(len(yk)) - rohk * np.dot(yk,sk.T)) + rohk * np.dot(sk, sk.T)

        k+=1

        # b) Berechne Suchrichtung
        pk = np.matmul(Bk, (-gradfxkplus1))

        # c) Bestimme Schrittweite, die die Armijo-Bed. erfüllt
        alphak = armijo(f, pk, xkplus1, rho, c1)

        # d) Bestimme neues x^k
        xk = xkplus1
        xkplus1 = xkplus1 + alphak * pk

        # Update Werte und Gradienten und berechne Wert und Gradienten an x^(k+1)
        valk = valkplus1
        gradfxk = gradfxkplus1
        valkplus1, gradfxkplus1, _ = f(xkplus1)

        # gebe alle Einträge aus, sofern display=True ist
        if (display):
            print(f'Ausgabe der Werte in der {k}-ten Iteration:')
            print(f'alpha^k ist: {alphak}')
            print(f'x^k ist: {xk}')
            print(f'f(x^k) ist: {valk}')

        # speichere alle Einträge im log
        if plot:
            log['x_list'].append(xkplus1)
            log['val_list'].append(valkplus1)
            log['norm_grad_list'].append(np.linalg.norm(gradfxkplus1))

    log['kqn'] = k
    log['xqn'] = xkplus1
    # END SOLUTION
    return log


if __name__ == '__main__':
    # TODO: Aufgabenteil 3. Test-Werte anlegen, Verfahren für tau = 1e-3 und tau = 1e-10 testen und Verlauf plotten
    # BEGIN SOLUTION
    f = lambda x: rosenbrock(*x.T)

    #Für tau=1e-10
    print('----------------Testen Verlauf für tau = 1e-10 und x0 = (-1.2, 1)----------------------')
    result_tau_big = quasi_newton(f, np.array([-1.2,1]), np.identity(2), rho = 0.5, c1 = 1e-3, eps = 1e-10, kmax = 100, display = False)
    minimizer = result_tau_big['xqn']
    print(f'Der gefundene Minimierer ist: {minimizer}')


    #Für tau=1e-2
    print('----------------Testen Verlauf für tau = 1e-2 und x0 = (-1.2, 1)----------------------')
    result_tau_small = quasi_newton(f, np.array([-1.2,1]), np.eye(2), rho = 0.5, c1 = 1e-3, eps = 1e-2, tau = 1e-2, kmax = 100, display = False)
    minimizer = result_tau_small['xqn']
    print(f'Der gefundene Minimierer ist: {minimizer}\n')

    #Plotte die Ergebnisse
    fig = plot_iteration_rosenbrock(result_tau_big, "Quasi-Newton mit tau = 1e-10")
    plt.show()

    fig = plot_iteration_rosenbrock(result_tau_small, "Quasi-Newton mit tau = 1e-2")
    plt.show()

    # END SOLUTION

    # TODO: Aufgabenteil 3. Verfahren für x0 = (0, -0.75) testen und Verlauf plotten
    # BEGIN SOLUTION

    #Für tau=1e-10
    print('----------------Testen Verlauf für tau = 1e-10 und x0 = (0, -0.75)----------------------')
    result_tau_big1 = quasi_newton(f, np.array([0, -0.75]), np.eye(2), rho = 0.5, c1 = 1e-3, eps = 1e-10, kmax = 100, display = False)
    minimizer = result_tau_big1['xqn']
    print(f'Der gefundene Minimierer ist: {minimizer}')


    #Für tau=1e-3
    print('----------------Testen Verlauf für tau = 1e-2 und x0 = (0, -0.75)----------------------')
    result_tau_small1 = quasi_newton(f, np.array([0, -0.75]), np.eye(2), rho = 0.5, c1 = 1e-3, eps = 1e-2, tau = 1e-2, kmax = 100, display = False)
    minimizer = result_tau_small1['xqn']
    print(f'Der gefundene Minimierer ist: {minimizer}\n')

    #Plotte die Ergebnisse
    fig = plot_iteration_rosenbrock(result_tau_big1, "Quasi-Newton mit tau = 1e-10")
    plt.show()

    fig = plot_iteration_rosenbrock(result_tau_small1, "Quasi-Newton mit tau = 1e-2")
    plt.show()

    # END SOLUTION

    # TODO: Aufgabenteil 3. Diskussion
    # BEGIN SOLUTION
    print("Bei beiden Startwerten hat die Größe von Tau keinen Einfluss auf die Anzahl der Iterationen,\n"
          "da im ersten Fall der Minimierer jeweils nach 8 Iterationen gefunden wird und im zweiten Fall beide Male\n"
          " die 100 Iteration durchlaufen werden. \n")
    # END SOLUTION
