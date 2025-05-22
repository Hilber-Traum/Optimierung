## Hier werden saemtliche Pakete und Module geladen, die fuer das Notebook benoetigt sind.
from utils import *
import numpy as np
# BEGIN SOLUTION
...
# END SOLUTION
np.random.seed(0)



# TODO: Aufgabenteil 1: Implementiere Standardform des CG-Verfahrens
def conjugate_gradient(A, b, c, x0, tol=1e-10, kmax=100, xstar=None, plot=False):
    """
    Standardform des CG-Verfahrens

    Input:
        A: Matrix (spd!),
        b: Vektor,
        c: Skalar,
        x0: Vektor (Startwert),
        tol: Skalar (Toleranz),
        kmax: Skalar (Maximale Anzahl an Iterationen),
        xstar: Vektor (Minimierer von f), optional.
        plot: Boolescher Wert (gibt an, ob Daten zur Ploterstellung gespeichert werden sollen).

    Output:
        log: Dictionary vom Verlauf der Optimierung.
    """
    # Quadratische Funktion, die (val, grad, hess) der zugehörigen quadratischen Funktion zurückgibt.
    f = lambda x: quadratic_function(x, A, b, c)

    # Dictionary zum Abspeichern der Ergebnisse.
    log = {
        'xstar': xstar,           # Tatsächlicher Minimierer
        'val_star': None if xstar is None else f(xstar)[0],  # Tatsächliches Minimum
        'x0': x0,                 # Startwert
        'xcg': None,              # Lösung des CG-Verfahrens
        'kcg': None,              # Anzahl benötigter Iterationen
        'x_list': [],             # Liste der Iterierten
        'val_list': [],           # Liste des Funktionswerts in den Iterierten
        'norm_grad_list': [],     # Liste der Norm des Gradienten in den Iterierten
    }

    # BEGIN SOLUTION

    #initialisierung
    xk = x0
    rk = b - A @ xk
    pk = rk

    #initiales loging
    log['x_list'].append(xk.copy())
    if plot:
        val, grad, _ = f(xk)
        log['val_list'].append(val)
        log['norm_grad_list'].append(np.linalg.norm(grad))

    # iterative Berechnung
    for k in range(1,kmax+1):
        test_konjugent = pk @ A # erster Teil vom Test
        vk = A @ pk
        alphak = rk @ rk / np.inner(pk,vk)
        xk = xk + alphak * pk

        #loging
        log['x_list'].append(xk.copy())
        if plot:
            val, grad, _ = f(xk)
            log['val_list'].append(val)
            log['norm_grad_list'].append(np.linalg.norm(grad))

        betak = 1 / (rk @ rk) # erster Teil von beta
        rk = rk - alphak * vk
        if np.linalg.norm(rk) <= tol:
            log['xcg'] = xk.copy()
            log['kcg'] = k
            break
        betak = rk @ rk * betak # zweiter Teil von beta
        pk = rk + betak * pk
        test_konjugent = test_konjugent @ pk # zweiter Teil vom Test
        if not (test_konjugent < 10**(-10)):
            print("Im Schritt ",k," sind die aufeinanderfolgenden Richtungen nicht konjugiert!")

    else:
        log['xcg'] = xk.copy()
        log['kcg'] = kmax
    # END SOLUTION
    return log


def test_aufgabenteil_2():
    # TODO: Aufgabenteil 2: Teste Implementierung auf 2-dim. Funktionen.
    # Hinweis: Die plot-Funktion aus utils.py soll nur die Plots erstellen und die Figure zurückgeben.
    # Stellen Sie die Abbildung bitte dar, indem Sie HIER plt.show() aufrufen.

    kmax = 200
    tol = 1e-10
    b = np.array([3, 6])
    c = np.pi
    x0 = np.array([0, 0])

    A1 = np.array([[13, 0], [0, 13]])
    A2 = np.array([[1, 1 / 2], [1 / 2, 1]])
    A3 = np.array([[100, 0], [0, 1]])

    # BEGIN SOLUTION
    f = lambda x: quadratic_function(x, A3, b, c)
    xstar = np.linalg.solve(A1, b)
    log = conjugate_gradient(A1, b, c, x0, tol=1e-10, kmax=100, xstar=xstar, plot=True)
    plot_iteration_process(log, "A1",f)
    print(log)
    plt.show()
    # END SOLUTION
    return


def test_aufgabenteil_3():
    # TODO: Aufgabenteil 3: Teste Implementierung auf Tridiagonalmatrix.
    # Hinweis: Importieren Sie das benötigte Paket am Anfang der Python-Datei wie folgt: import scipy.sparse as ssp

    print('---------------------------------------')
    print('Test mit A4')
    print('---------------------------------------')
    # BEGIN SOLUTION
    ...
    # END SOLUTION
    return


def test_aufgabenteil_4():
    # TODO: Aufgabenteil 4.i: Teste Implementierung auf Normalgleichung mit A5.
    print('\n---------------------------------------')
    print('Test mit A5')
    print('---------------------------------------')
    # BEGIN SOLUTION
    ...
    # END SOLUTION


    # TODO: Aufgabenteil 4.ii: Teste Implementierung auf Normalgleichung mit A6.
    print('\n---------------------------------------')
    print('Test mit A6')
    print('---------------------------------------')
    # BEGIN SOLUTION
    ...
    # END SOLUTION


    # TODO: Aufgabenteil 4.ii: Teste Implementierung auf Normalgleichung mit A6mod.
    print('\n---------------------------------------')
    print('test mit A6mod')
    print('---------------------------------------')
    # BEGIN SOLUTION
    ...
    # END SOLUTION
    return


if __name__ == '__main__':
    test_aufgabenteil_2()
    #test_aufgabenteil_3()
    #test_aufgabenteil_4()
