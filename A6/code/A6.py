## Hier werden saemtliche Pakete und Module geladen, die fuer das Notebook benoetigt sind.
from utils import *
import numpy as np
# BEGIN SOLUTION
import scipy.sparse
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
        if not (test_konjugent < 1e-10):
            print("Im Schritt ",k," sind die aufeinanderfolgenden Richtungen nicht konjugiert!\n")

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
    matrizen = [A1,A2,A3]
    for i in range(3):
        f = lambda x: quadratic_function(x, matrizen[i], b, c)
        xstar = np.linalg.solve(matrizen[i], b)
        log_cg = conjugate_gradient(matrizen[i], b, c, x0, tol=1e-10, kmax=100, xstar=xstar, plot=True)
        matrixname = ''.join(["A",str(i+1)])
        figure_cg = plot_iteration_process(log_cg, "CG: " + matrixname, f)
        plt.show()

        log_ga = gradient_descent(f, x0, tol, kmax, xstar)
        figure_gd = plot_iteration_process(log_ga, "Gradientenabstieg: " + matrixname, f)
        plt.show()
    print("Das CG-Verfahren benötigt jeweils nur 2 Schritte.\n"
          "Das Gradientenabstiegsverfahren wiedrrum benötigt bei der A2 und A3 Matrix deutlich mehr Iterationen.\n"
          "Bei der letzten wird sogar das kmax-Abbruchskriterium erreicht.\n")
    # END SOLUTION
    return


def test_aufgabenteil_3():
    # TODO: Aufgabenteil 3: Teste Implementierung auf Tridiagonalmatrix.
    # Hinweis: Importieren Sie das benötigte Paket am Anfang der Python-Datei wie folgt: import scipy.sparse as ssp

    print('---------------------------------------')
    print('Test mit A4')
    print('---------------------------------------')
    # BEGIN SOLUTION
    # Matrix A und Vektor b:
    A4 = scipy.sparse.spdiags(np.ones((1, 250)) * np.array([[-1], [2], [-1]]),
                              np.array([-1, 0, 1]), 250, 250).toarray()
    b4 = 0.1 * np.random.randn(250)

    #sonstige Initialisierungen:
    tol = 1e-10
    kmax = 3000
    xstar = np.linalg.solve(A4, b4)
    x0 = np.zeros(250)
    c = 0
    f = lambda x: quadratic_function(x, A4, b4, c)

    # Berechnungen und Plots
    # Gradientenabstiegsverfahren:
    log_ga = gradient_descent(f, x0, tol, kmax, xstar)
    figure_ga = plot_iteration_process(log_ga, "Gradientenabstieg: A4", f=None)
    print("\nGradientenabstiegsverfahren:")
    print("Schritte: ", log_ga['kgd'], "\n")
    print("Residuumsnorm: ", np.linalg.norm(A4 @ log_ga['xgd'] - b4),"\n")
    print("Normdifferenz zu xstar: ", np.linalg.norm(log_ga['xgd'] - xstar),"\n")
    plt.show()

    # CG-Verfahren:
    log_cg = conjugate_gradient(A4, b4, c, x0, tol, kmax, xstar, plot=True)
    figure_cg = plot_iteration_process(log_cg, "CG: A4", f=None)
    print("\nCG-Verfahren:")
    print("Schritte: ", log_cg['kcg'],"\n")
    print("Residuumsnorm: ", np.linalg.norm(A4 @ log_cg["xcg"] - b4),"\n")
    print("Normdifferenz zu xstar: ", np.linalg.norm(log_cg["xcg"] - xstar),"\n")
    plt.show()
    # END SOLUTION
    return


def test_aufgabenteil_4():
    # TODO: Aufgabenteil 4.i: Teste Implementierung auf Normalgleichung mit A5.
    print('\n---------------------------------------')
    print('Test mit A5')
    print('---------------------------------------')
    # BEGIN SOLUTION
    # Initialisierungen
    A5 = np.array([[1, 2],[3, 4],[5, 6]])
    b5 = np.array([np.pi, 2, 1])
    tol = 1e-2
    c = 0
    x0 = np.zeros(2)

    # Matrizenrechnungen
    ATA = A5.T @ A5
    ATb = A5.T @ b5

    f = lambda x: quadratic_function(x, ATA, ATb, c)
    log_cg = conjugate_gradient(ATA, ATb, c, x0)
    min_norm_lsg = log_cg['xcg']
    xstar = np.linalg.solve(ATA, ATb)
    print("Die Differenz zwischen der berechneten Minimum-Norm-Lösung"
          "und dem Ergebnis von np.linalg.solve() für A5 und b5 ist: ",
          np.linalg.norm(min_norm_lsg-xstar),"\n")
    # END SOLUTION


    # TODO: Aufgabenteil 4.ii: Teste Implementierung auf Normalgleichung mit A6.
    print('\n---------------------------------------')
    print('Test mit A6')
    print('---------------------------------------')
    # BEGIN SOLUTION
    A6 = np.diag(np.arange(1, 101))
    b6 = np.random.rand(100)
    x0 = np.zeros(100)

    f1 = lambda x: quadratic_function(x, A6, b6, c)
    xstar1 = np.linalg.solve(A6, b6)
    log1 = conjugate_gradient(A6, b6, c, x0, tol=tol)
    print("Benötigte Iterationen: ", log1["kcg"])

    # END SOLUTION


    # TODO: Aufgabenteil 4.ii: Teste Implementierung auf Normalgleichung mit A6mod.
    print('\n---------------------------------------')
    print('test mit A6mod')
    print('---------------------------------------')
    # BEGIN SOLUTION
    A6mod = A6.T @ A6
    b6mod = A6.T @ b6

    f2 = lambda x: quadratic_function(x, A6mod, b6mod, c)
    xstar2 = np.linalg.solve(A6mod, b6mod)
    log2 = conjugate_gradient(A6mod, b6mod, c, x0, tol=tol)
    print("Benötigte Iterationen: ", log2["kcg"])

    print("Die modifizierte Matrix/Vektor benötigen deutlich mehr Iterationen.\n"
          "Außerdem sind bei dieser für viele Suchrichtungen jeweils 2 aufeinanderfolgende nicht konjugiert zueinander.")
    # END SOLUTION
    return


if __name__ == '__main__':
    test_aufgabenteil_2()
    test_aufgabenteil_3()
    test_aufgabenteil_4()
