## Hier werden saemtliche Pakete und Module geladen, die fuer das Notebook benoetigt sind.
import matplotlib.pyplot as plt

from utils import *
import numpy as np
import scipy.sparse as ssp
np.random.seed(0)



# TODO: Aufgabenteil 1: Implementierung des präkonditionierten CG-Verfahrens
def preconditioned_cg(A, b, c, P, x0, tol=1e-10, kmax=100, xstar=None, plot=False):
    """
    Praekonditioniertes CG-Verfahren

    Input:
        A: Matrix (spd!),
        b: Vektor,
        c: Skalar,
        P: Matrix (Praekonditionierungsmatrix),
        x0: Vektor (Startwert),
        tol: Skalar (Toleranz),
        kmax: Skalar (Maximale Anzahl an Iterationen),
        xstar: Vektor (Minimierer von f), optional.
        plot: Boolescher Wert (gibt an, ob Daten zur Ploterstellung gespeichert werden sollen).

    Output:
        log: Dictionary vom Verlauf der Optimierung.
    """
    # Funktion, die (val, grad, hess) der zugehörigen quadratischen Funktion zurückgibt.
    f = lambda x: quadratic_function(x, A, b, c)

    # Dictionary zum Abspeichern der Ergebnisse.
    log = {
        'xstar': xstar,  # Tatsächlicher Minimierer
        'val_star': None if xstar is None else f(xstar)[0],  # Tatsächliches Minimum
        'x0': x0,  # Startwert
        'xpcg': None,  # Lösung des PCG-Verfahrens
        'kpcg': None,  # Anzahl benötigter Iterationen
        'x_list': [],  # Liste der Iterierten
        'val_list': [],  # Liste des Funktionswerts in den Iterierten
        'norm_grad_list': [],  # Liste der Norm des Gradienten in den Iterierten
    }

    # BEGIN SOLUTION
    # Praekonditionierung
    A_quer = P.T @ A @ P
    b_quer = (P.T @ b).reshape(50,)
    x_quer = np.linalg.solve(P,x0).reshape(50,)

    # initialisierung
    xk = x_quer
    rk = b_quer - A_quer @ xk
    pk = rk.copy()

    # initiales loging
    log['x_list'].append(x0.copy())
    if plot:
        val, grad, _ = f(x0)
        log['val_list'].append(val)
        log['norm_grad_list'].append(np.linalg.norm(grad))

    # iterative Berechnung VERAENDERT AUS VORHERIGER ABGABE
    print("A= ",A_quer.shape,"\np= ",pk.shape,"\nb= ",b.shape,"\nx= ",x_quer.shape)
    for k in range(kmax):
        vk = A_quer @ pk.reshape(50,)
        alphak = rk @ rk / np.inner(pk,vk)
        xk += alphak * pk

        #loging
        x_log = P @ x_quer #Ruecktransformation
        log['x_list'].append(x_log.copy())
        if plot:
            val, grad, _ = f(x_log)
            log['val_list'].append(val)
            log['norm_grad_list'].append(np.linalg.norm(grad))

        betak = 1 / (rk @ rk) # erster Teil von beta
        rk -= alphak * vk
        #Abbruchbedingung
        if np.linalg.norm(rk) <= tol:
            log['xpcg'] = x_log.copy()
            log['kpcg'] = k
            break
        betak = rk @ rk * betak # zweiter Teil von beta
        pk = rk + betak * pk

    else:
        log['xpcg'] = xk.copy()
        log['kpcg'] = kmax
    # END SOLUTION
    return log


if __name__ == '__main__':
    # TODO: Aufgabenteil 2: Teste implementierung für verschiedene Praekonditionierungsmatrizen
    n = 50
    tol = 1e-5 # IN AUFGABENSTELLUNG STEHT 1e-5 NIVHT 1e-8 DAHER GEAENDERT
    kmax = 100
    b = 0.1 * np.random.randn(n)
    c = 0
    x0 = np.zeros_like(b)

    ones = np.ones(n)
    A = ssp.spdiags([-ones, 2*ones, -ones], [-1, 0, 1], n, n).toarray()
    L2 = np.tril(A)
    L3 = np.eye(n) - ssp.spdiags(np.ones(n-1), -1, n, n)

    P1 = np.eye(n)
    P2 = np.linalg.inv(L2.T)
    P3 = np.linalg.inv(L3.T)

    for P, whichP in zip([P1, P2, P3], ['P1', 'P2', 'P3']):
        print('---------------------------------------')
        print(f'Test mit {whichP}:')
        print('---------------------------------------')
        # BEGIN SOLUTION
        x_star = np.linalg.solve(A, b)
        log = preconditioned_cg(A, b, c, P, x0, tol=tol)
        PTAP =  P.T @ A @ P
        kond_PTAP = np.linalg.cond(PTAP)
        dif_EWe_PTAP = len(np.unique(np.linalg.eigvals(PTAP)))
        xpcg = log['xpcg']
        norm_abstand = np.linalg.norm(xpcg - x_star)
        iterationen = log['kpcg']
        print(f"Kondition von ${whichP}^T A {whichP}$: {kond_PTAP:.2e}")
        print(f"Anzahl unterschiedlicher Eigenwerte: {dif_EWe_PTAP}")
        print(f"Anzahl Iterationen: {iterationen}")
        print(f"||xpcg - x_star|| = {norm_abstand:.2e}")
        # END SOLUTION

        # TODO: Aufgabenteil 3: Ergänze die plot-Funktion in utils.py, um Plots für die Visualisierung der Optimierung zu erstellen
        fig = ...
        plt.show()

    # Diskussion
    # BEGIN SOLUTION
    ...
    # END SOLUTION
