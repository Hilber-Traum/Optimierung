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
    # Bei der P3 wird hier b_quer mit der Dimension (1,50) berechnet. (WARUM!?)
    # Infolge dessen finden sich im Folgenden wilde Verwendungen von '.T' ,
    # welche im Durchlauf mit P1 und P2 aufgrund der Dimension (50,) keine Auswirkung haben,
    # aber bei der P3 zur (hoffentlich richtigen) Loesung fuehren.
    # Habe gerade an ravel() gedacht :c
    # Funktioniert so trotzdem....
    b_quer = (P.T @ b).ravel()
    x_quer = np.linalg.solve(P,x0)

    # initialisierung
    if np.any(x0 != 0):
        xk = x_quer
        rk = b_quer - A_quer @ xk
        pk = rk
    else:
        xk = np.zeros(n)
        rk = b_quer
        pk = rk

    # initiales loging
    log['x_list'].append(x0.copy())
    if plot:
        val, grad, _ = f(x0)
        log['val_list'].append(val)
        log['norm_grad_list'].append(np.linalg.norm(grad))

    # iterative Berechnung VERAENDERT AUS VORHERIGER ABGABE
    for k in range(kmax):
        vk = (A_quer @ pk.T)
        alphak = rk @ rk.T / np.dot(pk,vk)
        xk = xk + alphak * pk

        #loging
        x_log = np.ravel((P @ xk.T).T) #Ruecktransformation
        log['x_list'].append(x_log.copy())
        if plot:
            val, grad, _ = f(x_log)
            log['val_list'].append(val)
            log['norm_grad_list'].append(np.linalg.norm(grad))

        betak = 1 / (rk @ rk.T) # erster Teil von beta
        rk = rk - alphak * vk.T
        #Abbruchbedingung
        if np.linalg.norm(rk) <= tol:
            log['xpcg'] = x_log.copy()
            log['kpcg'] = k
            break
        betak = np.dot(rk,rk.T) * betak # zweiter Teil von beta
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
    L3 = np.eye(n) - ssp.spdiags(ones, -1, n, n)

    P1 = np.eye(n)
    P2 = np.linalg.inv(L2.T)
    P3 = np.linalg.inv(L3.T)

    for P, whichP in zip([P1, P2, P3], ['P1', 'P2', 'P3']):
        print('---------------------------------------')
        print(f'Test mit {whichP}:')
        print('---------------------------------------')
        # BEGIN SOLUTION
        # Vorbereitende Berechnungen/Bestimmungen
        x_star = np.linalg.solve(A, b)
        log = preconditioned_cg(A, b, c, P, x0, tol=tol, xstar=x_star, plot=True)
        PTAP =  P.T @ A @ P
        #Kondition
        kond_PTAP = np.linalg.cond(PTAP)
        # Anzahl verschiedener EWe
        # Runden da minimale Abweichungen mutmaßlich numerische Fehler sind
        dif_EWe_PTAP = len(np.unique(np.round(np.linalg.eigvals(PTAP),decimals=16)))
        # bestimmtes x und weiteres
        xpcg = log['xpcg']
        norm_abstand = np.linalg.norm(xpcg - x_star)
        iterationen = log['kpcg']
        # Ausgaben
        print(f"Kondition von ${whichP}^T A {whichP}$: {kond_PTAP:.2e}")
        print(f"Anzahl unterschiedlicher Eigenwerte: {dif_EWe_PTAP}")
        print(f"Anzahl Iterationen: {iterationen}")
        print(f"||xpcg - x_star|| = {norm_abstand:.2e}")
        # END SOLUTION

        # TODO: Aufgabenteil 3: Ergänze die plot-Funktion in utils.py, um Plots für die Visualisierung der Optimierung zu erstellen
        fig = plot_iteration_process_nd(log, f"Verlauf für {whichP}")
        plt.show()

    # Diskussion
    # BEGIN SOLUTION
    print("\nDas 'gewöhnliche' CG-Verfahren, also hier die P1 Matrix, benoetigt die meisten Iterationen und die Matrix\n"
          "hat hier die schlechteste Kondition und auch n verschiedene EWe.\n"
          "Mit der P2 werden nurnoch 21 Iterationen, also weniger als die Hälfte im Vergleich zur P1, benötigt.\n"
          "Die Kondition ist auch besser, jedoch hat die Matrix weiterhin n verschiedene EWe.\n"
          "Zu beachten ist, dass die Normdifferenz zwar größer als bei der P1 ist, dies jedoch am gewählten\n"
          "'tol=1e-5' liegt.\n"
          "Bei der 'maßgeschneiderten Lösung' P3 werden nur noch 2 Iterationen benötigt.\n"
          "Die Anzahl der EWe ist wie in der VL beschrieben 2 und die Kondition auch die Beste der 3 Matrizen.\n"
          "Anzumerken ist, dass die absolute Verbesserung der Differenz der Funktionswerte bei der P1 und P2\n"
          "bereits nach 30 bzw. 10 Iterationen deutlich nachlässt.\n"
          "Zu erkennen ist, dass die maßgeschneiderte Lösung erwartbar am sinnvollsten für die Lösung des Problems ist\n"
          "jedoch bereits der Problemunspezifische Ansatz der P2 eine deutliche Verbesserung zum nativen CG-Verfahren ist.")
    # END SOLUTION
