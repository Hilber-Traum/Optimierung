import matplotlib.pyplot as plt
import numpy as np
from utils import *


def newton_sqp(f, g, x0, lmb0, tol=1e-7, kmax=100, plot=True):
    """
    Newton-SQP-Verfahren mit Vollschritt zur Minimierung der Funktion f unter Gleichungsnebenbedingungen g(x)=0.

    Input:
        f: Funktion, wobei f(x) = (val, grad, hess),
        g: Liste von Funktionen g_1(x),...,g_m(x), wobei g_i(x) = (val, grad, hess),
        x0: Vektor (n,) (Startwert für die Iterierten),
        lmb0: Vektor (m,) (Startwert für die Lagrange-Multiplikatoren),
        tol: Skalar (Toleranz, Abbruch wenn die L2-Norm von (deltax, deltalmb) <= tol ist),
        kmax: Integer (Anzahl maximaler Iterationen),
        plot: Boolescher Wert (gibt an, ob Daten zur Ploterstellung gespeichert werden sollen).

    Output:
        log: Dictionary vom Verlauf der Optimierung.
    """

    # Dictionary zum Abspeichern der Ergebnisse.
    log = {
        'x0': x0,           # Startwert
        'xsqp': None,       # Loesung des Newton-SQP-Verfahrens (Iterierte)
        'lmbsqp': None,     # Zur Loesung zugehoerige Lagrange-Multiplikatoren
        'ksqp': None,       # Anzahl benoetigter Iterationen
        'x_list': [],       # Liste der Iterierten
        'val_f_list': [],   # Liste des Funktionswerts in den Iterierten
        'g_res_list': [],   # Liste von Residuen der Nebenbedingungen in den Iterierten, d.h. |g_1(x)| + ... + |g_m(x)|
    }

    # TODO: Aufgabenteil 1. Newton-SQP-Verfahren implementieren
    # BEGIN SOLUTION
    xk = x0
    lmbk = lmb0

    # Überprüfen, ob g eine Liste ist, wenn nicht, dann in eine Liste umwandeln
    if not isinstance(g, list):
        g = [g] # Gleichbedeutend mit m = 1
        lmbk = [lmbk]

    # Bestimme die Dimensionen
    n = len(xk)
    m = len(g)

    for k in range(kmax):
        # Bestimmen die ABleitungen von f
        _, grad_fxk, hess_fxk = f(xk)

        #Beginnen mit der Berechnung von Hk
        Hk = hess_fxk
        #Bestimmen die Ableitungen von g
        # Konstruieren zunächst die Vektoren und Matrizen zum Speichern aufgrund der mehreren Nebenbedingungen
        jacobi_g = np.zeros((m, n))
        val_gxk = np.zeros(m)
        # Bestimmen fuer jede Nebenbedingung die Ableitungen
        for i in range(m):
            val_gixk, grad_gixk, hess_gixk = g[i](xk)
            val_gxk[i] = val_gixk
            jacobi_g[i,:] = grad_gixk
            # Berechnen Hk
            Hk = Hk + lmbk[i] * hess_gixk

        #Erstellen die Matrix Ak
        Ak = jacobi_g

        # Erstellen die Blockmatrix und den loesungsvektor
        block_matrix = np.block([[Hk, Ak.T],[Ak, np.zeros((m, m))]])

        loesungsvektor = np.concatenate([-grad_fxk-np.dot(Ak.T,lmbk), - val_gxk])

        # Loesen das Gleichungssystem
        delta_xlmb = np.linalg.solve(block_matrix, loesungsvektor)
        [delta_xk, delta_lmbk] = np.split(delta_xlmb, [n])

        # Logging der Daten für den Plot
        if plot:
            log['x_list'].append(xk.copy())
            val_f, _, _ = f(xk)
            log['val_f_list'].append(val_f)
            log['g_res_list'].append(np.sum(np.abs([g[i](xk)[0] for i in range(m)])))

        # Pruefen Abbruchkriterium
        if np.linalg.norm(delta_xk)**2 + np.linalg.norm(delta_lmbk)**2 <= tol:
            log['xsqp'] = xk
            log['lmbsqp'] = lmbk
            log['ksqp'] = k+1
            return log

        # Update
        xk = xk + delta_xk
        lmbk = lmbk + delta_lmbk
    else:
        log['xsqp'] = xk
        log['lmbsqp'] = lmbk
        log['ksqp'] = kmax
    # END SOLUTION
    return log


def main():
    # TODO: Aufgabenteil 2i. Funktionen für Problem 1 konstruieren
    #  Hinweis: Sowohl f als auch g1 kann als quadratische Funktionen dargestellt werden (Implementierung in utils.py)
    # BEGIN SOLUTION
    #Bestimmen Matrixen und Vektoren für die quadratischen Funktionen
    A1 = 2 * np.array([[-1, 0], [0, -1]]) # *2 damit das 1/2 ausgeglichen wird
    b1 = np.array([8, 0])
    c1 = 9
    A2 = - A1
    b2 = np.array([0, 0])
    c2 = -1

    f_problem1 = lambda x: quadratic_function(x, A1, b1, c1)
    g1_problem1 = lambda x: quadratic_function(x, A2, b2, c2)
    # END SOLUTION



    # TODO: Aufgabenteil 2i. Implementierung auf Problem 1 testen, Ergebnis ausgeben und Plot erstellen
    #  Hinweis: Sowohl f als auch g1 kann als quadratische Funktionen dargestellt werden (Implementierung in utils.py)
    # BEGIN SOLUTION
    lmb0 = 0
    tol = 1e-10
    kmax = 100
    x0 = [np.array([1.5, 5.0]), np.array([-0.1, 0.0])]
    xstar = np.array([-1.0, 0.0]) # Loesung vom letzten Blatt?
    for x in x0:
        # kopiert/verändert von unten
        log_sqp = newton_sqp(f=f_problem1, g=g1_problem1, x0=x, lmb0=lmb0, tol=tol, kmax=kmax)
        print(f'----------------------------------------- \n'
              f' SQP auf Problem 1 (Startwert: x0 = {x})\n'
              f'-----------------------------------------')
        print(f"""        Anzahl benoetigter Iterationen: ksqp = {log_sqp['ksqp']}
        Gefundene Loesung: xsqp = {np.round(log_sqp['xsqp'], 4)}
        Optimale Lagrange Multiplikatoren: lmb = {log_sqp['lmbsqp'].flatten()}
        """)
        fig = plot(f=f_problem1, g=g1_problem1, log=log_sqp,
                   xstar = xstar,
                   title='Optimierungsverlauf bei Problem 1, x0 = ' + str(x))
        fig.show()
    # END SOLUTION

    # Diskussion
    # BEGIN SOLUTION
    print("Abhängig von der Wahl des x0 werden hier die beiden kritischen Punkte gefunden, dabei führt der erste\n"
          "Startwert jedoch zum Maximierer und lediglich der zweite zum Minimierer unter der Nebenbedingung.\n")
    # END SOLUTION

    # TODO: Aufgabenteil 2ii. Implementierung auf Problem 2 testen und Ergebnis ausgeben
    #  Hinweis: Sowohl f als auch g1 kann als quadratische Funktionen dargestellt werden (Implementierung in utils.py)
    # BEGIN SOLUTION
    A3 = 2 * np.array([[100, 0, 0], [0 , 25, 0], [0, 0, 100/9]]) # *2 damit das 1/2 ausgeglichen wird
    A4 = np.zeros((3,3))
    b3 = np.array([-1300, -735, -440])
    b4 = np.array([1, 1, -1])
    c3 = 13983.25
    c4 = 0

    f_problem2 = lambda x: quadratic_function(x, A3, b3, c3)
    g1_problem2 = lambda x: quadratic_function(x, A4, b4, c4)
    x0 = [np.array([0.0, 0.0, 0.0]),np.array([6.5, 14.7, 19.8])]
    for x in x0:
        # kopiert/verändert von unten
        log_sqp = newton_sqp(f=f_problem2, g=g1_problem2, x0=x, lmb0=lmb0, tol=tol, kmax=kmax, plot=False)
        print(f'----------------------------------------- \n'
              f' SQP auf Problem 2 (Startwert: x0 = {x})\n'
              f'-----------------------------------------')
        print(f"""        Anzahl benoetigter Iterationen: ksqp = {log_sqp['ksqp']}
        Gefundene Loesung: xsqp = {np.round(log_sqp['xsqp'], 4)}
        Optimale Lagrange Multiplikatoren: lmb = {log_sqp['lmbsqp'].flatten()}
        """)
    # END SOLUTION

    # Diskussion
    # BEGIN SOLUTION
    print("In diesem Fall führen beide Startwerte zu der gleichen Lösung, welche auch die berechnete Lösung aus\n"
          "Aufgabe 1 ist. Interessant dabei ist, dass trotz der deutlich unterschiedlichen Distanz zwischen den\n"
          "Startwerten und dem Minimierer wird in beiden Fällen der gleiche Lagrange-Multiplikator und\n"
          "auch die gleiche Anzahl an Iterationen benötigt.\n")
    # END SOLUTION


    # TODO: Aufgabenteil 3. Implementierung von newton_sqp auf Problem 1 mit mehreren Nebenbedingungen erweitern
    #  Hier ist nichts zu tun - Code testet Implementierung automatisch
    #  Es könnte allerdings notwendig sein, nachträglich in obigen Funktionsaufrufen g1 durch [g1] zu ersetzen
    try:
        lmb0 = np.array([0, 0])
        x0 = np.array([1.5, 5.0])
        xstar = np.array([3/4, np.sqrt(7)/4])
        g2_problem1 = lambda x: g1_problem1(x-np.array([1.5, 0]))
        log_sqp = newton_sqp(f=f_problem1, g=[g1_problem1, g2_problem1], x0=x0, lmb0=lmb0)
        print(f'----------------------------------------- \n'
              f' SQP auf Problem 1 mit 2 Nebenbedingungen \n'
              f'-----------------------------------------')
        print(f"""        Anzahl benötigter Iterationen: ksqp = {log_sqp['ksqp']}
        Gefundene Lösung: xsqp = {np.round(log_sqp['xsqp'], 4)}
        Residuum: ||xsqp - x*||^2 = {np.linalg.norm(log_sqp['xsqp'] - xstar)}
        Optimale Lagrange Multiplikatoren: lmb = {log_sqp['lmbsqp'].flatten()}
        """)
        try:
            fig = plot(f=f_problem1, g=[g1_problem1, g2_problem1], log=log_sqp, xstar=xstar,
                       title='Optimierungsverlauf bei 2 Nebenbedingungen')
            fig.show()
        except:
            print('Implementierung funktioniert für mehr als eine Nebenbedingung, allerdings nicht im Plot')
    except:
        print('Die Implementierung funktioniert leider nicht für mehr als eine Nebenbedingung :(')


if __name__ == '__main__':
    main()
