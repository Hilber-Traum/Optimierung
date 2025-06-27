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
        #L(x,lmb) = f(x) + lmb.T g(x)

        # Bestimmen die ABleitungen von f
        _, grad_fxk, hess_fxk = f(xk)

        #Beginnen mit der Berechnung von Hk
        Hk = hess_fxk
        #Bestimmen die Ableitungen von g
        jacobi_g = np.zeros((m, n))
        val_gxk = np.zeros(m)
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
    # END SOLUTION
    else:
        log['xsqp'] = xk
        log['lmbsqp'] = lmbk
        log['ksqp'] = 100
    return log


def main():
    # TODO: Aufgabenteil 2i. Funktionen für Problem 1 konstruieren
    #  Hinweis: Sowohl f als auch g1 kann als quadratische Funktionen dargestellt werden (Implementierung in utils.py)
    # BEGIN SOLUTION
    A1 = np.array([[-1, 0], [0, -1]])
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
    xstar = np.array([np.sqrt(2), 0.0]) # Was ist hier richtig?!
    for x in x0:
        # kopiert von unten
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
                   title='Optimierungsverlauf bei Problem 1')
        fig.show()
    # END SOLUTION

    # Diskussion
    # BEGIN SOLUTION
    print("Der eine Punkt liegt im 'Tal' und der andere 'außerhalb' der Nebenbedingung. Beim Start im 'Tal'\n"
          "wird zunächst ein Schritt ganz weit raus gemacht, da g1(x) dort nicht erfüllt ist, sondern sogar beinahe\n"
          "minimiert wird. Danach wird sich sukzessive dem Rand genähert, bis die Nebenbedingung erfüllt ist.\n"
          "Beim anderen Punkt wird zunächst ebenfalls ein Schritt weg von der Nebenbedingung gemacht, um sich dann\n"
          "auf an der Achse entlang zu bewegen, bis die Nebenbedingung erfüllt ist.\n")
    # END SOLUTION

    # TODO: Aufgabenteil 2ii. Implementierung auf Problem 2 testen und Ergebnis ausgeben
    #  Hinweis: Sowohl f als auch g1 kann als quadratische Funktionen dargestellt werden (Implementierung in utils.py)
    # BEGIN SOLUTION
    A3 = np.array([[100, 0, 0], [0 , 25, 0], [0, 0, 100/9]])
    A4 = np.zeros((3,3))
    b3 = np.array([-1300, -735, -440])
    b4 = np.array([1, 1, -1])
    c3 = 13983.25
    c4 = 0

    f_problem2 = lambda x: quadratic_function(x, A3, b3, c3)
    g1_problem2 = lambda x: quadratic_function(x, A4, b4, c4)
    x0 = [np.array([0.0, 0.0, 0.0]),np.array([6.5, 14.7, 19.8])]
    for x in x0:
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
    ...
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
