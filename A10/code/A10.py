import numpy as np
import matplotlib.pyplot as plt
import time
from utils import *


def two_loop_lbfgs(q, s, y, B0k):
    """
    Two-Loop L-BFGS Recursion

    Input:
        q: Vektor (alte Suchrichtung),
        s: list(nd.array(n,)) (Liste der letzten m Vektoren sk-1, ..., sk-m)
        y: list(nd.array(n,)) (Liste der letzten m Vektoren yk-1, ..., yk-m)
        B0k: Matrix (Schaetzung für die inverse Hesse-Matrix zum Zeitpunkt k-m)
    Output:
        p: Vektor (neue Suchrichtung)
    """
    p = None
    # TODO: Aufgabenteil 1. Two-Loop L-BFGS Recursion implementieren.
    # BEGIN SOLUTION
    roh = np.zeros(len(s))
    alpha = np.zeros(len(s))
    for i in range(len(s)):
        roh[i] = 1/np.dot(y[i], s[i])
        alpha[i] = roh[i] * np.dot(s[i], q)
        q -= alpha[i] * y[i]
    p = B0k @ q
    for i in range(len(s) - 1, -1, -1):
        beta = roh[i] * np.dot(y[i], p)
        p += (alpha[i] - beta) * s[i]
    # END SOLUTION
    return p


def lbfgs(f, x0, B0, m, rho=0.5, c1=1e-3, eps=1e-10, tau=1e-10, kmax=100, plot=True, display=True):
    """
    Limited-Memory BFGS-basiertes Quasi-Newton-Verfahren zur Minimierung der Funktion f mit
    Backtracking-Liniensuche, Abbruchkriterien nach Gill-Murray-Wright und Two-Loop L-BFGS Recursion.

    Input:
        f: Funktion, wobei f(x) = (val, grad, hess),
        x0: Vektor (Startwert),
        B0: Matrix (Initiale Schätzung für die inverse Hesse-Matrix),
        m: Integer (Memory-Größe im l-BFGS-Verfahren)
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
        'xlbfgs': None,              # Lösung des CG-Verfahrens
        'klbfgs': None,              # Anzahl benötigter Iterationen
        'x_list': [],             # Liste der Iterierten
        'val_list': [],           # Liste des Funktionswerts in den Iterierten
        'norm_grad_list': [],     # Liste der Norm des Gradienten in den Iterierten
    }

    # TODO: Aufgabenteil 2. L-BFGS-Verfahren implementieren.
    #   Hinweis. Die Listen-Methoden pop() und insert() können hilfreich seien.
    # BEGIN SOLUTION
    # Verwenden Quasi_Newton() aus utils.py als Basis

    # Initialisierung VERAENDERT
    xk = x0
    valk, gradk, _ = f(xk)
    k = 0
    s_list = []
    y_list = []

    # Iteration
    while True:
        if plot:
            log['x_list'].append(xk)
            log['val_list'].append(valk)
            log['norm_grad_list'].append(np.linalg.norm(gradk))

        # Bestimme Suchrichtung NEU
        if len(s_list) > 0:
            pk = -two_loop_lbfgs(gradk, s_list, y_list, B0)
        else:
            pk = -B0 @ gradk

        # Alt
        # Liniensuche
        alphak = armijo(f, p=pk, x=xk, rho=rho, c1=c1)

        if display:
            print(f'Iteration k = {k}: \talpha_k = {alphak: .4e}, '
                  f'\txk = {str(np.round(xk, 4).flatten()): <30}, \tf(xk) = {np.round(valk, 4)}')

        xkplus1 = xk + alphak * pk
        valkplus1, gradkplus1, _ = f(xkplus1)

        # Test der Abbruchkriterien
        if gill_murray_wright(xk, xkplus1, valk, valkplus1, gradk, k, tau, eps, kmax):
            break

        # Anpassung von Bk
        sk = xkplus1 - xk
        yk = gradkplus1 - gradk

        # Warnung, wenn yk @ sk nicht > 0
        if sk @ yk < 1e-12:
            print(f"ACHTUNG: sk @ yk = {sk @ yk}")

        # NEU
        # Füge s^k und y^k zu den Listen hinzu
        s_list.append(sk)
        y_list.append(yk)
        # Falls bereits m Vektoren in Listen: entferne das älteste Element
        if len(s_list) > m:
            s_list.pop(0)
            y_list.pop(0)

        # Update ALT
        xk, valk, gradk = xkplus1, valkplus1, gradkplus1
        k += 1

    log['xlbfgs'] = xk
    log['klbfgs'] = k
    # END SOLUTION
    return log


if __name__ == '__main__':
    m = 2
    x0 = np.array([.5, .5])
    tau = 1e-6
    eps = 1e-10
    kmax = 500
    rho = .5
    c1 = 1e-3

    f = lambda x: rosenbrock(*x)
    B0 = np.eye(2)

    # TODO: Aufgabenteil 3. l-BFGS-Verfahren testen
    # BEGIN SOLUTION
    log_lbfgs = lbfgs(f, x0, B0, m, rho=rho, c1=c1, tau=tau, eps=eps, kmax=kmax, plot=False, display=True)
    print(f"Lösung: {log_lbfgs['xlbfgs']}")
    print(f"Anzahl der Iterationen: {log_lbfgs['klbfgs']}")
    # END SOLUTION

    # TODO. Aufgabenteil 4. Vergleich von l-BFGS-, Quasi-Newton-, Newton- und Gradientenabstiegsverfahren
    #   Hinweis: Ein Implementierungsvorschlag der "älteren" Verfahren ist in utils.py gegeben
    # BEGIN SOLUTION
    # Erstellen x0-Werte
    x0_werte = [np.array([-1.2, 1.0]), np.array([0.0, -0.71])]

    # Bestimmen Verfahrennamen und deren Anwendung
    verfahren = [("l-BFGS (m=2)", lambda x0: lbfgs(f, x0, B0, m, rho=rho, c1=c1, tau=tau, eps=eps, kmax=kmax, plot=True, display=False)),
        ("BFGS", lambda x0: quasi_newton(f, x0, B0, rho=rho, c1=c1, tau=tau, eps=eps, kmax=kmax, plot=True, display=False)),
        ("Newton", lambda x0: newton_erweitert(f, x0, rho=rho, c1=c1, tau=tau, eps=eps, kmax=kmax)),
        ("Gradientenverfahren", lambda x0: gradient_descent_erweitert(f, x0, rho=rho, c1=c1, tau=tau, eps=eps, kmax=kmax))]

    # Iteration über Startwerte und Verfahren
    for x0 in x0_werte:
        for name, anwendung in verfahren:
            log = anwendung(x0)
            fig = plot_iteration_rosenbrock(log, f"{name}, Startwert: {x0}")
            plt.show()
    # END SOLUTION


    # TODO. Aufgabenteil 4. Diskussion
    # BEGIN SOLUTION
    print("Es zeigt sich, dass Newton die wenigsten Iterationen benötigt, dies geht natürlich auf Kosten von Speicherbedarf und Rechenzeit.\n"
          "Das Gradientenverfahren benötigt immer die kmax Iteration und terminiert durch GMW 3. Der berechnete Wert ist daher nicht optimal.\n"
          "Das BFGS-Verfahren und das l-BFGS-Verfahren benötigen beide (deutlich) weniger Iterationen als das Gradientenverfahren,\n"
          "wobei l-BFGS deutlich weniger Speicher benötigt. Dabei ist zu beobachten, dass abhängig vom Startwert jeweils eins der beiden\n"
          "Verfahren fast so schnell wie Newton (nach Iterationen) konvergiert, beim anderen Punkt aber deutlich schlechter.\n"
          "Wobei selbst hier das l-BFGS-Verfahren nicht so viele Iterationen benötigt wie das BFGS beim anderen Ausgangspunkt.\n")
    # END SOLUTION
