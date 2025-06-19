import matplotlib.pyplot as plt
import numpy as np


# Uebernommen von Blatt 8
def armijo(f, p, x, rho, c1, alpha0=1):
    """
    Armijo Backtracking Liniensuche

    Input:
        f: Funktion, wobei f(x) = (val, grad, hess),
        p: Vektor (Abstiegsrichtung)
        x: Vektor (Iterierte)
        rho: Skalar (Reduktionsfaktor für alpha)
        c1: Skalar (Parameter für die Armijo-Bedingung)
        alpha0: Skalar (Initial getesteter Startwert)

    Output:
        alpha: Skalar (gesuchte Schrittweite)
    """

    # Initialisierung
    alpha = alpha0

    # Vorberechnung reduziert Laufzeit in der Schleife
    valk, gradk, _ = f(x)
    descent = gradk @ p

    for lsiter in range(100):
        valkp1, _, _ = f(x + alpha * p)

        # Testen der ``sufficient decrease condition``
        if valkp1 <= valk + c1 * alpha * descent:
            return alpha

        # Reduzieren von alpha
        alpha = rho * alpha

    # Fehlermeldung, falls die Liniensuche nicht auskonvergiert ist
    print("Liniensuche wurde nach 100 Schritten frühzeitig abgebrochen.")
    return alpha


# Uebernommen von Blatt 8
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
        eps: Skalar (Tolerenz für Gradient -- Kriterium 3)
        kmax: Integer (maximale Anzahl der Iterationen)

    Output:
        test: Boolescher Wert (Ergebnis der Abbruchkriterien nach Gill, Murray, Wright)
    """

    test = False
    if (valk - valkplus1 < tau * (1 + np.abs(valk))
            and np.linalg.norm(xkplus1 - xk) < np.sqrt(tau) * (1 + np.linalg.norm(xk))
            and np.linalg.norm(gradk) < np.cbrt(tau) * (1 + np.abs(valk))):
        test = True
        print('\nAbbruch nach GMW 1 erfuellt')

    if np.linalg.norm(gradk) < eps:
        test = True
        print('\nAbbruch nach GMW 3 erfuellt')

    if k == kmax:
        test = True
        print('\nAbbruch nach GMW 3 erfuellt')

    return test



# Uebernommen von Blatt 8
def gradient_descent_erweitert(f, x0, rho=0.5, c1=1e-3, eps=1e-10, tau=1e-10, kmax=100, xstar=None):
    """
    Gradientenabstiegsverfahren

    Input:
        f: Funktion, wobei f(x) = (val, grad, hess),
        x0: Vektor (Startwert),
        rho: Skalar (Armijo Parameter für die Anpassung der Schrittweite)
        c1: Skalar (Armijo Parameter für das Testen der Bedinungung),
        tau: Skalar (Tolerenz für Kriterium 1),
        eps: Skalar (Tolerenz für Gradient in GMW-Kriterium 3),
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

    k = 0
    xk = x0
    while True:
        valk, gradk, hessk = f(xk)

        log['x_list'].append(xk)
        log['val_list'].append(valk)
        log['norm_grad_list'].append(np.linalg.norm(gradk))

        pk = -gradk
        alpha0 = max(- pk @ gradk / (pk @ hessk @ pk), 1)
        alphak = armijo(f, pk, xk, rho=rho, c1=c1, alpha0=alpha0)  # ---- NEU: Armijo Liniensuche ----
        xkplus1 = xk + alphak * pk

        if gill_murray_wright(xk, xkplus1, valk, valkplus1=f(xkplus1)[0], gradk=gradk, k=k, tau=tau, eps=eps,
                              kmax=kmax):
            break

        k += 1
        xk = xkplus1

    log['xgd'] = xk
    log['kgd'] = k
    return log


# Uebernommen von Blatt 8
def newton_erweitert(f, x0, rho=0.5, c1=1e-3, eps=1e-10, tau=1e-10, kmax=100, xstar=None):
    """
    Newton-Verfahren zur Minimierung der Funktion f.

    Input:
        f: Funktion, wobei f(x) = (val, grad, hess),
        x0: Vektor (Startwert),
        rho: Skalar (Armijo Parameter für die Anpassung der Schrittweite)
        c1: Skalar (Armijo Parameter für das Testen der Bedinungung),
        tau: Skalar (Tolerenz für Kriterium 1),
        eps: Skalar (Tolerenz für Gradient in GMW-Kriterium 3),
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

    # Initialisierung
    k = 0
    xk = x0

    # Iteration
    while True:
        valk, gradk, hessk = f(xk)
        log['x_list'].append(xk)
        log['val_list'].append(valk)
        log['norm_grad_list'].append(np.linalg.norm(gradk))

        # Newton Schritt
        pk = np.linalg.solve(hessk, -gradk)
        alphak = armijo(f, pk, xk, rho=rho, c1=c1)  # ---- NEU: Armijo Liniensuche ----
        xkplus1 = xk + alphak * pk

        if gill_murray_wright(xk, xkplus1, valk, valkplus1=f(xkplus1)[0], gradk=gradk, k=k,  tau=tau, eps=eps,
                              kmax=kmax):
            break

        # Update
        k += 1
        xk = xkplus1

    log['xnv'] = xk
    log['knv'] = k
    return log


# Uebernommen von Blatt 9
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

    # Initialisierung
    xk = x0
    valk, gradk, _ = f(xk)
    Bk = B0
    k = 0
    I = np.eye(xk.size)

    # Iteration
    while True:
        if plot:
            log['x_list'].append(xk)
            log['val_list'].append(valk)
            log['norm_grad_list'].append(np.linalg.norm(gradk))

        # Iterationsschritt
        pk = - Bk @ gradk
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

        rhok = 1 / (yk @ sk)
        Bkplus1 = (I - rhok * np.outer(sk, yk)) @ Bk @ (I - rhok * np.outer(yk, sk)) + rhok * np.outer(sk, sk)

        # Warnung, wenn yk @ sk nicht > 0
        if sk @ yk < 1e-12:
            print(f"ACHTUNG: sk @ yk = {sk @ yk}")

        # Update
        xk, valk, gradk = xkplus1, valkplus1, gradkplus1
        Bk = Bkplus1
        k += 1

    log['xqn'] = xk
    log['kqn'] = k
    return log


# Uebernommen von Blatt 8
def rosenbrock(x, y):
    """
    Implementierung der Rosenbrock-Funktion f(x, y) = 100(y-x**3)**3 + (1-x)**3

    Input:
        x, y: Skalare bzw. Arrays zur Auswertung der Funktion.

    Output:
        val: Zielfunktionswert(e),
        grad: Gradient(en),
        hess: Hessematri(x/zen).
    """
    val = 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
    grad = np.stack((- 400 * x * (y - x ** 2) - 2 * (1 - x), 200 * (y - x ** 2)), axis=-1)
    hess = np.stack((np.stack((-400 * (y - x ** 2) + 800 * x ** 2 + 2, -400 * x), axis=-1),
                     np.stack((-400 * x, 200 * np.ones_like(x)), axis=-1)), axis=1)

    return val, grad, hess


# Angepasst von Blatt 8
def plot_iteration_rosenbrock(log, title):
    """
    Funktion zur Erstellung von Plots zum Optimierungsverlauf der Rosenbrock-Funktion anhand der Ergebnisse in log.
    Im ersten Plot wird der Iterationsverlauf auf einem Surface-Plot dargestellt, in weiteren 3 Konvergenzeigenschaften.
    Die Funktion liefert fig zurück. Stellen Sie diese dar, indem Sie plt.show( ) in Ihrer Hauptdatei aufrufen.

    Input:
        log: Dictionary, das mindestens folgende Einträge enthält:
            log = {
                'x0': Vektor,                 # Startwert
                'x_list': Liste,              # Iterierte
                'val_list': Liste,            # Funktion ausgewertet an den Iterierten
                'norm_grad_list': Liste,      # Norm des Gradienten ausgewertet an dem Iterierten
            }
        title: Titel des Plots.

    Output:
        fig: hergestellte Figure mit den Plots
    """

    # Figure initialisieren
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(9, 6)
    fig.tight_layout(pad=3, h_pad=3)
    fig.suptitle(title)
    epsilon = np.finfo(float).eps  # Vermeide Probleme mit Teilen durch 0 in den Plots später

    # Surface-Plot mit Iterationsverlauf
    x = np.linspace(-1.5, 1.5, 651)
    y = np.linspace(-1., 1.5, 651)
    xx, yy = np.meshgrid(x, y)
    zz = rosenbrock(xx.flatten(), yy.flatten())[0].reshape((651, 651))
    CS = ax[0, 0].contour(xx, yy, zz, levels=[2, 25, 75, 150, 300, 500, 700])
    ax[0, 0].clabel(CS, inline=1, fontsize=10)
    ax[0, 0].scatter(*log['x0'], marker='o')
    ax[0, 0].scatter(1, 1, marker='*')
    for k in range(len(log['x_list']) - 1):
        xk = log['x_list'][k]
        xkplus1 = log['x_list'][k + 1]
        ax[0, 0].plot([xk[0], xkplus1[0]], [xk[1], xkplus1[1]])
    ax[0, 0].set_title('Höhenlinien mit Iterationsverlauf')
    ax[0, 0].set_aspect('equal', 'datalim')  # Um die Orthogonalität der Suchrichtungen besser zu visualisieren

    # Plotten der Zielfunktionsfehler
    val_list = log['val_list']
    val_star = 0
    for k, valk in enumerate(val_list):
        ax[0, 1].scatter(k, valk - val_star)
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_title(r'$f(x^k) - f(x^\ast)$')

    # Plotten der Gradientennorm
    norm_grad_list = log['norm_grad_list']
    for k, norm_gradk in enumerate(norm_grad_list):
        ax[1, 0].scatter(k, norm_gradk)
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_title(r'$||\nabla f(x^k) ||$')

    # Plotten der Konvergenz der Iterierten
    x_list = log['x_list']
    xstar = np.array([1, 1])
    for k, xk in enumerate(x_list[1:]):
        ax[1, 1].scatter(k, np.linalg.norm(xk - xstar))
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_title(r'$||x^k - x^\ast||$')

    return fig
