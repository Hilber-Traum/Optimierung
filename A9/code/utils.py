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


# Anpepasst von Blatt 8
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

    # Surface-Plot mit Iterationsverlauf
    x = np.linspace(-1., 1.5, 651)
    xx, yy = np.meshgrid(x, x)
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
