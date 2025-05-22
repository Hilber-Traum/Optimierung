import numpy as np
import matplotlib.pyplot as plt



# Quadratische Funktion, übernommen vom Blatt 5
def quadratic_function(x, A, b, c):
    """
    Quadratische Funktion

    Input:
        x: Variable,
        A: Matrix,
        b: Vektor,
        c: Skalar.

    Output:
        val: Skalar (f(x)),
        grad: Vektor ( \nabla f(x)),
        hess: Matrix ( \nablaˆ2 f(x)).
    """

    # Note: The ndarray x must have size (n,) for one coordinate reps. (m,n) for multiple coordinates
    val = 0.5 * np.einsum('...i,...i->...', x @ A, x) - np.inner(b, x) + c
    grad = x @ A - b  # Note: x @ A instead of A @ x in order to deal with multiple coordinates
    hess = A
    return val, grad, hess


# Gradientenabstiegsverfahren, übernommen vom Blatt 5, erweitert um Speichern der Anzahl der Iterationen.
def gradient_descent(f, x0, tol=1e-10, kmax=100, xstar=None):
    """
    Gradientenabstiegsverfahren

    Input:
        f: Funktion, wobei f(x) = (val, grad, hess),
        x0: Vektor (Startwert),
        tol: Skalar (Toleranz),
        kmax: Skalar (Maximale Anzahl an Iterationen),
        xstar: Vektor (Minimierer von f), optional.

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

        if log['norm_grad_list'][-1] <= tol or k == kmax:
            break

        pk = -gradk
        alphak = - pk @ gradk / (pk @ hessk @ pk)
        xkplus1 = xk + alphak * pk

        k += 1
        xk = xkplus1

    log['xgd'] = xk
    log['kgd'] = k
    return log



# Plot-Funktion, übernommen vom Blatt 5
def plot_iteration_process(log, title, f=None):
    """
    Funktion zur Erstellung von Plots zum Optimierungsverlauf anhand der Ergebnisse in log.
    Die Funktion liefert fig zurück, Stellen Sie sie bitte dar,
    indem Sie anschließend plt.show( ) in Ihrer Hauptdatei aufrufen.

    Input:
        log: Dictionary mit mindestens folgenden Einträgen
            log = {
                'xstar': Vektor,              # Tatsächlicher Minimierer (falls bekannt)
                'val_star': Skalar,           # Tatsächliches Minimum (falls bekannt)
                'x0': Vektor,                 # Startwert
                'x_list': Liste,              # Liste der Iterierten
                'val_list': Liste,            # Liste des Funktionswerts in den Iterierten
                'norm_grad_list': Liste,      # Liste der Norm des Gradienten in den Iterierten
            }
        title: Titel des Plots,
        f: Falls 2D: Optimierte Funktion, wobei f(x) = (val, grad, hess), für Erstellung der Höhenlinien, optional.

    Output:
        fig: Erzeugte Figure mit den Plots
    """
    # In dieser Funktion initialisieren wir die Figure mit den Subplots
    def initialize_figure():
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.set_size_inches(8, 6)
        fig.tight_layout(pad=4.0)
        fig.suptitle(title)
        return fig, axes

    # Berechne Hoehenlinien
    def hoehen_linien():
        xx, yy, zz = None, None, None

        # Definitionsbereich wählen (x0 und xstar helfen abzuschätzen, wie groß dieser sein muss)
        n = 50
        xstar = log['xstar']
        x0 = log['x0']
        xaxis = np.linspace(min(x0[0], xstar[0]) - 5, max(x0[0], xstar[0]) + 5, n)
        yaxis = np.linspace(min(x0[1], xstar[1]) - 3, max(x0[1], xstar[1]) + 3, n)

        # Gitterpunkte und Funktionswerte auf den Gitterpunkten
        xx, yy = np.meshgrid(xaxis, yaxis, sparse=False, indexing='ij')
        zz, _, _ = f(np.concatenate((xx.reshape(n * n, 1), yy.reshape(n * n, 1)), axis=-1))
        zz = zz.reshape(n, n)
        return xx, yy, zz

    # Initialisiere Abbildung
    fig, ax = initialize_figure()
    epsilon = np.finfo(float).eps  # Vermeide Probleme mit Teilen durch 0 in den Plots später

    # Plotten der Höhenlinien - Nur möglich für 2D-Funktionen (dann f als lambda-Funktion übergeben)
    if f is not None:
        ax[0, 0].set_title('Höhenlinien mit Iterationsverlauf')
        ax[0, 0].set_aspect('equal', 'datalim')  # Um die Orthogonalität der Suchrichtungen besser zu visualisieren
        xx, yy, zz = hoehen_linien()
        x0 = log['x0']
        xstar = log['xstar']
        x_list = log['x_list']
        CS = ax[0, 0].contour(xx, yy, zz)
        ax[0, 0].clabel(CS, inline=1, fontsize=10)
        ax[0, 0].scatter(*x0, marker='o')
        ax[0, 0].scatter(*xstar, marker='*')
        for k in range(len(x_list)-1):
            xk = x_list[k]
            xkplus1 = x_list[k+1]
            ax[0, 0].plot([xk[0], xkplus1[0]], [xk[1], xkplus1[1]])

    # Plotten der Zielfunktionsfehler
    ax[0, 1].set_title(r'$f(x^k) - f(x^\ast)$')
    val_list = log['val_list']
    val_xstar = log['val_star'] if log['val_star'] is not None else log['val_list'][-1]
    for k, valk in enumerate(val_list):
        ax[0, 1].scatter(k, valk - val_xstar)
    ax[0, 1].set_ylim(min(0, ax[0, 1].get_ylim()[0]), max(1, ax[0, 1].get_ylim()[1]))

    # Plotten der Konvergenzrate bezüglich der Zielfunktionswerte
    ax[1, 0].set_title(r'$(f(x^k) - f(x^\ast)) / (f(x^{k-1}) - f(x^\ast))$')
    val_list = log['val_list']
    val_xstar = log['val_star'] if log['val_star'] is not None else log['val_list'][-1]
    for k in range(1, len(val_list)):
        valk = val_list[k]
        valkminus1 = val_list[k-1]
        if valkminus1 - val_xstar > epsilon:
            ax[1, 0].scatter(k, (valk - val_xstar) / (valkminus1 - val_xstar))
    ax[1, 0].set_ylim(min(0, ax[1, 0].get_ylim()[0]), max(1, ax[1, 0].get_ylim()[1]))

    # Plotten der Konvergenzrate bezüglich der Iterierten
    ax[1, 1].set_title(r'$||x^k - x^\ast|| / ||x^{k-1} - x^\ast||$')
    x_list = log['x_list']
    xstar = log['xstar'] if log['xstar'] is not None else log['x_list'][-1]
    for k in range(1, len(x_list)):
        xk = x_list[k]
        xkminus1 = x_list[k-1]
        if np.linalg.norm(xkminus1 - xstar) > epsilon:
            ax[1, 1].scatter(k, np.linalg.norm(xk - xstar) / np.linalg.norm(xkminus1 - xstar))
    ax[1, 1].set_ylim(min(0, ax[1, 1].get_ylim()[0]), max(1, ax[1, 1].get_ylim()[1]))

    return fig
