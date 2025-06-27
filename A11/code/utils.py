import numpy as np
import matplotlib.pyplot as plt

# Quadratische Funktion, angepasst von Blatt 5. ACHTUNG: Hier x.T @ A @ x + b.T @ x + c -> positiver linearer Anteil!
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
    val = 0.5 * np.einsum('...i,...i->...', x @ A, x) + np.inner(b, x) + c
    grad = x @ A + b  # Note: x @ A instead of A @ x in order to deal with multiple coordinates
    hess = A
    return val, grad, hess


# Plot-Funktion, übernommen vom Blatt 5
def plot(f, g, log, xstar, title):
    """
    Funktion zur Erstellung von Plots zum Optimierungsverlauf anhand der Ergebnisse in log.
    Es werden Höhenlinien und Iterationsverlauf auf einem 2D-Gitter sowie Konvergenzeigenschaften dargestellt.
    Die Funktion liefert fig zurück. Stellen Sie diese dar, indem Sie plt.show( ) in Ihrer Hauptdatei aufrufen.

    Input:
        f: Optimierte Funktion
        log: Dictionary, das mindestens folgende Einträge enthält:
            log = {
                'x0': Vektor,                 # Startwert
                'x_list': Liste,              # Iterierte
                'val_f_list': Liste,          # Funktion ausgewertet an den Iterierten
                'g_res_list': Liste           # Residuen der Nebenbedingungen in den Iterierten in 1-Norm
            }
        xstar: Vektor, Tatsächliche Lösung des Optimierungsproblems
        title: Titel des Plots.

    Output:
        fig: hergestellte Figure mit den Plots
    """
    # In dieser Funktion initialisieren wir die Figure mit den Subplots
    def initialize_figure():
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.set_size_inches(8, 6)
        fig.tight_layout(pad=4.0)
        fig.suptitle(title)
        return fig, axes

    # Berechne Hoehenlinien
    def hoehen_linien(fctn):
        # Definitionsbereich wählen (x0 und xstar helfen abzuschätzen, wie groß dieser sein muss)
        n = 50
        x0 = log['x0']
        xaxis = np.linspace(min(x0[0], xstar[0]) - 5, max(x0[0], xstar[0]) + 5, n)
        yaxis = np.linspace(min(x0[1], xstar[1]) - 3, max(x0[1], xstar[1]) + 3, n)

        # Gitterpunkte und Funktionswerte auf den Gitterpunkten
        xx, yy = np.meshgrid(xaxis, yaxis, sparse=False, indexing='ij')
        zz, _, _ = fctn(np.concatenate((xx.reshape(n * n, 1), yy.reshape(n * n, 1)), axis=-1))
        zz = zz.reshape(n, n)
        return xx, yy, zz

    # Initialisiere Abbildung
    fig, ax = initialize_figure()

    # Plotten der Höhenlinien
    ax[0, 0].set_title('Höhenlinien mit Iterationsverlauf')
    ax[0, 0].set_aspect('equal', 'datalim')  # Um die Orthogonalität der Suchrichtungen besser zu visualisieren

    xx, yy, zzf = hoehen_linien(f)
    x0 = log['x0']
    x_list = log['x_list']
    CSf = ax[0, 0].contour(xx, yy, zzf)
    ax[0, 0].clabel(CSf, inline=1, fontsize=10)

    if isinstance(g, list):
        for i, gi in enumerate(g):
            _, _, zzg = hoehen_linien(gi)
            CSg = ax[0, 0].contour(xx, yy, zzg, levels=[0], colors='red', linewidths=2)
            ax[0, 0].clabel(CSg, inline=True, fmt=rf'$g_{i}(x)=0$', fontsize=10)
    else:
        _, _, zzg = hoehen_linien(g)
        CSg = ax[0, 0].contour(xx, yy, zzg, levels=[0], colors='red', linewidths=2)
        ax[0, 0].clabel(CSg, inline=True, fmt='g(x)=0', fontsize=10)

    ax[0, 0].scatter(*x0, marker='o')
    ax[0, 0].scatter(*xstar, marker='*')
    for k in range(len(x_list)-1):
        xk = x_list[k]
        xkplus1 = x_list[k+1]
        ax[0, 0].plot([xk[0], xkplus1[0]], [xk[1], xkplus1[1]])

    # Plotten der Zielfunktionsfehler
    ax[0, 1].set_title(r'$f(x^k) - f(x^\ast)$')
    val_list = log['val_f_list']
    val_star = f(xstar)[0]
    for k, valk in enumerate(val_list):
        ax[0, 1].scatter(k, valk - val_star)

    # Plotten der Residuen der Gleichungsnebenbedingungen
    ax[1, 0].set_title(r'$|g_1(x^k)| + ... + |g_m(x^k)|$')
    g_res_list = log['g_res_list']
    for k, g_res_k in enumerate(g_res_list):
        ax[1, 0].scatter(k, g_res_k)
    ax[1, 0].set_yscale('log')

    # Plotten der Konvergenz der Iterierten
    ax[1, 1].set_title(r'$||x^k - x^\ast||$')
    x_list = log['x_list']
    for k, xk in enumerate(x_list):
        ax[1, 1].scatter(k, np.linalg.norm(xk - xstar))
    ax[1, 1].set_yscale('log')

    return fig
