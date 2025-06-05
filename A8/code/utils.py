import numpy as np
import matplotlib.pyplot as plt


def plot_iteration_rosenbrock(log, title, rosenbrock):
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