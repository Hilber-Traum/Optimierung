import numpy as np
import matplotlib.pyplot as plt


def plot_iteration_process(log, title, f=None):
    """
    Funktion zur Erstellung von Plots zum Optimierungsverlauf anhand der Ergebnisse in log.
    Die Funktion liefert fig zurück, Stellen Sie sie bitte dar,
    indem Sie anschließend plt.show( ) in Ihrer Hauptdatei aufrufen.

    Input:
        log: Diktionary der Form
            log = {
                'xstar': Vektor,              # Tatsächlicher Minimierer
                'val_star': Skalar,           # Tatsächliches Minimum
                'x0': Vektor,                 # Startwert
                'xgd': Vektor,                # Lösung des Gradientenabstiegverfahrens
                'x_list': Liste,              # Iterierten
                'val_list': Liste,            # Funktion ausgewertet an den Iterierten
                'norm_grad_list': Liste,      # Norm des Gradienten ausgewertet an dem Iterierten
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
    # TODO: Aufgabenteil 5. Ergänze die Funktion hoehen_linien. Die Funktion soll den Definitionsbereich
    #  als Meshgrid xx, yy zurückgeben, sowie die Funktionswerte zz auf diesem Definitionsbereich.
    def hoehen_linien():
        xx, yy, zz = None, None, None

        # BEGIN SOLUTION
        # Grenzen bestimmen
        x_list = np.array(log['x_list'])
        x_min = x_list[:, 0].min()-2
        x_max = x_list[:, 0].max()+2
        y_min = x_list[:, 1].min()-1
        y_max = x_list[:, 1].max()+1

        #Meshgrid erstellen
        xx,yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        # Z-Werte bestimmen
        zz = np.zeros_like(xx)
        for k in range(xx.shape[0]):
            for l in range(xx.shape[1]):
                zz[k,l] = f(np.array([xx[k,l], yy[k,l]]))[0]
        # END SOLUTION
        return xx, yy, zz

    # Initialisiere Abbildung
    fig, ax = initialize_figure()
    epsilon = np.finfo(float).eps  # Vermeide Probleme mit Teilen durch 0 in den Plots später

    # TODO: Aufagabenteil 5. Visualisiere Optimierungsverlauf
    # Plotten der Höhenlinien - Nur möglich für 2D-Funktionen (dann f als lambda-Funktion übergeben)
    if f is not None:
        ax[0, 0].set_title('Höhenlinien mit Iterationsverlauf')
        ax[0, 0].set_aspect('equal', 'datalim')  # Um die Orthogonalität der Suchrichtungen besser zu visualisieren
        # BEGIN SOLUTION
        # Contouren
        xx,yy,zz = hoehen_linien()
        contour = ax[0, 0].contour(xx, yy, zz, cmap='viridis')
        # X-Werte
        x_array = np.array(log['x_list'])
        # Plotsachen
        ax[0, 0].plot(x_array[:, 0], x_array[:, 1], 'o-', label='Pfad')
        ax[0, 0].scatter(log['x0'][0],log['x0'][1],
                         color='red', marker='x', s=120, label='Start')
        if log['xstar'] is not None:
            ax[0, 0].scatter(log['xstar'][0],log['xstar'][1],
                             color='orange' , marker='*', s=150, label='Minimum')
        ax[0, 0].legend()
        # END SOLUTION

    # Plotten der Zielfunktionsfehler
    ax[0, 1].set_title(r'$f(x^k) - f(x^\ast)$')
    # BEGIN SOLUTION
    fx_fehler = np.array(log['val_list'])-log['val_star']
    # Plotsachen
    ax[0, 1].plot(range(len(fx_fehler)), fx_fehler)
    ax[0, 1].grid(True)
    ax[0, 1].set_xlabel("k")
    # END SOLUTION

    # Plotten der Konvergenzrate bezüglich der Zielfunktionswerte
    ax[1, 0].set_title(r'$(f(x^k) - f(x^\ast)) / (f(x^{k-1}) - f(x^\ast))$')
    # BEGIN SOLUTION
    # Nutzen np.finfo(float).eps um nicht durch Null zu teilen
    fx_konvergenz = fx_fehler[1:] / (fx_fehler[:-1] + np.finfo(float).eps)
    # Plotsachen
    ax[1, 0].plot(range(1, len(fx_konvergenz) + 1), fx_konvergenz)
    ax[1, 0].grid(True)
    ax[1, 0].set_xlabel("k")
    # END SOLUTION

    # Plotten der Konvergenzrate bezüglich der Iterierten
    ax[1, 1].set_title(r'$||x^k - x^\ast|| / ||x^{k-1} - x^\ast||$')
    # BEGIN SOLUTION
    x_werte = np.array(log['x_list'])
    xstar = log['xstar']
    # Auch hier nuten wir das eps um nicht durch 0 zu teilen
    x_kov_rate = (np.linalg.norm(x_werte[1:] - xstar, axis=1) /
                  (np.linalg.norm(x_werte[:-1] - xstar + np.finfo(float).eps, axis=1)))
    # Plotsachen
    ax[1, 1].plot(range(1, len(x_werte)), x_kov_rate)
    ax[1, 1].grid(True)
    ax[1, 1].set_xlabel("k")
    # END SOLUTION

    return fig
