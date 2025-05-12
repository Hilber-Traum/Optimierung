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
        ...
        # END SOLUTION

    # Plotten der Zielfunktionsfehler
    ax[0, 1].set_title(r'$f(x^k) - f(x^\ast)$')
    # BEGIN SOLUTION
    ...
    # END SOLUTION

    # Plotten der Konvergenzrate bezüglich der Zielfunktionswerte
    ax[1, 0].set_title(r'$(f(x^k) - f(x^\ast)) / (f(x^{k-1}) - f(x^\ast))$')
    # BEGIN SOLUTION
    ...
    # END SOLUTION

    # Plotten der Konvergenzrate bezüglich der Iterierten
    ax[1, 1].set_title(r'$||x^k - x^\ast|| / ||x^{k-1} - x^\ast||$')
    # BEGIN SOLUTION
    ...
    # END SOLUTION

    return fig
