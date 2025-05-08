import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import *


def petras_optimizer(f, x0=0, d0=1, kmax=50, plot=False):
    """
    Petras 1D-Optimierungsverfahren

    Input:
        f: Funktion,
        x0: float (Startwert),
        d0: float (Initialer Abstand),
        kmax: Skalar (Maximale Anzahl an Iterationen),
        plot: Parameter zur Steuerung eines Live-Plots im Iterationsverlauf:
                False => kein Plot
                [xmin, xmax] => Erstellung eines Plot auf Intervall (xmin, xmax)

    Output:
        log: Dictionary vom Verlauf der Optimierung.
    """
    # Dictionary zum Abspeichern der Ergebnisse.
    log = {
        'x0': x0,                 # Startwert
        'd0': d0,                 # Initialer Abstand
        'xpetra': None,           # Erreichter Minimierer
        'x_list': [],             # Liste der Iterierten (xk)
        'val_list': [],           # Liste von Funktionswerten der Iterierten (f(xk))
    }

    print(f'### Starte Petras Optimierung für x0={x0}, d0={d0} ###')
    # Initialisiere Verfahren
    xk = x0
    dk = d0

    # Plot initialisieren
    if plot is not False:
        matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your system
        plt.ion()
        x = np.linspace(plot[0], plot[1], 200)
        y = f(x)
        _, ax = plt.subplots()
        ax.plot(x, y, label='f(x)')
        # Initialisiere Punkte für Plot. Koordinaten können mittels plot_xo.set_data([x], [y]) aktualisiert werden
        plot_xo, = ax.plot([], [], 'ko', label=r'$x_\circ$')  # Plot-Punkt der aktuellen Iterierten x_\circ
        plot_xm, = ax.plot([], [], 'ro', label=r'$x_-$')  # Plot-Punkt des aktuellen x_-
        plot_xp, = ax.plot([], [], 'ro', label=r'$x_+$')  # Plot-Punkt des aktuellen x_+
        plt.legend()

    # TODO: Aufgabenteil b / c. Verfahren implementieren / Live-Visualisierung implementieren
    # BEGIN SOLUTION
    for k in range(kmax):
        xm=xk-dk
        xp=xk+dk
        y0=f(xk)
        ym=f(xm)
        yp=f(xp)
        print(f"Iteration {k}:")
        print(f"  x- = {xm}, f(x-) = {ym}")
        print(f"  x0 = {xk}, f(x°) = {y0:}")
        print(f"  x+ = {xp}, f(x+) = {yp:}")

        if ym<y0 and yp<y0:
            if ym<yp:
                xk=xm
            else:
                xk=xp
            dk=0.5*dk
        elif ym<y0 and yp>=y0:
            xk=xm
            dk=2*dk
        elif ym>=y0 and yp<y0:
            xk=xp
            dk=2*dk
        elif ym>=y0 and yp>=y0:
            print("ERRRROR")
            xk=x0
            dk=0.5*dk
    # END SOLUTION

    log['xpetra'] = xk
    print(f'### Minimaler Wert f(x)={f(xk): .4f} gefunden in xk={xk: .8f} ###\n')

    if plot is not False:
        plt.ioff()
        plt.show()

    return log


if __name__ == '__main__':
    # Teste Optimierungsverfahren für f(x) = x**2
    f = lambda x: x**2
    log = petras_optimizer(f, x0=2, d0=1, plot=[-2,4])
    fig = plot_iteration_process(log, r'Iterationsverlauf für $f(x)=x^2$ mit x0=2, d0=1')
    plt.show()

    # Hinweis: Die plot-Funktion aus utils.py soll für publish.py nur die Plots erstellen und die Figure zurückgeben.
    # Stellen Sie die Abbildung bitte dar, indem Sie HIER plt.show() aufrufen.

    # TODO: Aufgabenteil d. Verfahren für andere Startwerte testen
    # BEGIN SOLUTION
    ...
    # END SOLUTION

    # TODO: Aufgabenteil e. Verfahren für f(x) = x**4 - 4 * x**2 testen
    # BEGIN SOLUTION
    ...
    # END SOLUTION

    # TODO: Aufgabenteil f. Verfahren für f(x) = x und f(x) = cos(pi * x) + 1 / (1 + x**2) testen
    # BEGIN SOLUTION
    ...
    # END SOLUTION
