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
        # Zuweisungen
        xm=xk-dk
        xp=xk+dk
        y0=f(xk)
        ym=f(xm)
        yp=f(xp)
        # Ausgabe
        print(f"Iteration {k}:")
        print(f"  x- = {xm}, f(x-) = {ym}")
        print(f"  x0 = {xk}, f(x°) = {y0:}")
        print(f"  x+ = {xp}, f(x+) = {yp:}")

        # Loggen der Werte
        log["x_list"].append(xk)
        log["val_list"].append(y0)

        # Fallunterscheidungen
        if ym<y0 and yp<y0:
            dk = 0.5 * dk
            if ym<yp:
                xk=xm
            else:
                xk=xp
        elif ym<y0 and yp>=y0:
            xk=xm
            dk=2*dk
        elif ym>=y0 and yp<y0:
            xk=xp
            dk=2*dk
        elif ym>=y0 and yp>=y0:
            xk=xk
            dk=0.5*dk
        # Aufgabenteil c)
        if k<20:
            plot_xm.set_data([xm], [ym])
            plot_xo.set_data([xk], [y0])
            plot_xp.set_data([xp], [yp])
            plt.draw()
            plt.pause(0.25)


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
    f1_x = petras_optimizer(f, x0=2, d0=0.1, plot=[-4,4])
    figure1 = plot_iteration_process(f1_x, r'Iterationsverlauf für $f(x)=x^2$ mit x0=2, d0=0.1')
    plt.show()
    f2_x = petras_optimizer(f, x0=-10, d0=1, plot=[-15,15])
    figure2 = plot_iteration_process(f2_x, r'Iterationsverlauf für $f(x)=x^2$ mit x0=-10, d0=1')
    plt.show()

    print("Diskussion der Ergebnisse: \n"
          "Im ersten Durchlauf aus d) wird die Schrittweite im Vergleich zu vorher verringert. \n"
          "Der Startwert bleibt der gleiche. Das Verfahren benötigt mehr Iterationen um auf den Minimierer zu kommen.\n"
          "Im zweiten Durchlauf ist die Schrittweite wieder 1, der Startwert jedoch weiter weg vom Minimierer. \n"
          "Auch hier braucht es mehr Iterationen um auf den Minimierer zu kommen.\n"
          "Das wählen eines sinnvollen Startwertes und einer geeigneten Schrittweite beschleunigen das Verfahren.\n"
    )
    # END SOLUTION

    # TODO: Aufgabenteil e. Verfahren für f(x) = x**4 - 4 * x**2 testen
    # BEGIN SOLUTION
    f2 = lambda x: x**4 - 4*x**2
    f3_x = petras_optimizer(f2, x0=0, d0=1, plot=[-5,5])
    figure3 = plot_iteration_process(f3_x, r'Iterationsverlauf für $f(x)=x^4-4x^2$ mit x0=0, d0=1')
    plt.show()
    # END SOLUTION

    # TODO: Aufgabenteil f. Verfahren für f(x) = x und f(x) = cos(pi * x) + 1 / (1 + x**2) testen
    # BEGIN SOLUTION
    f3 = lambda x: x
    f4_x = petras_optimizer(f3, x0=0, d0=1, plot=[-10**6, 5])
    figure4 = plot_iteration_process(f4_x, r'Iterationsverlauf für $f(x)=x$ mit x0=0, d0=1')
    plt.show()

    f4 = lambda x: np.cos(np.pi * x) + 1 / (1 + x ** 2)
    f5_x = petras_optimizer(f4, x0=1, d0=2, plot=[-5, 2.5*10**6])
    figure5 = plot_iteration_process(f5_x, r'Iterationsverlauf für $f(x) = cos(\pi x) + \frac{1}{1 + x^2}$ mit x0=1, d0=2')
    plt.show()

    print(
        "Diskussion der Ergebnisse:\n"
        "Bei $f(x) = x$ lässt sich kein Minimierer finden, weshalb die Schrittweite sich immer verdoppelt. \n"
        "Offensiochtlich kann kein Minimierer gefunden werden, was sich aber nicht durch ausschließliches anwenden des Verfahren zeigen lassen könnte.\n"
        "Bei $f(x) = cos(pi * x) + 1 / (1 + x**2)$ hat bei zwei lokalen Minima das mit Betragsmäßig größerem x-Wert \n"
        "auch einen niedrigeren Funktionswert. Daher findet sich immerwieder ein besserer Wert je weiter die Iterationen durchlaufen.\n"
        "Auch hier mit immer größeren Schrittweiten, bis dann die Maschinengenauigkeit erreicht ist. Dann sinkt die Schrittweite wieder.\n"
        "Der innerhalb der Maschinenzahlen beste Minimierer wurde gefunden."
        "Der wesentliche Unterschied zwischen den beiden Beispielen ist, dass das die Minima bei der zweiten Funktionen augenscheinlich konvergieren "
        "während bei der ersten Funktion dies nicht der Fall ist.\n"
        "Somit kann zumindest bei der zweiten Funktion der Algorithmus einen nicht ganz 'unsinnigen' Wert bestimmen"
    )
    # END SOLUTION