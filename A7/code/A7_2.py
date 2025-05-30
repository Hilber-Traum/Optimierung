import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # wird für colorbar benötigt
from utils import *

def newton(f, x0, tol=1e-10, kmax=100, xstar=None):
    """
    Newton-Verfahren zur Minimierung der Funktion f.

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
        'val_star': None if xstar is None else (f(xstar))[0],  # Tatsächliches Minimum
        'x0': x0,                 # Startwert
        'xnv': None,              # Lösung des Gradientenabstiegverfahrens
        'knv': None,              # Anzahl benötigter Iterationen
        'x_list': [],             # Liste der Iterierten
        'val_list': [],           # Liste des Funktionswerts in den Iterierten
        'norm_grad_list': [],     # Liste der Norm des Gradienten in den Iterierten
    }

    # TODO: Aufgabenteil 1. Newtonverfahren implementieren.
    # BEGIN SOLUTION
    #Startwerte abspeichern
    log['x0'] = x0
    log['xstar'] = xstar

    
    xk = x0
    #Berechne den ersten Gradienten, Wert und Hessematrix und Norm von grad und speichere sie
    val, grad, hess = f(xk)
    normgradf = np.linalg.norm(grad)
    log['norm_grad_list'].append(normgradf)
    log['val_list'].append(val)
    log['x_list'].append(xk)

    #Berechnungs-loop nach Skript
    k = 0
    while(k < kmax and normgradf > tol):
        #Berechne p^k
        pk = np.matvec(np.linalg.inv(hess), -grad) 

        #Berechne x^(k+1) (alpha^k ist ja 1, da es ungedaempft ist)
        xk = xk + pk

        #Berechne neuen Gradienten, Wert und Hessematrix und Norm von grad
        val, grad, hess = f(xk)
        normgradf = np.linalg.norm(grad)

        #speichere alle neuen Werte ab
        log['val_list'].append(val)
        log['x_list'].append(xk)
        log['norm_grad_list'].append(normgradf)

        k += 1

        log['xnv'] = xk
        log['knv'] = k

    # END SOLUTION
    return log


def objective_fun(x1, x2):
    """
    Implementierung der Funktion f(x1, x2) = x1**4 + x1*x2 + (1+x2)**2
    gradf(x1,x2) = (4*x1**3 + x2, x1 + 2+2*x2))
    gradgradf(x1,x2) = (12*x**2, 2)

    Input:
        x1, x2: Skalare bzw. Arrays zur Auswertung der Funktion.

    Output:
        val: Zielfunktionswert(e),
        grad: Gradient(en),
        hess: Hessematri(x/zen).
    """
    val, grad, hess = None, None, None

    # TODO: Aufgabenteil 2.i. Funktion ergaenzen.
    #   Hinweis: np.stack koennte hilfreich sein - achtet dabei darauf, die richtige axis zu waehlen

    # BEGIN SOLUTION
    #Falls m = 1
    #Berechnet die Funktion, ihren Gradienten und Hessematrix. Für m>1 gibt es einen extra-Fall (sorry ist etwas messy, funktioniert aber hoffentlich)
    if(not isinstance(x1, (list, tuple, np.ndarray, ))):
        val = x1**4 + x1*x2 + (1+x2)**2
        grad = np.array([4*x1**3 + x2, x1 + 2+2*x2])
        hess = np.array([np.array([12*x1**2, 1]), np.array([1, 2])])
    else:
        val, grad, hess = np.array([]),np.array([]),np.array([])
        val = np.append(val, x1[0]**4 + x1[0]*x2[0] + (1+x2[0])**2)
        grad = [4*x1[0]**3 + x2[0], x1[0] + 2+2*x2[0]]
        hess =  [[12*x1[0]**2, 1], [1, 2]]
        for i in range(1,len(x1)):
            val = np.append(val, x1[i]**4 + x1[i]*x2[i] + (1+x2[i])**2)
            grad = np.stack((grad,[4*x1[i]**3 + x2[i], x1[i] + 2+2*x2[i]]))
            hess = np.stack((hess, [[12*x1[i]**2, 1], [1, 2]]))
        

    # END SOLUTION

    return val, grad, hess


if __name__ == '__main__':
    # Sanity check fuer objective_fun. Sie brauchen hier nichts aendern.
    val, grad, hess = objective_fun(3, 4)
    val_correct, grad_correct, hess_correct = 118, np.array([112, 13]), np.array([[108, 1], [1, 2]])
    for (v, v_correct, v_which) in \
            zip((val, grad, hess), (val_correct, grad_correct, hess_correct), ('val', 'grad', 'hess')):
        if not  np.array_equal(v, v_correct):
            print(f'----\nSanity check fuer objective_fun in einzelnem Vektor: {v_which} scheint nicht zu stimmen.')
            print(f'Ihre Rechnung: {v_which} = \n{v}')
            print(f'     Erwartet: {v_which} = \n{v_correct}')
            break
    val, grad, hess = objective_fun(np.array([2, 3]), np.array([5, 4]))
    val_correct, grad_correct, hess_correct = np.array([62, 118]), \
        np.array([[37, 14], [112, 13]]), np.array([[[48, 1], [1, 2]], [[108, 1], [1, 2]]])
    for (v, v_correct, v_which) in \
            zip((val, grad, hess), (val_correct, grad_correct, hess_correct), ('val', 'grad', 'hess')):
        if not  np.array_equal(v, v_correct):
            print(f'----\nSanity check fuer objective_fun in mehreren Vektoren: {v_which} scheint nicht zu stimmen.')
            print(f'Ihre Rechnung: {v_which} = \n{v}')
            print(f'     Erwartet: {v_which} = \n{v_correct}')
            break


    # TODO: Aufgabenteil 2.ii. Estelle surface-plot von objective_fun
    #   Siehe: Musterloesung A0_4.py oder https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
    # BEGIN SOLUTION
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    #Erstelle den Funktionsplot (surfplot) nach Aufgabenstellung
    x = np.linspace(-3.25, 3.25, 651)
    X, Y = np.meshgrid(x, x)
    Z = X**4 + X*Y + (1+Y)**2
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

    #Plot beschriften
    plt.title("f(x,y) = x^4 + x*y + (1+y)^2")
    plt.xlabel('x-Axis')
    plt.ylabel('y-Axis')
    plt.show()

    
    # END SOLUTION


    # TODO: Aufgabenteil 3. Test fuer 2 verschiedene Startwerte
    # Berechne xstar
    xstar2 = (1/4) * ((-4 - np.cbrt(36-np.sqrt(1290)) / 6**(2/3)) - 1/np.cbrt(6 * (36 - np.sqrt(1290))))
    xstar1 = -2 * (1 + xstar2)
    xstar = np.array([xstar1, xstar2])

    kmax = 100
    tol = 1e-10
    f = lambda x: objective_fun(*x.T)  # Wichtig, damit Funktionsaufruf in plot-Funktion fuer x.shape=(m,2) funktioniert

    # BEGIN SOLUTION
    startval_1 = np.array([3,0])
    startval_2 = np.array([-2,-2])

    #Zweimaliges Testen des Newton-Verfahrens mit anschließender Ausgabe des gefundenen Minimieres und Anzahl Schritte
    #1. Startwert
    log = newton(f, startval_1, tol, kmax, xstar)
    crit_point = log['xnv']
    iteration_count = log['knv']
    print(f'Nach Newton Verfahren ist {crit_point} ein kritischer Punkt')
    print(f'Im Newton Verfahren wurden {iteration_count} Schritte gemacht')

    #Ergebnis plotten
    fig = plot_iteration_process_2d(log, 'Newton, Startpunkt 1')
    plt.show()

    #2. Startwert
    log = newton(f, startval_2, tol, kmax, xstar)
    crit_point = log['xnv']
    iteration_count = log['knv']
    print(f'Nach Newton Verfahren ist {crit_point} ein kritischer Punkt')
    print(f'Im Newton Verfahren wurden {iteration_count} Schritte gemacht')

    #Ergebnis plotten
    fig = plot_iteration_process_2d(log, 'Newton, Startpunkt 2')
    plt.show()

    # END SOLUTION


    # TODO: Aufgabenteil 4. Test mit tol1.
    #   Hinweis: Das Plot-Argument zorder kann beim Plotten auf 3D-Achsen dafür verwendet werden, Elemente in den
    #   Hintergrund oder Vordergrund zu verschieben.
    #   Siehe: https://matplotlib.org/stable/gallery/misc/zorder_demo.html#sphx-glr-gallery-misc-zorder-demo-py
    tol1 = 1e-3

    # Abbildung vorbereiten
    fig = plt.figure()
    ax0 = fig.add_subplot(121, projection='3d')
    ax0.set_title('Surface plot \nmit Start und Endpunkten der Iterationen')
    ax1 = fig.add_subplot(122)
    ax1.set_title('Anzahl benötigter Iterationen')
    fig.suptitle(f'tol = {tol1}')

    # Algorithmus auf Startwerte anwenden und Plots ergaenzen
    # BEGIN SOLUTION
    iterations = []
    startingpoints = []
    endpoints = []

    #Berechne den Minimierer mit 300 verschiedenen Startwerten. Speichere Startwerte und Minimierer ab
    for l in range(0,300):
        startpoint = np.array([3*np.cos(2*np.pi*l/300), 3*np.sin(2*np.pi*l/300)])
        log = newton(f, startpoint, tol1, kmax)
        iterations.append(log['knv'])
        crit_point = log['xnv']

        #Speichere Start-und Endpunkte
        startingpoints.append([3*np.cos(2*np.pi*l/300), 3*np.sin(2*np.pi*l/300)])
        endpoints.append(crit_point)
    
    sum_mild_tol = np.sum(iterations)

    #Plotte die Startwerte und Minimierer als Scatterplot
    X, Y = np.meshgrid(startingpoints, endpoints)
    Z = X**4 + X*Y + (1+Y)**2
    surf = ax0.scatter(X, Y, Z)

    plt.plot(range(0, 300), iterations)

    # END SOLUTION

    fig.set_size_inches(9, 3)
    fig.tight_layout(pad=.5, h_pad=0)
    plt.show()


    # TODO: Aufgabenteil 5. Test mit tol2.
    tol2 = 1e-15

    # Abbildung vorbereiten
    fig = plt.figure()
    ax0 = fig.add_subplot(121, projection='3d')
    ax0.set_title('Surface plot \nmit Start und Endpunkten der Iterationen')
    ax1 = fig.add_subplot(122)
    ax1.set_title('Anzahl benötigter Iterationen')
    fig.suptitle(f'tol = {tol2}')

    # Algorithmus testen und Plots ergänzen
    # BEGIN SOLUTION
    iterations = []
    startingpoints = []
    endpoints = []

    #Berechne den Minimierer mit 300 verschiedenen Startwerten. Speichere Startwerte und Minimierer ab
    for l in range(0,300):
        startpoint = np.array([3*np.cos(2*np.pi*l/300), 3*np.sin(2*np.pi*l/300)])
        log = newton(f, startpoint, tol2, kmax)
        iterations.append(log['knv'])
        crit_point = log['xnv']

        #Speichere Start-und Endpunkte
        startingpoints.append([3*np.cos(2*np.pi*l/300), 3*np.sin(2*np.pi*l/300)])
        endpoints.append(crit_point)
    
    #Plotte die Startwerte und Minimierer als Scatterplot
    X, Y = np.meshgrid(startingpoints, endpoints)
    Z = X**4 + X*Y + (1+Y)**2
    surf = ax0.scatter(X, Y, Z)

    plt.plot(range(0, 300), iterations)

    sum_strict_tol = np.sum(iterations)
    # END SOLUTION

    fig.set_size_inches(9, 3)
    fig.tight_layout(pad=.5, h_pad=0)

    plt.show()


    # TODO: Aufgabenteil 5. Diskussion.
    # BEGIN SOLUTION
    print(f"Ist die Toleranz niedriger, so benötigt das Verfahren im Schnitt weniger Iterationen. Dies sieht man daran, dass die Schleife mit der niedrigeren Toleranz {sum_strict_tol - sum_mild_tol} Iterationen weniger benötigt hat. \n Dafür sind die Ergebnisse mit der niedrigeren Toleranz natürlich genauer, als die Ergebnisse mit der hohen Toleranz. \n Dies ist verständlich, da eine niedrigere Toleranz zu einem früheren Abbruch des Verfahrens führt (und da sich das Verfahren bei jedem Schritt verbessert, sorgt ein frührere Abbruch zu ungenauerem Ergebnis.)")
    # END SOLUTION


