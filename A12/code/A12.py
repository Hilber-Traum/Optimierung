# TODO: Benötigte Pakete importieren
# BEGIN SOLUTION
import cvxpy as cp
import numpy as np
from utils import plot_surface
import matplotlib.pyplot as plt
# END SOLUTION


# TODO: Aufgabenteil 2i. 1. Problem lösen.
print('### Aufgabenteil 2i ###\n')
# BEGIN SOLUTION
# initialisieren unser x
x = cp.Variable(2)
x1 = x[0]
x2 = x[1]

# legen Funktion fest, die zu Minimieren ist
obj = cp.Minimize( (x1 - 3/2)**2 + (x2 - 1/2)**4 )

# Definieren nebenbedingungen
nebenbedingungen = [x1 + x2 - 1 <= 0,
                    x1 - x2 - 1 <= 0,
                    -x1 + x2 - 1 <= 0,
                    -x1 - x2 - 1 <= 0]

#lösen das Problem
problem = cp.Problem(obj, nebenbedingungen)
#minimum = problem.solve(verbose = True)
minimum = problem.solve()

print("Verwendete Solver:", problem.solver_stats.solver_name)
print("Optimierungsstatus:", problem.status)
print("ANzahl Iterationen:", problem.solver_stats.num_iters)
print("benötigte Rechenzeit: ", problem.solver_stats.solve_time, "s")
print("Bestimmter Minimierer: x* = ", x.value)
print("Bestimmtes Minimum: f(x*) = ", minimum)

# END SOLUTION

# TODO: Aufgabenteil 2ii. Iterierte bestimmen und visualisieren.
print('\n### Aufgabenteil 2ii ###\n')
# Hinweis: Verwenden Sie die in utils.py gegebene Funktion
# BEGIN SOLUTION
anzahl_iterationen = int(problem.solver_stats.num_iters)
x_list = []
for i in range(anzahl_iterationen+1):
    opt_loop = problem.solve(max_iter = i)
    x_list.append([x.value[0], x.value[1]])
    print(f"Iteration: {i}, bestimmter Minimierer: [{x.value[0]:.8e},{x.value[1]:.8e}]")
fig = plot_surface(x_list)
plt.show()
# END SOLUTION

# TODO: Aufgabenteil 2iii. Problem mit Solver SCS lösen
print('\n### Aufgabenteil 2iii ###\n')
# BEGIN SOLUTION
#lösen das Problem mit SCS
#minimum = problem.solve(solver=cp.SCS, verbose = True)
minimum = problem.solve(solver=cp.SCS)

print("Verwendete Solver:", problem.solver_stats.solver_name)
print("Optimierungsstatus:", problem.status)
print("ANzahl Iterationen:", problem.solver_stats.num_iters)
print("benötigte Rechenzeit: ", problem.solver_stats.solve_time, "s")
print("Bestimmter Minimierer: x* = ", x.value)
print("Bestimmtes Minimum: f(x*) = ", minimum)

print("SCS ist mehr als doppelt so schnell wie CLARABEL, benötigt jedoch mehr als 12 mal so viele Iterationen\n"
      "(wird durch Abbruchbedingung gestoppt), der bestimmte Minimierer und das Minimum unterscheiden sich unwesentlich.\n"
      "In beiden Fällen wird die Lösung als optimal angegeben. Allerdings wird auc eine Warnung ausgegeben:\n"
      "'Solution may be inaccurate.'.")
# END SOLUTION

# TODO: Aufgabenteil 3. Ottos Optimierungsproblem lösen
print('\n### Aufgabenteil 3 ###\n')
# BEGIN SOLUTION
n = 2
# initialisieren unser x und einen Hilfsvektor
x_otto = cp.Variable(n+1)
zeros_5 = np.zeros(n+1)
zeros_5[-1] = -5

# Definieren zu minimierende Funktion
obj_otto = cp.Minimize(cp.sum_squares(x_otto-zeros_5)-5)
# Definieren nebenbedingungen
nebenbedingungen_otto = [cp.norm(x_otto[:-1]) + x_otto[-1]/5 <= 0,
                         2 * cp.sum_squares(x_otto[:-1]) - 7 - x_otto[-1] <= 0]

#lösen das Problem
problem_otto = cp.Problem(obj_otto, nebenbedingungen_otto)
#problem_otto.solve(verbose = True)
problem_otto.solve()

#lösen das Problem für verschiedene n
n_werte = [1,3,100000]
for n in n_werte:
    x_otto = cp.Variable(n+1)
    zeros_n = np.zeros(n+1)
    zeros_n[-1] = -5
    obj_otto = cp.Minimize(cp.sum_squares(x_otto-zeros_n)-5)
    nebenbedingungen_otto = [cp.norm(x_otto[:-1]) + x_otto[-1]/5 <= 0,
                             2 * cp.sum_squares(x_otto[:-1]) - 7 - x_otto[-1] <= 0]
    problem_otto = cp.Problem(obj_otto, nebenbedingungen_otto)
    minimum = problem_otto.solve()
    print(f"n={n}, Anzahl Iterationen: {problem.solver_stats.num_iters} , benötigte Rechenzeit: {problem.solver_stats.solve_time}s, Minimierer: {x_otto.value}, Minimum: {minimum}")
# END SOLUTION


# TODO: Aufgabenteil 4. Letztes Problem nicht lösen :D
print('\n### Aufgabenteil 4 ###\n')
# BEGIN SOLUTION
#definieren analog wie vorher
x_4 = cp.Variable(2)
obj_4 = cp.Minimize(-(x[0]+1)**2-(x[1]+1)**2)
nebenbedingungen_4 = [cp.sum_squares(x) - 2 <= 0,
                      x[0] - 2**(1/2) <= 0]
problem_4 = cp.Problem(obj_4, nebenbedingungen_4)
# problem_4.solve() #gibt Fehlermeldung "Funktion ist konkav"
print("Das lösen schlägt Fehl, da die Funktion unter den Nebenbedingungen Konkav ist.") #??????????????????????
# END SOLUTION