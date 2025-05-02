import numpy as np
import time as t
import matplotlib.pyplot as plt

## TODO: Aufgabenteil 6a. Fehlerhaften Code debuggen
print('### 6a ###')
# Fehlerhafter Code
def summe_und_produkt(a, b):
    c1 = a + b
    c2 = a * b
    c = [c1, c2]
    return c


def summe_plus_produkt(a, b):
    c1, c2 = summe_und_produkt(a, b)
    result = c1 + c2
    return result


result = summe_plus_produkt(2, 3)
print(f'Result = {result}')


#Ein


## TODO: Aufgabenteil 6b. Primzahlen bestimmen
print('\n### 6b ###')
def prime(n):
    #Nutzen das Sieb des Erastosthenes
    primelist = []
    # BEGIN SOLUTION
    #Iteriere über alle zahlen greosser zwei und kleiner n
    for i in range (2, n):
        primcount = len(primelist)
        isprime = True

        #ueberpruefe, ob die aktuelle Zahl restlos durch alle bisher gefundenen primzahlen teilbar ist.
        for j in range(0, primcount):
            if(i % primelist[j] == 0):
                isprime = False
                break

        #Falls die Zahl nicht durch eine bisher gefundene Primzahl teilbar ist, wird sie zum Primzahlarray hinzugefügt
        if(isprime):
            primelist.append(i)

    return primelist
        

    # END SOLUTION

print(f'Alle Primzahlen kleiner als  5: {prime(5)}')
print(f'Alle Primzahlen kleiner als 15: {prime(15)}')
print(f'Alle Primzahlen kleiner als 50: {prime(50)}')
print(f'Alle Primzahlen kleiner als  0: {prime(0)}')


## TODO: Aufgabenteil 6c. Annäherung der Zahl pi
print('\n### 6c ###')
# (i) Funktionen
def leibniz_mit_for_schleifen(N):
    leibniz_pi, leibniz_time = 0, 0
    start_time = t.time()
    # BEGIN SOLUTION
    for k in range(0,N+1):
        leibniz_pi = leibniz_pi + (-1)**k / (2*k + 1)

    leibniz_time = t.time() - start_time

    # END SOLUTION
    return 4 * leibniz_pi, leibniz_time


def leibniz_ohne_for_schleifen(N):
    leibniz_pi, leibniz_time = 0, 0
    # BEGIN SOLUTION
    start_time = t.time()

    #erstellen ein Array A = [1,...,N]
    i = np.arange(N)
    leibniz_arr = (-1)**i / (2*i+1)

    leibniz_pi = 4*np.sum(leibniz_arr)

    leibniz_time = t.time()-start_time
    # END SOLUTION
    return leibniz_pi, leibniz_time


def euler_mit_for_schleifen(N):
    euler_pi, euler_time = 0, 0
    # BEGIN SOLUTION
    start_time = t.time()
    for i in range(1, N+1):
        euler_pi = euler_pi + 1/(i**2)

    euler_time = t.time() - start_time
    euler_pi = (6*euler_pi)**(1/2)
    # END SOLUTION
    return euler_pi, euler_time


def euler_ohne_for_schleifen(N):
    euler_pi, euler_time = 0, 0
    # BEGIN SOLUTION
    start_time = t.time()

    #erstellen ein Array A = [1,...,N]
    i = np.arange(N)
    euler_arr = 1/((i+1)**2)

    euler_pi = (6*np.sum(euler_arr))**(1/2)

    euler_time = t.time()-start_time
    # END SOLUTION
    return euler_pi, euler_time

#Eine Methode, die wie vorgegeben die Ergebnisse der Berechnungen von Pi ausgibt
def pi_berechnungsmethoden_auswertung(value, time, method, iteration_count):
    print(f"Die verwendete Methode war: {method}")
    print(f"Die benötigte Zeit war {time} mit einer Iterationsanzahl von {iteration_count}")
    print(f"Der Fehler zum maschinengenauen Wert lag bei {abs(value - np.pi)} \n")


# (ii) Testen
N1 = int(1e2)
N2 = int(1e7)
# BEGIN SOLUTION
#Für den nachfolgenden Code ist eine Entschuldigung fällig: SORRY!


[value, time] = euler_mit_for_schleifen(N1)
pi_berechnungsmethoden_auswertung(value, time, "Euler, For", N1)

[value, time] = euler_mit_for_schleifen(N2)
pi_berechnungsmethoden_auswertung(value, time, "Euler, For", N2)

[value, time] = euler_ohne_for_schleifen(N1)
pi_berechnungsmethoden_auswertung(value, time, "Euler, Ohne-For", N1)

[value, time] = euler_ohne_for_schleifen(N2)
pi_berechnungsmethoden_auswertung(value, time, "Euler, Ohne-For", N2)

[value, time] = leibniz_mit_for_schleifen(N1)
pi_berechnungsmethoden_auswertung(value, time, "Leibniz, For", N1)

[value, time] = leibniz_mit_for_schleifen(N2)
pi_berechnungsmethoden_auswertung(value, time, "Leibniz, For", N2)

[value, time] = leibniz_ohne_for_schleifen(N1)
pi_berechnungsmethoden_auswertung(value, time, "Leibniz, Ohne-For", N1)

[value, time] = leibniz_ohne_for_schleifen(N2)
pi_berechnungsmethoden_auswertung(value, time, "Leibniz, Ohne-For", N2)

print("Die Varianten, die keine for-Schleife in Python nutzen, sind deutlich schneller, da hier eine interne Schleife genutzt wird, die in c geschrieben ist")


# END SOLUTION

# (iii) Plots
# BEGIN SOLUTION
N = [5,10, 50, 100, 500, 1000]
approx_euler = []
approx_leibniz = []



#Iteriere über die Liste N und speichere die Approximationen der jeweiligen Methoden ab
for i in range(len(N)):
    [value, time] = euler_ohne_for_schleifen(N[i])
    approx_euler.append(abs(value - np.pi))

    [value, time] = leibniz_ohne_for_schleifen(N[i])
    approx_leibniz.append(abs(value - np.pi))


#Plotte die Approximationen
plt.plot(N, approx_euler)
plt.plot(N, approx_leibniz)
plt.title("Approximationsfehler von Euler und Leibniz für Pi")
plt.xlabel("Anzahl Summanden")
plt.ylabel("Approximation")
plt.show()

print("Der Plot zeigt, dass beide Varianten gleich schnell gute Ergebnisse liefern. Lässt man sich den Fehler in der Konsole ausgeben,sieht man, dass die Variante von Euler minimal genauer ist.")

# END SOLUTION
