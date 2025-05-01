import numpy as np
import time
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


## TODO: Aufgabenteil 6b. Primzahlen bestimmen
print('\n### 6b ###')
def prime(n):
    # BEGIN SOLUTION
    return [i for i in range (2,n) if all(i % j != 0 for j in range(2,int(i/2)+1))]
    '''
    Bestimme alle i von 2 bis n-1 für die gilt, dass für alle j kleiner i/2 gilt:
    j ist kein Teiler von i.
    Für alle j größer oder gleich i/2 ist die Überprüfung nicht notwendig, 
    da bereits ein Teiler kleiner i/2 existieren müsste.
    '''
    # END SOLUTION

print(f'Alle Primzahlen kleiner als  5: {prime(5)}')
print(f'Alle Primzahlen kleiner als 10: {prime(15)}')
print(f'Alle Primzahlen kleiner als 25: {prime(50)}')
print(f'Alle Primzahlen kleiner als  0: {prime(0)}')


## TODO: Aufgabenteil 6c. Annäherung der Zahl pi
print('\n### 6c ###')
# (i) Funktionen
def leibniz_mit_for_schleifen(N):
    leibniz_pi, leibniz_time = None, None
    # BEGIN SOLUTION
    time_start = time.time()
    leibniz_pi = 0
    for k in range(N):
        leibniz_pi += (-1)**k / (2*k+1)
    leibniz_pi = 4 * leibniz_pi
    leibniz_time = time.time() - time_start
    # END SOLUTION
    return leibniz_pi, leibniz_time


def leibniz_ohne_for_schleifen(N):
    leibniz_pi, leibniz_time = None, None
    # BEGIN SOLUTION
    time_start = time.time()
    array_N = np.arange(N)
    summen_glieder = (-1)**array_N / (2*array_N+1)
    leibniz_pi = 4 * np.sum(summen_glieder)
    leibniz_time = time.time() - time_start
    # END SOLUTION
    return leibniz_pi, leibniz_time


def euler_mit_for_schleifen(N):
    euler_pi, euler_time = None, None
    # BEGIN SOLUTION
    time_start = time.time()
    euler_pi = 0
    for k in range(1,N+1):
        euler_pi += 1 / k**2
    euler_pi = (6 * euler_pi)**(1/2)
    euler_time = time.time() - time_start
    # END SOLUTION
    return euler_pi, euler_time


def euler_ohne_for_schleifen(N):
    euler_pi, euler_time = None, None
    # BEGIN SOLUTION
    start_time = time.time()
    array_N = np.arange(1,N+1)
    summen_glieder = 1 / array_N**2
    euler_pi = (6 * np.sum(summen_glieder))**(1/2)
    euler_time = time.time() - start_time
    # END SOLUTION
    return euler_pi, euler_time


# (ii) Testen
N1 = int(1e1)
N2 = int(1e7)
# BEGIN SOLUTION
euler_for_N1 = euler_mit_for_schleifen(N1)
euler_for_N2 = euler_mit_for_schleifen(N2)
print("Euler Ansatz mit for-Schleife für N = ",N1," : ")
print(f"Pi = {euler_for_N1[0]}, Fehler = {np.pi-euler_for_N1[0]}, Dauer = {euler_for_N1[1]} s \n")
print("Euler Ansatz mit for-Schleife für N = ",N2," : ")
print(f"Pi = {euler_for_N2[0]}, Fehler = {np.pi-euler_for_N2[0]}, Dauer = {euler_for_N2[1]} s \n")

euler_np_N1 = euler_ohne_for_schleifen(N1)
euler_np_N2 = euler_ohne_for_schleifen(N2)
print("Euler Ansatz ohne for-Schleife für N = ",N1," : ")
print(f"Pi = {euler_np_N1[0]}, Fehler = {np.pi-euler_np_N1[0]}, Dauer = {euler_np_N1[1]} s \n")
print("Euler Ansatz ohne for-Schleife für N = ",N2," : ")
print(f"Pi = {euler_np_N2[0]}, Fehler = {np.pi-euler_np_N2[0]}, Dauer = {euler_np_N2[1]} s \n")


leibniz_for_N1 = leibniz_mit_for_schleifen(N1)
leibniz_for_N2 = leibniz_mit_for_schleifen(N2)
print("Leibniz Ansatz mit for-Schleife für N = ",N1," : ")
print(f"Pi = {leibniz_for_N1[0]}, Fehler = {np.pi-leibniz_for_N1[0]}, Dauer = {leibniz_for_N1[1]} s \n")
print("Leibniz Ansatz mit for-Schleife für N = ",N2," : ")
print(f"Pi = {leibniz_for_N2[0]}, Fehler = {np.pi-leibniz_for_N2[0]}, Dauer = {leibniz_for_N2[1]} s \n")

leibniz_np_N1 = leibniz_ohne_for_schleifen(N1)
leibniz_np_N2 = leibniz_ohne_for_schleifen(N2)
print("Leibniz ohne for-Schleife für N = ",N1," : ")
print("Leibniz Ansatz mit for-Schleife für N = ",N1," : ")
print(f"Pi = {leibniz_np_N1[0]}, Fehler = {np.pi-leibniz_np_N1[0]}, Dauer = {leibniz_np_N1[1]} s \n")
print("Leibniz Ansatz ohne for-Schleife für N = ",N2," : ")
print(f"Pi = {leibniz_np_N2[0]}, Fehler = {np.pi-leibniz_np_N2[0]}, Dauer = {leibniz_np_N2[1]} s \n")

print('Für kleine N ist die for-Schleife schneller, die Ergebnisse sind aber noch ungenau. \n'
      'Für große N ist die Variante ohne for-Schleife deutlich schneller und die Ergebnisse sind genauer.\n')
# END SOLUTION

# (iii) Plots
# BEGIN SOLUTION
...
# END SOLUTION
