import numpy as np
import time
import matplotlib.pyplot as plt

## TODO: Aufgabenteil 6a. Fehlerhaften Code debuggen
print('### 6a ###')
# Fehlerhafter Code
def summe_und_produkt(a, b):
    c1 = a + b
    c1 = a.dot(b)
    c = [c1, c2]
    return c[0]


def summe_plus_produkt(a, b)
    c1, c2 = summe_und_produkt(a, b)
    result = c1 + c2
    return result


result = summe_plus_produkt(2, 3)
print(f'Result = {result}')


## TODO: Aufgabenteil 6b. Primzahlen bestimmen
print('\n### 6b ###')
def prime(n):
    # BEGIN SOLUTION
    ...
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
    ...
    # END SOLUTION
    return leibniz_pi, leibniz_time


def leibniz_ohne_for_schleifen(N):
    leibniz_pi, leibniz_time = None, None
    # BEGIN SOLUTION
    ...
    # END SOLUTION
    return leibniz_pi, leibniz_time


def euler_mit_for_schleifen(N):
    euler_pi, euler_time = None, None
    # BEGIN SOLUTION
    ...
    # END SOLUTION
    return euler_pi, euler_time


def euler_ohne_for_schleifen(N):
    euler_pi, euler_time = None, None
    # BEGIN SOLUTION
    ...
    # END SOLUTION
    return euler_pi, euler_time


# (ii) Testen
N1 = int(1e2)
N2 = int(1e7)
# BEGIN SOLUTION
...
# END SOLUTION

# (iii) Plots
# BEGIN SOLUTION
...
# END SOLUTION
