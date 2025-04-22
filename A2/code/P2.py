
## TODO: Aufgabenteil 2a. Liste mit geraden Zahlen zwischen 1 und 10
print('### 2a ###')
# BEGIN SOLUTION
mylist=[2,4,6,8,10]
# END SOLUTION


## TODO: Aufgabenteil 2b. Werte ausgeben
print('\n### 2b ###')
# BEGIN SOLUTION
print(mylist[0], mylist[2], mylist[-1], mylist[1:3], mylist[1:], mylist[:4],mylist[:-2])
print("Die Indizierung  beginnt bei 0. Bei negativer Indizierung wird die Liste von hinten durchgezählt. 'x:y' gibt die Elemente von x bis exklusive y aus. Lässt man x oder y weg beginnt(endet) die Liste vom ersten Element(am letzten Element).")
# END SOLUTION


## TODO: Aufgabenteil 2c. Letzten Eintrag ändern
print('\n### 2c ###')
# BEGIN SOLUTION
mylist[-1]=100
print(mylist)
# END SOLUTION


## TODO: Aufgabenteil 2d. Numpy Array mit allen geraden Zahlen von 1 bis 10 auf 2 Arten erstellen
print('\n### 2d ###')
# Hinweis: Importieren Sie das Paket numpy ausnahmsweise hier
import numpy as np
# BEGIN SOLUTION
mylist[-1]=10 #Modifizieren das letzte Element erneut. Nach Aufgabe ist myList wieder eine Liste mit Hoechstelement 10.
array1 = np.array(mylist)
array2 = np.arange(2,11,2)
# END SOLUTION
print(array1,array2)

## TODO: Aufgabenteil 2e. Summiere Einträge
print('\n### 2e ###')
# BEGIN SOLUTION
minimum = np.min(array2)
maximum = np.max(array2)
summe = np.sum(array2)

print(minimum, maximum, summe)
# END SOLUTION
