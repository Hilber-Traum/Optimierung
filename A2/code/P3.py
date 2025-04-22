## Hier werden sämtliche in dieser Datei benötigten Pakete und Module geladen
import numpy as np


## TODO: Aufgabenteil 3a. Vektoren als eindim. Arrays erstellen und Rechnungen ausführen
print('### 3a ###')

a = np.array([1,2,3,4])
b = np.array([2,3])
print(f'a = \n{a} \nb = \n{b}')  # \n = neue Zeile in Terminal

# BEGIN SOLUTION
print("shape a: ",np.shape(a))
print("shape b: ",np.shape(b))
print("a transponiert: ", a.T)
print("b transponiert =", b.T)
print("shape a transponiert: ",str(np.shape(a.T)))
print("shape b transponiert: ",str(np.shape(b.T)))
print('Die transponierten Arrays sind gleich den nicht transponierten Arrays.')

print('a+a = ',a+a)
print('a+a.T = ',a+a.T)
print('Ergebnis der letzten beiden Ausgaben erwartbarerweise identisch.')
#print('a+b = ',a+b)
#print('a+b.T = ',a+b.T)
print('a+b und a+b.T ist aufgrund der unterschiedlichen Dimensionen nicht möglich.')

print('a*a =',a*a)
print('a*a.T =',a*a.T)
print('a.T*a =',a.T*a)
#print('a*b =',a*b)
#print('a*b.T =',a*b.T)
print('a*b und a*b.T sind nicht möglich da die Dimensionen unterschiedlich sind.')

print('a@a =',a@a)
print('a@a.T =',a@a.T)
print('Berechnet das innere Produkt der Vektoren.')
#print('a@b =',a@b)
#print('a@b.T =',a@b.T)
print('a@b und a@b.T sind nicht möglich da die Dimensionen unterschiedlich sind.')


# END SOLUTION


## TODO: Aufgabenteil 3b. Vektoren als zweidim. Arrays erstellen und Rechnungen ausführen
print('\n### 3b ###')

a2 = a.reshape(4,1)
b2 = b.reshape(2,1)
print(f'a2 = \n{a2} \nb2 = \n{b2}')

# BEGIN SOLUTION
print("shape a2: ",np.shape(a2))
print("shape b2: ",np.shape(b2))
print("a2 transponiert: ", a2.T)
print("b2 transponiert =", b2.T)
print("shape a2 transponiert: ",str(np.shape(a2.T)))
print("shape b2 transponiert: ",str(np.shape(b2.T)))
print('a2 und b2 werden als Spalte, a2.T und b2.T jedoch als Zeile ausgegeben.')

print('a2+a2 = ',a2+a2)
print('a2+a2.T = ',a2+a2.T)
print('Die letzte Ausgabe erzeugt eine Matrix, wobei die Einträge durch Komponentenweise Addition entstehen.')
#print('a2+b2 = ',a2+b2)
print('a2+b2.T = ',a2+b2.T)
print('a2+b2 ist aufgrund der unterschiedlichen Dimensionen nicht möglich. a2+b2.T analog zu a+a.T')

print('a2*a2 =',a2*a2)
print('a2*a2.T =',a2*a2.T)
print('a2.T*a2 =',a2.T*a2)
#print('a2*b2 =',a2*b2)
print('a2 und b2 haben keine verträglichen Dimensionen')
print('a2*b2.T =',a2*b2.T)
print('Der * Operator funktioniert bei Vektoren gleicher shape wie zuvor bei den anderen wird jetzt jedoch eine Matrix erzeugt.')

#print('a2@a2 =',a2@a2)
print('a2@a2.T =',a2@a2.T)
#print('a2@b2 =',a2@b2)
print('a2@b2.T =',a2@b2.T)
print('a2@a2 und a2@b2 sind nicht möglich da die Dimensionen nicht kompatibel sind.')
# END SOLUTION


## TODO: Aufgabenteil 3c. Matrizen erstellen
print('\n### 3c ###')
# BEGIN SOLUTION
reihe1 = np.zeros((1,4),dtype=int)
zeile1 = np.ones((3,1),dtype=int)
restmatrixA=np.array([[4,8,12],[1,7,6],[3,4,9]],dtype=int) #erstellen (3,3)-Teilmatrix
A = np.concatenate((zeile1,restmatrixA),axis=1) #erstellen (3,4)-Teilmatrix
A = np.concatenate((reihe1,A),axis=0) #stellen A fertig
print(A)
# END SOLUTION


## TODO: Aufgabenteil 3d. Matrix-Vektor-Produkte berechnen
print('\n### 3d ###')
# BEGIN SOLUTION
print('A@a = ',A@a)
print('A@a2 = ',A@a2)
print('a.T@A@a = ',a.T@A@a)
print('a2.T@A@a2 = ',a2.T@A@a2)
# END SOLUTION


## TODO: Aufgabenteil 3e. Spur, Eigenwerte, Determinante
print('\n### 3d ###')
# BEGIN SOLUTION
print('Spur(A) = ',np.trace(A))
print('Summe der Diagonale von A : ',np.sum(np.diag(A)))
print('Erwartbarerweise ist Spur(A) = Summe der Diagonale von A')
print('Eigenwerte von A : ',np.linalg.eigvals(A))
print('Determinante von A : ',np.linalg.det(A))
print('Summe der Eigenwerte von A: ',np.sum(np.linalg.eigvals(A)))
print('Produkt der EIgenwerte von A: ',np.prod(np.linalg.eigvals(A)))
print('Summe der Eigenwerte stimmt mit der Spur und das Produkt der Eigenwerte mit der Determinante überein.')
# END SOLUTION
