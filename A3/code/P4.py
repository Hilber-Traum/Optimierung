## Hier werden sämtliche in dieser Datei benötigten Pakete und Module geladen
import numpy as np
import matplotlib.pyplot as plt


## TODO: Aufgabenteil 4a. Dimensionsumformungen eines Vektors
print('### 4a ###')
# BEGIN SOLUTION
v = np.array(range(12))
v=v.reshape(3, 4)
print('v(3,4) =\n',v,'\n')
v=v.reshape(4, 3)
print('v(4,3) =\n',v,'\n')
v=np.ndarray.flatten(v)
print('v(12,) =\n',v,'\n')
# END SOLUTION


## TODO: Aufgabenteil 4b. Logische Indizierung bei Vektoren
print('\n### 4b ###')
# BEGIN SOLUTION
v1 = np.array([],dtype=int) #Erstelle Arrays zum Speichern
v2 = np.array([],dtype=int)
for i in range(len(v)):
    if v[i]>3:                     #Falls das i-te Element in v >3 wird dies in v1 gespeichert.
        v1=np.append(v1,v[i])
print('v_i aus v mit v_i>3 : ',v1,'\n')
for i in range(len(v)):
    if v[i]<7:                     # Analog zu vorher mit 'i-tes Element <7'
        v2=np.append(v2,v[i])
print('v_i aus v mit v_i<7 : ',v2,'\n')
# END SOLUTION


## TODO: Aufgabenteil 4c. Einheitsmatrix erzeugen
print('\n### 4c ###')
# BEGIN SOLUTION
C1 = np.eye(100)
C2 = np.diag(np.ones(100))
print('C1=C2 : ', np.array_equal(C1, C2))
# END SOLUTION


## TODO: Aufgabenteil 4d. Hauptminor der Einheitsmatrix visualisieren
# BEGIN SOLUTION
C3 = C1[0:9,0:9]
plt.spy(C3)
plt.show()
# END SOLUTION


## TODO: Aufgabenteil 4e. exp und log von ndarrays
print('\n### 4e ###')
A = np.array([[1, 0, 3, 4], [3, 1, 1, 0], [0, -1, 2, 3], [1, 0, 0, -1]])
B = np.array([[8, 10, -13, 6], [5, 5.5, -1, -4], [4, 6.5, -11, 7], [0, 0, 2, -2]])
# BEGIN SOLUTION
print('exp(A) =\n',np.exp(A),'\n')
print('log(A) =\n',np.log(A),'\n')
print('Die Funktionen wenden Komponentenweise die Exponential-Funktion bzw. den Logarithmus auf die Matrix an.\n'
      'Dabei tritt das Problem auf, dass log(0) bzw allgemien log(x) für x<=0 nicht definiert ist.\n'
      'Hier wird "-inf" bzw. "nan" (not a number) ausgegeben')
# END SOLUTION


## TODO: Aufgabenteil 4f. Zeile und Spalte ausgeben
print('\n### 4f ###')
# BEGIN SOLUTION
print('Erste Spalte von B: \n',B[:,0])
print('Erste Zeile von B: \n',B[0,:])
# END SOLUTION


## TODO: Aufgabenteil 4g. LGS lösen
print('\n### 4g ###')
# BEGIN SOLUTION
print('Da det(A) =',np.linalg.det(A),'ungleich 0 ist, ist A invertierbar.\n')

X = np.zeros((4,4))
max_residuen = []
for i in range(4):
    b_i = B[:,i]
    x_i = np.linalg.inv(A)@b_i
    X[:,i] = x_i
    residuen = A@x_i-b_i
    max_residuen.append(float(np.max(np.abs(residuen))))
print('i) \nX = \n',X,'\nDas Ergebnis der Maxinumsnorm der Resdiuen der Spalten : ',max_residuen,'\n')

X2 = np.linalg.solve(A,B)
#X3 = np.linalg.inv(A)@B #berechnet Numerisches Ergebnis wie in Aufgabenteil i)
print('ii) \nX = \n',X2)
Residuum = np.linalg.norm(A@X2-B,np.inf)
print('Das Betragsmäßig größte Element der Residualmatrix ist: ',Residuum,'\n')
# END SOLUTION
