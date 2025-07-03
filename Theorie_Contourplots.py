import matplotlib.pyplot as plt
import numpy as np

#Matplotlibpatches

def f(x1,x2):
    return(x1 - 3/2)**2 + (x2 -1/2)**4
    

h1 = lambda x1,x2: x1+x2-1
h2 = lambda x1,x2: x1-x2-1
h3 = lambda x1,x2: -x1+x2-1
h4 = lambda x1,x2: -x1-x2-1

    
x1 = np.linspace(-2,2,500)
x2 = np.linspace(-2,2,500)

# Scatter plot with multiple customizations
X1, X2 = np.meshgrid(x1, x2)

f_v = np.vectorize(f)

Z = f_v(X1,X2)

fig = plt.figure()

levels = np.array([-1/2,-1/4,-1/8,1/8,1/4,1/2,1, 3/2, 5/2, 4])

zulassigerBereich = ((h1(X1,X2)<=0) & (h2(X1,X2)<=0) & (h3(X1,X2)<=0) & (h4(X1,X2)<=0))

plt.contour(X1, X2, Z, levels=20)
plt.colorbar()
plt.contourf(X1, X2, zulassigerBereich, alpha=0.4)
plt.show()




#Blatt 12
import matplotlib.pyplot as plt
import numpy as np

c = 1.05

def f(x1,x2):
    return -(x1 + 1)**2 - (x2+1)**2
    


    
x1 = np.linspace(-2,2,500)
x2 = np.linspace(-2,2,500)

X1, X2 = np.meshgrid(x1, x2)

f_v = np.vectorize(f)

Z = f_v(X1,X2)

fig = plt.figure()

levels = np.array([-1/2,-1/4,-1/8,1/8,1/4,1/2,1, 3/2, 5/2, 4])

plt.contour(X1, X2, Z, levels=20)
plt.colorbar()

c = 1+0.000001
stepsize = 0.05
#Iteriere über verschiedene 1<c<sqrt(2)
while(c < 2**(1/2)):
    h1 = lambda x1,x2: x1**2+x2**2-2
    h2 = lambda x1,x2: x1 - c
    zulassigerBereich = ((h1(X1,X2)<=0) & (h2(X1,X2)<=0))
    
    #Plotte für jeden Schritt den erlaubten Bereich. Damit sind die Bereiche umso blasser, desto größer c werden muss, damit sie erlaubt sind.
    plt.contourf(X1, X2, zulassigerBereich, alpha=0.1)
    c += stepsize
    
plt.show()