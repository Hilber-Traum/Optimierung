import numpy as np
import matplotlib.pyplot as plt


## TODO: Aufgabenteil 5a. Sinus- und Kosinus-Plots erstellen
# Plot 1
x = np.linspace(0, 2*np.pi,100)  # Gitterpunkte
plt.plot(x, np.sin(x), 'g--', label='sin(x)')  # Sinus

# Ergänzen Sie die weiteren Bestandteile des Plots
# BEGIN SOLUTION
plt.plot(x,np.cos(x),'b-.',label='cos(x)') #Cosinus
# Zusaetzliche Punkte:
punkte_x = [i for i in range(7)]
punkte_y = [(-1)**(i+1) * 0.5 for i in range(7)]
plt.plot(punkte_x,punkte_y,'r^',label='Punkte')
#Kosmetische Anpassungen:
plt.xlabel('x-Achse Werte im Intervall [0, 2'r'$\pi$'']')
plt.ylabel('x-Achse im Intervall [−1, 1]')
plt.title('Sinus und Cosinus')
plt.legend()
plt.grid(True)
# END SOLUTION
plt.show()

# Plot 2
fig, ax = plt.subplots(nrows=2, ncols=2)
fig.tight_layout(pad=1.5)

ax[0, 0].plot(x, np.sin(x))
ax[0, 0].set_title('sin(x)')

# Ergänzen Sie die weiteren Bestandteile des Plots
# BEGIN SOLUTION
ax[0, 1].plot(x, np.sin(x)*(-1))
ax[0, 1].set_title('-sin(x)')

ax[1, 0].plot(x, np.sin(-x))
ax[1, 0].set_title('sin(-x)')

ax[1, 1].plot(x, -np.sin(-x))
ax[1, 1].set_title('-sin(-x)')
# END SOLUTION
plt.show()


## TODO: Aufgabenteil 5b. Einheitskreis plotten
# BEGIN SOLUTION
plt.close('all')
t = np.linspace(0, 2*np.pi, 1000)   # Diskretierungspunkte
plt.plot(np.cos(t),np.sin(t))   # Plotten des Einheitskreises
plt.title('Einheitskreis: (cos(t), sin(t))')
#Anpassen der Achsen
plt.grid(True)
plt.xticks(np.arange(-1, 1.1, 0.5))
plt.yticks(np.arange(-1, 1.1, 0.5))
plt.axis('equal')
plt.show()
# END SOLUTION
