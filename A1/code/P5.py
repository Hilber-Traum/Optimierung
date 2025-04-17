import numpy as np
import matplotlib.pyplot as plt


## TODO: Aufgabenteil 5a. Sinus- und Kosinus-Plots erstellen
# Plot 1
x = np.linspace(0, 2*np.pi,100)  # Gitterpunkte
plt.plot(x, np.sin(x), 'g--', label='sin(x)')  # Sinus

# Ergänzen Sie die weiteren Bestandteile des Plots
# BEGIN SOLUTION
...
# END SOLUTION
plt.show()

# Plot 2
fig, ax = plt.subplots(nrows=2, ncols=2)
fig.tight_layout(pad=1.5)

ax[0, 0].plot(x, np.sin(x))
ax[0, 0].set_title('sin(x)')

# Ergänzen Sie die weiteren Bestandteile des Plots
# BEGIN SOLUTION
...
# END SOLUTION
plt.show()


## TODO: Aufgabenteil 5b. Einheitskreis plotten
# BEGIN SOLUTION
...
# END SOLUTION
