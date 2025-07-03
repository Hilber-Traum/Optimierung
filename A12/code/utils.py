import numpy as np
import matplotlib.pyplot as plt

def plot_surface(xlist=None):
    # Create a grid of x1 and x2 values
    x = np.linspace(-1.2, 1.2, 400)
    X1, X2 = np.meshgrid(x, x)

    # Define the objective function
    Z = (X1 - 3/2)**2 + (X2 - 1/2)**4

    # Define constraints
    c1 = X1 + X2 - 1 <= 0
    c2 = X1 - X2 - 1 <= 0
    c3 = -X1 + X2 - 1 <= 0
    c4 = -X1 - X2 - 1 <= 0

    # Initialise Plot
    fig, ax = plt.subplots()

    # Plot the objective function contours
    contours = ax.contour(X1, X2, Z, levels=20, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)

    # Plot constraint boundaries
    ax.plot(x, 1 - x, 'r--', label='x1 + x2 <= 1')
    ax.plot(x, x - 1, 'g--', label='x1 - x2 <= 1')
    ax.plot(x, x + 1, 'b--', label='-x1 + x2 <= 1')
    ax.plot(x, -x - 1, 'm--', label='-x1 - x2 <= 1')

    # Shade non-feasible regions
    ax.contourf(X1, X2, ~c1, levels=[0.5, 1], colors=['lightgray'], alpha=0.5)
    ax.contourf(X1, X2, ~c2, levels=[0.5, 1], colors=['lightgray'], alpha=0.5)
    ax.contourf(X1, X2, ~c3, levels=[0.5, 1], colors=['lightgray'], alpha=0.5)
    ax.contourf(X1, X2, ~c4, levels=[0.5, 1], colors=['lightgray'], alpha=0.5)

    # Plot iteration process
    if xlist is not None:
        ax.scatter(xlist[0][0], xlist[0][1], marker='x')
        for k in range(len(xlist) - 1):
            xk = xlist[k]
            xkplus1 = xlist[k + 1]
            ax.plot([xk[0], xkplus1[0]], [xk[1], xkplus1[1]])

    # Set aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # Labels and legend
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Objective Function with Feasible Region')
    ax.legend(loc='upper right')

    return fig
