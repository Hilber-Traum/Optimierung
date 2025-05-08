import matplotlib.pyplot as plt


def plot_iteration_process(log, title='Iterationsverlauf'):
    # Stellt Iterierte und Funktionswerte dar, die über das Dictionary log übergeben wurden
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(log['x_list'], 'o')
    ax[0].set_title(r'Iterierte $x_k$')
    ax[1].plot(log['val_list'], 'o')
    ax[1].set_title(r'Funktionswerte $f(x_k)$')
    fig.suptitle(title)

    return fig
