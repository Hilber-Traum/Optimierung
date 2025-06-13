import numpy as np
from skimage import io
from scipy.sparse import spdiags, vstack, kron
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from A4.code.utils import plot_iteration_process
from A9_2 import quasi_newton

# weitere Module importieren
# BEGIN SOLUTION
...
# END SOLUTION

import os
class RigidRegistration:
    def __init__(self):
        self.ref_image = io.imread(os.path.join(os.path.split(__file__)[0], 'dataR.png')).astype(np.float64)
        self.tmp_image = io.imread(os.path.join(os.path.split(__file__)[0], 'dataT.png')).astype(np.float64)

        self.m = self.tmp_image.shape
        self.omega = np.array([0, self.m[0], 0, self.m[1]])
        self.h = (self.omega[1::2] - self.omega[0::2]) / np.array([self.m[1], self.m[0]])

        self.ref_grid = self.grid()
        self.tmp_grid = self.grid()

    def grid(self):
        # Get cell centered grid
        grid_xi = lambda i: np.array([self.omega[2*i] + (k+1/2)*self.h[i] for k in np.arange(self.m[i])])
        xi = list(reversed([grid_xi(i) for i in range(2)]))
        xc = list(np.meshgrid(*xi, indexing='ij'))
        xc = np.vstack([xc[i].flatten() for i in range(2)])
        return xc

    def deform_grid(self, grid, w):
        c = ((self.omega[1::2] - self.omega[0::2]) / 2).reshape(2, 1)
        theta, t = w[0], w[1:]
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        dR = np.array([[-np.sin(theta), -np.cos(theta)], [np.cos(theta), -np.sin(theta)]])

        phi = R @ (grid - c) + c + t[:, np.newaxis]
        dphi = dR @ (self.tmp_grid - c)[np.newaxis, :, :]
        dphi = np.concatenate((dphi, np.tile(np.eye(2)[:, :, np.newaxis], (1, 1, dphi.shape[2]))), axis=0)
        return phi, dphi

    def interpolate(self, image, grid):
        def Bspline3(x):
            condition = lambda i: np.logical_and(i <= x, x < i+1)

            f = np.where(condition(0), (1/6) * (x**3), 0)
            f = np.where(condition(1), (1/6) * (-3 * x**3 + 12 * x**2 - 12 * x + 4), f)
            f = np.where(condition(2), (1/6) * (3 * x**3 - 24 * x**2 + 60 * x - 44), f)
            f = np.where(condition(3), (1/6) * (4 - x)**3, f)

            g = np.where(condition(0), (1/6) * (3 * x**2), 0)
            g = np.where(condition(1), (1/6) * (-9 * x**2 + 24 * x - 12), g)
            g = np.where(condition(2), (1/6) * (9 * x**2 - 48 * x + 60), g)
            g = np.where(condition(3), (1/6) * -3 * (4 - x)**2, g)
            return f, g

        def paddedMatrix(dim):
            ones = np.ones(dim + 3)
            A = spdiags(data=[ones, 4*ones, ones], diags=[0, 1, 2], m=dim+1, n=dim+3)

            tmp_0 = spdiags(data=[6*ones, -12*ones, 6*ones], diags=[0, 1, 2], m=1, n=dim+3)
            tmp_m = spdiags(data=[6*ones, -12*ones, 6*ones], diags=[dim+0, dim+1, dim+2], m=1, n=dim+3)

            A_padded = vstack((tmp_0, A, tmp_m)) / 6
            return A_padded

        def naturalSpline3():
            A_padded = kron(paddedMatrix(self.m[0] - 1), paddedMatrix(self.m[1] - 1))
            data_padded = np.pad(image, (1, 1), 'constant', constant_values=(0, 0))
            C = spsolve(A_padded.tocsc(), data_padded.flatten()).reshape(self.m[0]+2, self.m[1]+2)
            return C

        # Redistribute Grid: Map grid points from [h/3, omega-h/3] to [h/3, m[i]-h/3]
        points_x, points_y = grid
        points_x = (points_x - self.omega[0]) / self.h[0]
        points_y = (points_y - self.omega[2]) / self.h[1]

        # Interpolate
        C = naturalSpline3()

        j = np.arange(-3, C.shape[1]-3) + .5
        k = np.arange(-3, C.shape[0]-3) + .5

        shift_x = points_x[:, :, np.newaxis, np.newaxis] - j[np.newaxis, np.newaxis, :, np.newaxis]
        shift_y = points_y[:, :, np.newaxis, np.newaxis] - k[np.newaxis, np.newaxis, np.newaxis, :]

        splines_x = Bspline3(shift_x)
        splines_y = Bspline3(shift_y)

        f = splines_x[0] * splines_y[0]
        s = np.einsum('ijkl, kl->ij', f, C)

        g = np.stack((splines_x[1] * splines_y[0],
                      splines_x[0] * splines_y[1]), axis=0)
        ds = np.einsum('dijkl, kl->dij', g, C)
        return s, ds

    def SSD(self, dataR, dataT):
        hd = np.prod(self.h)

        r = dataR - dataT
        ssd = .5 * hd * np.linalg.norm(r, ord='fro')**2
        dssd = r * hd
        return ssd, dssd

    def plot(self, dataR, dataT, phi, w, state=''):
        fig = plt.figure()
        fig.set_size_inches(12, 8)
        fig.tight_layout(pad=4.0)

        fig.suptitle(f'{state}\nRotation um {np.round(w, 2)[0]} Rad '
                     f'und Translation um {tuple(np.round(w, 2)[1:])}', fontsize=16)

        ax1 = fig.add_subplot(231)
        im1 = ax1.imshow(self.ref_image, vmin=0, vmax=255, cmap='bone')
        ax1.axis('image')
        ax1.set_title('R')

        ax2 = fig.add_subplot(232)
        ax2.imshow(self.tmp_image, vmin=0, vmax=255, cmap='bone')
        ax2.set_title(r'T und $\varphi$')

        step = self.m[0] // 10
        step_r = int((self.m[0] % 10) // 4)
        grid_plot = [x.reshape(self.m) for x in phi[::-1]]
        grid_plot = [x[step_r:-1:step, step_r:-1:step].flatten() for x in grid_plot]
        ax2.scatter(*grid_plot, c='w', marker='.')

        ax2.set_xlim([0, self.m[0]-1])
        ax2.set_ylim([0, self.m[1]-1][::-1])

        ax3 = fig.add_subplot(233)
        ax3.imshow(self.ref_image - self.tmp_image, vmin=-255, vmax=255, cmap='bwr')
        ax3.set_title(rf'$diff(R, T)$, SSD = {round(self.SSD(self.ref_image, self.tmp_image)[0], 2)}')

        ax5 = fig.add_subplot(235)
        ax5.imshow(dataT, vmin=0, vmax=255, cmap='bone')
        ax5.set_title(rf'$T  \circ \varphi$: Transformiertes T')

        ax6 = fig.add_subplot(236)
        diff = ax6.imshow(dataR - dataT, vmin=-255, vmax=255, cmap='bwr')
        ax6.set_title(rf'$diff(R, T \circ \varphi)$, SSD = {round(self.SSD(dataR, dataT)[0], 2)}')

        cax1 = plt.axes([.03, 0.11, .02, .77])
        fig.colorbar(im1, cax=cax1, aspect=1, pad=.1)
        cax2 = plt.axes([.92, 0.11, .02, .77])
        fig.colorbar(diff, cax=cax2, aspect=1, pad=.1)

        plt.show()

    def objective_fun_ir(self, w, plot, title):
        phi, dphi = self.deform_grid(self.tmp_grid, w)

        dataR, ddataR = self.interpolate(self.ref_image, [x.reshape(*self.m) for x in self.ref_grid])
        dataT, ddataT = self.interpolate(self.tmp_image, [x.reshape(*self.m) for x in phi])

        ssd, dssd = self.SSD(dataR, dataT)

        val = ssd
        grad = dphi * np.reshape(-ddataT * dssd[np.newaxis, :, :], (1, 2, np.prod(ddataT.shape[1:])))
        grad = np.sum(grad, axis=(1, 2))
        if plot:
            self.plot(dataR, dataT, phi, w, state=title)
        return val, grad


if __name__ == '__main__':
    rr = RigidRegistration()

    # TODO: Aufgabenteil 1. Verschiedene w0 testen und sich mit objective_fun_ir vertraut machen.
    #   Achtung: Drehwinkel in Radian und nicht in Grad.
    test_w0 = {
        0:    np.array([0, 0, 0]),  # weder Rotation noch Translation
        1:  np.array([0.4, 0, 0]),  # Rotation
        2: np.array([-0.4, -1, -1]),  # Rotation
        3:    np.array([0, 5, 0]),  # Translation x-Richtung
        4:    np.array([0, 0, 2]),  # Translation y-Richtung
        5: np.array([-0.51532779, 0.00690073,  2.00917066])
    }

    # Hier testen:
    #for i in range(6):
    #     _, _ = rr.objective_fun_ir(test_w0[i], plot=True, title='Beispielaufruf der Zielfunktion')

    # TODO: Aufgabenteil 1. Für Verwendung des Quasi-Newton-Verfahrens geeignete Funktion f erstellen.
    #   Hinweis: Der "*"-Operator zur Auflösung von Tupeln könnte hilfreich sein.
    # BEGIN SOLUTION
    f = lambda w: (*rr.objective_fun_ir(w, plot=False, title=None), None)
    # END SOLUTION

    # TODO: Aufagenteil 3. Parameter der Bildregistrierung für verschiedene Startwerte mit Quasi-Newton optimieren.
    #   Hinweis: - Importieren Sie Ihre Quasi-Newton-Implementierung aus A9_2.
    #            - Bildregistrierung ist rechenintensiv, Geduld.
    # BEGIN SOLUTION
    B0 = np.eye(3)
    rho = 1/2
    c1 = 1e-3
    tau = 1e-5
    eps = 1e-10
    kmax = 100
    w0 = np.array([0, 0, 0])
    w0_self = np.array([-0.4, 0, 2])

    rr.objective_fun_ir(w0_self, plot=True, title='Vor der Optimierung mit w0 = [-0.3, 1, 2]')
    log1 = quasi_newton(f = f, x0 = w0_self, B0 = B0, rho = rho, c1 = c1, eps = eps, tau = tau, kmax = kmax, plot = False, display = False)
    rr.objective_fun_ir(log1['xqn'], plot=True, title='Nach der Optimierung mit w0 = [-0.3, 1, 2]')

    rr.objective_fun_ir(w0, plot=True, title='Vor der Optimierung mit w0 = [0, 0, 0]')
    log2 = quasi_newton(f = f, x0 = w0, B0 = B0, rho = rho, c1 = c1, eps = eps, tau = tau, kmax = kmax, plot = False, display = False)
    rr.objective_fun_ir(log2['xqn'], plot=True, title='Nach der Optimierung mit w0 = [0, 0, 0]')

    # END SOLUTION

    # TODO: Aufagenteil 3. Diskussion.
    # BEGIN SOLUTION
    print("Verändert man das Tau und das Epsilon so stellt man fest, dass z.T bessere Ergebnisse mit größeren \n"
          "Tau und Epsiolon erreicht werden.\n"
          "Deutlich wichtiger für das Finden eines guten Minimierers ist der Startwert.")
    # END SOLUTION
