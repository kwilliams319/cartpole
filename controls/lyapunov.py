import numpy as np
from numpy import sin as s, cos as c
import matplotlib.pyplot as plt
import control

import os, imageio

class Pendulum:
    def __init__(self):
        # self.x = np.zeros(2,)
        self.x = np.array([np.pi/12, 0])

        
        self.params = m, g, l = 1, 9.81, 1

        # Compute the optimal K matrix using LQR
        A = np.array([[0,    1],
                      [g/l, 0]])
        B = np.array([[0],
                      [1/(m*l**2)]])
        Q = np.eye(2)
        R = np.eye(1)
        self.K, _, _ = control.lqr(A, B, Q, R)

        # for render
        self.swingup = False
        self.u = 0

        # for gif
        self.image_folder = "images"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        self.frame_count = 0

    def render(self):
        plt.cla()
        plt.plot([-2, 2], [0, 0], 'k-')

        th, _ = self.x

        p = np.array([[0, 0],
                      [0, -1]])
        
        R = np.array([[c(th), -s(th)],
                      [s(th),  c(th)]])
        
        p = R @ p
        x, y = p[0], p[1]

        plt.plot(x, y, 'o-')
        plt.plot(x[1], y[1], 'o', markersize=20, c='tab:blue')

        control = "Lyapnuov" if self.swingup else "LQR"
        plt.title(f"{control} Torque: {self.u:.2f}")
        plt.axis('equal')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.pause(.01)

        plt.savefig(f"{self.image_folder}/frame_{self.frame_count:03d}.png")
        self.frame_count += 1

    # @staticmethod
    def dynamics(self, x, u=0):
        m, g, l = self.params
        #b = .1
        th, thd = x
        thdd = u / (m*l**2) - g * s(th) / l  # - b * thd/ (m*l**2)

        return np.array([thd, thdd])
    
    def control(self):
        m, g, l = self.params
        th, thd = self.x

        th = ((th + np.pi) % (2 * np.pi)) - np.pi

        # print(th)

        if np.abs(th) <= np.pi*.75:
            k = 1
            E = m*l**2*thd**2/2 - m*g*l*c(th)
            u = k * thd * (m*g*l - E)

            self.swingup = True
        else:
            if th >= 0:
                th_des = np.pi
            else:
                th_des = -np.pi

            e = np.array([[th_des - th],
                          [-thd]])

            u = (self.K @ e)[0, 0]
            # u += np.random.uniform(-1, 1)  # disturbances
            self.swingup = False
            
        return np.clip(u, -2, 2)

    def step(self):
        # self.x += self.dynamics() * .01  # euler integration

        self.u = u = self.control()

        h = .05  # rk4 integration
        x, dyn = self.x, self.dynamics

        k1 = dyn(x,          u)
        k2 = dyn(x + h*k1/2, u)
        k3 = dyn(x + h*k2/2, u)
        k4 = dyn(x + h*k3,   u)

        self.x += h/6 * (k1 + 2*k2 + 2*k3 + k4)

    def create_gif(self, gif_filename):
        images = []
        for filename in os.listdir(self.image_folder):
            if filename.endswith(".png"):
                images.append(imageio.imread(os.path.join(self.image_folder, filename)))
        imageio.mimsave(gif_filename, images)
        if os.path.exists(self.image_folder):
            import shutil
            shutil.rmtree(self.image_folder)


if __name__ == "__main__":
    p = Pendulum()

    for i in range(700):
        p.step()
        p.render()

        if not p.swingup and i % 50 == 0:
            p.x[1] += np.random.uniform(-1, 1)

    p.create_gif("Lyapunov.gif")