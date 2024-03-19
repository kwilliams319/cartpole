import casadi as ca
import numpy as np
from numpy import sin as s, cos as c
import matplotlib.pyplot as plt
import sympy as sp
from sympy.functions.elementary.trigonometric import sin, cos
import control

from matplotlib.animation import FuncAnimation


class Cartpole:
    def __init__(self):
        self.params = 1, .1, .5, 9.81  # mc, mp, l, g

        # Number of time steps
        self.N = 200

        # Time horizon
        self.T = 2.0

        A, B = self.linearize()

        # Define the Q and R matrices for the LQR cost function
        Q = np.eye(4)  # Weight matrix for state variables

        Q[1, 1] /= 100
        Q[2, 2] /= 100
        Q[3, 3] /= 100

        R = np.eye(1)*1e-5  # Weight matrix for control input

        # Compute the optimal K matrix using LQR
        self.K, _, _ = control.lqr(A, B, Q, R)


    def linearize(self):
        mc, mp, l, g = self.params
        # Define symbolic variables
        x, x_d, th, th_d, u = sp.symbols('x x_d th th_d u')

        # Define the dynamics
        x_dd = (1 / (mc + mp * sp.sin(th) ** 2)) * (mp * sp.sin(th) * (l * th_d ** 2 + g * sp.cos(th)) + u)
        th_dd = (1 / (l * (mc + mp * sp.sin(th) ** 2))) * (
                    -mp * l * (th_d ** 2) * sp.cos(th) * sp.sin(th) - (mc + mp) * g * sp.sin(th) - u * sp.cos(th))

        # Define the state vector
        state = sp.Matrix([x, x_d, th, th_d])

        # Define the vector field (right-hand side of the system)
        f = sp.Matrix([x_d, x_dd, th_d, th_dd])

        # Compute the Jacobian matrix
        A = f.jacobian(state)

        # Evaluate the Jacobian matrix at the specified point
        A_eval = A.subs({x: 0, x_d: 0, th: np.pi, th_d: 0, u: 0})

        # Convert the symbolic matrix to a numpy array
        A_eval = np.array(A_eval).astype(np.float64)

        # Compute the Jacobian matrix with respect to input u
        f_u = f.diff(u)

        # Evaluate the derivative at the specified point
        B_eval = f_u.subs({x: 0, x_d: 0, th: np.pi, th_d: 0, u: 0})

        # Convert the symbolic matrix to a numpy array
        B_eval = np.array(B_eval).astype(np.float64)

        return A_eval, B_eval

    def render(self, q):
        x, th = [q[j] for j in [0, 2]]  # Extract cart position and pole angle
        l = 1  # Length of the pole for animation purposes

        # Calculate the coordinates of the cart and pole
        x_cart = np.array([x, x + l * np.sin(th)])
        y_cart = np.array([0, -l * np.cos(th)])

        # Clear the current plot
        plt.cla()

        # Set axis limits
        plt.axis('equal')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])

        # Plot the ground and the cart-pole system
        plt.plot([-2, 2], [0, 0], 'k')  # Ground
        plt.plot(x_cart, y_cart, 'o-')  # Cart-pole system
        plt.scatter(x_cart[0], y_cart[0], marker='s')

        # Pause to create animation
        plt.pause(0.01)


    def cartpole_dynamics(self, x, u):
        mc, mp, l, g = self.params
        x, x_d, th, th_d = [x[j] for j in [0, 1, 2, 3]]
        u = u[0]

        x_dd = 1/(mc + mp*s(th)**2) * (mp*s(th) * (l*th_d**2 + g*c(th)) + u)# - x_d  # damping ?
        th_dd = 1/(l*(mc+mp*s(th)**2))*(-mp*l*(th_d**2)*c(th)*s(th) - (mc + mp)*g*s(th) - u*c(th))# - th_d   # damping ?
        return np.array([x_d, x_dd, th_d, th_dd])
    
    def simulate(self):

        # Define the initial and final states
        x_init = np.array([-1, 0, np.pi, 0])  # Initial state: [x_cart, theta, dx_cart, dtheta]
        x_set = np.array([[0, 0, np.pi, 0]]).T  # Final state: [x_cart, theta, dx_cart, dtheta]

        X = np.zeros((4, self.N + 1))  # State trajectory
        U = np.zeros((1, self.N))  # Control trajectory

        X[:, 0] = x_init

        # opti.subject_to(-1. <= U)
        # opti.subject_to(U <= 1.)

        # Dynamics constraints
        for k in range(self.N):
            U[:, [k]] = self.K @ (x_set - X[:, [k]])
            X[:, k + 1] = X[:, k] + self.cartpole_dynamics(X[:, k], U[:, k]) * self.T / self.N

        return X, U

    def animate(self, x_opt):
        # Animate the optimal trajectory
        for i, q_opt in enumerate(x_opt.T):
            if i % 3 == 0:
                self.render(q_opt)  # Plot the current state
                # print(i/self.N)
        plt.show()

    def plot(self, x_opt, u_opt):
        # Plot the results
        t = np.linspace(0, self.T, self.N + 1)
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, x_opt[0, :], label='Cart Position')
        plt.plot(t, x_opt[1, :], label='Cart Velocity')
        plt.plot(t, x_opt[2, :], label='Pole Angle')
        plt.plot(t, x_opt[3, :], label='Pole Angle Rate')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t[:-1], u_opt[0, :], label='Control Input')
        plt.xlabel('Time')
        plt.ylabel('Control Input')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def animate_to_gif(self, x_opt, output_path='cartpole_animation.gif', fps=30):
        fig, ax = plt.subplots()

        def animate(frame):
            # ax.clear()
            self.render(x_opt[:, frame*3])
            
        # Create animation
        ani = FuncAnimation(fig, animate, frames=self.N//3, interval=20)

        # Save animation as GIF
        ani.save(output_path, writer='imagemagick', fps=fps)


cp = Cartpole()
x_opt, u_opt = cp.simulate()
cp.animate(x_opt)
cp.plot(x_opt, u_opt)

# cp.animate_to_gif(x_opt, output_path='cartpole_animation.gif', fps=30)
