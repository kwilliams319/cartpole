import casadi as ca
import numpy as np
from numpy import sin as s, cos as c
import matplotlib.pyplot as plt

class Cartpole:
    def __init__(self):
        self.params = 1, .1, .5, 9.81  # mc, mp, l, g

        # Number of time steps
        self.N = 1000

        # Time horizon
        self.T = 10.0

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
        return ca.vertcat(x_d, x_dd, th_d, th_dd)
    
    def optimize(self):

        # Define the initial and final states
        x_init = np.array([0, 0, 0, 0])  # Initial state: [x_cart, theta, dx_cart, dtheta]
        x_final = np.array([0, 0, np.pi, 0])  # Final state: [x_cart, theta, dx_cart, dtheta]


        # Define the decision variables
        opti = ca.Opti()
        X = opti.variable(4, self.N + 1)  # State trajectory
        U = opti.variable(1, self.N)  # Control trajectory

        # Formulate the optimal control problem
        opti.minimize(ca.sumsqr(U))# + ca.sumsqr(X[:, 0]))  # Minimize the sum of squared control inputs

        # Initial and final state constraints
        opti.subject_to(X[:, 0] == x_init)
        opti.subject_to(X[:, -1] == x_final)

        opti.subject_to(-1. <= U)
        opti.subject_to(U <= 1.)

        # Dynamics constraints
        for k in range(self.N):
            X_next = X[:, k] + self.cartpole_dynamics(X[:, k], U[:, k]) * self.T / self.N
            opti.subject_to(X[:, k + 1] == X_next)

        # Solve the optimal control problem
        opti.solver('ipopt')
        sol = opti.solve()

        # Extract the optimal solution
        x_opt = sol.value(X)
        u_opt = sol.value(U)

        return x_opt, u_opt

    def animate(self, x_opt):
        # Animate the optimal trajectory
        for i, q_opt in enumerate(x_opt.T):
            if i % 3 == 0:
                self.render(q_opt)  # Plot the current state
        plt.show()

    def plot(self, x_opt):
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
        plt.plot(t[:-1], u_opt, label='Control Input')
        plt.xlabel('Time')
        plt.ylabel('Control Input')
        plt.legend()

        plt.tight_layout()
        plt.show()


cp = Cartpole()
x_opt, u_opt = cp.optimize()
cp.animate(x_opt)
cp.plot(x_opt, u_opt)