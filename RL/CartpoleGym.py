import numpy as np
from numpy import sin as s, cos as c
import matplotlib.pyplot as plt


class CartPoleEnv:
    def __init__(self):
        self.mc, self.mp, self.l, self.g = 1, 0.1, 0.5, 9.81  # mass_cart, mass_pole, length, gravity
        self.dt = 0.02  # seconds between state updates

        self.max_force = 10  # maximum force applied to cart
        self.setpoint = np.array([0.0, 0.0, np.pi, 0.0], dtype=np.float32)  # target state

        self.state = np.zeros(4, dtype=np.float32)  # Initial state: [x_cart, dx_cart, theta, dtheta]

    def reset(self):
        # Reset state to a uniformly sampled value within bounds
        lim = np.array([1, 1, np.pi, 1])
        self.state = np.random.uniform(low=-lim, high=lim)
        return self.state

    def step(self, action):
        clipped_action = np.clip(action, -self.max_force, self.max_force)  # Clip action to control input bounds

        self.state += self.dynamics(clipped_action) * self.dt

        # Ensure theta remains between plus and minus pi by wrapping around
        self.state[2] = ((self.state[2] + np.pi) % (2 * np.pi)) - np.pi

        reward, done = self.is_done_get_reward()

        self.render()

        return self.state, reward, done

    def dynamics(self, force):
        mc, mp, l, g = self.mc, self.mp, self.l, self.g
        x, x_dot, theta, theta_dot = self.state
        
        x_ddot = (force + mp * s(theta) * (l * theta_dot ** 2 + g * c(theta))) / (mc + mp * s(theta) ** 2)
        theta_ddot = (-force * c(theta) - mp * l * theta_dot ** 2 * s(theta) * c(theta) - (mc + mp) * g * s(theta)) / (l * (mc + mp * s(theta) ** 2))
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


    def is_done_get_reward(self):
        # Calculate the difference between setpoint and absolute value of the state
        dist = np.linalg.norm(self.setpoint - np.abs(self.state))

        done = dist < 0.1  # Check if norm of difference is greater than 0.1
        reward = 1.0 if not done else 0.0  # Reward for staying upright
        reward -= dist / 100  # Additional penalty proportional to distance from setpoint

        return reward, done

    def render(self):
        x, theta = self.state[0], self.state[2]
        l = self.l

        plt.cla()

        plt.plot([-2, 2], [0, 0], 'k')  # Ground
        plt.plot([x, x + l * s(theta)], [0, -l * c(theta)], 'o-')  # Pole
        plt.scatter(x, 0, marker='s')  # Cart position

        plt.axis('equal')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.pause(0.001)


if __name__ == "__main__":
    env = CartPoleEnv()
    obs = env.reset()
    done = False

    while not done:
        action = 0 # Zero action
        obs, reward, done = env.step(action)

    # plt.show()
