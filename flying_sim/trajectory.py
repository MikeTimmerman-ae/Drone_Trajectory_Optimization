from flying_sim.drone import Drone
import configs.config
import numpy as np
import os
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
import cvxpy as cvx
import yaml
import imageio
from PIL import Image
matplotlib.use('TkAgg')


class Trajectory:

    def __init__(self, config, drone, waypoints, trajectory_type, seed=15):
        self.min_dist = config.traj_config.min_dist
        self.max_dist = config.traj_config.max_dist
        self.v_des = config.traj_config.v_des
        self.traj_time = config.traj_config.tf
        self.drone = drone

        self.max_iter = 5
        self.eps = 1e-2  # SCP convergence tolerance
        self.P = 1e0 * np.eye(12)  # terminal state cost matrix
        self.Q = 1e0 * np.eye(3)   # state cost matrix
        self.Q_ = 5e2 * np.eye(3)   # state cost matrix
        self.R = 1e-1 * np.eye(4)  # control cost matrix
        self.xρ = 1.                # trust region parameter
        self.uρ = 0.5                # trust region parameter
        self.r = 1.5                 # radius of gates

        self.filenames_3d = []
        self.filenames_2d = []
        self.config = config
        self.set_seed(seed)
        self.xf = np.zeros((12, ))
        self.generate_waypoints(waypoints)
        if trajectory_type == "spline":
            self.spline_trajectory()
        elif trajectory_type == "optimize":
            self.optimized_trajectory()
        elif trajectory_type == "CPC":
            self.CPC_trajectory()

    def position_ref(self, t: float) -> np.ndarray:
        return np.array([
            self.traj_x(t),
            self.traj_y(t),
            self.traj_z(t)
        ])

    def velocity_ref(self, t: float) -> np.ndarray:
        return np.array([
            self.traj_dx(t),
            self.traj_dy(t),
            self.traj_dz(t)
        ])

    def acceleration_ref(self, t: float) -> np.ndarray:
        return np.array([
            self.traj_ddx(t),
            self.traj_ddy(t),
            self.traj_ddz(t)
        ])

    def set_seed(self, seed):
        np.random.seed(seed)

    def generate_waypoints(self, waypoints):
        if waypoints == "random":
            self.time = [0]
            self.waypoint_idx = [0]
            self.waypoints = np.array([self.config.traj_config.init_pos])
            # Generate random waypoints
            while self.time[-1] < self.traj_time:
                x = np.random.uniform(low=0.0, high=50.0, size=None)
                y = np.random.uniform(low=0.0, high=50.0, size=None)
                z = np.random.uniform(low=-30.0, high=-2.0, size=None)
                new_waypoint = np.array([x, y, z])

                if self.min_dist < np.linalg.norm(self.waypoints[-1] - new_waypoint) < self.max_dist:
                    self.time.append(self.time[-1] + np.linalg.norm(self.waypoints[-1, :] - new_waypoint) / self.v_des)
                    self.waypoint_idx.append(int(self.time[-1] / self.drone.dt))
                    self.waypoints = np.vstack((self.waypoints, new_waypoint))
        elif waypoints == "figure-8":
            self.time = [0]
            self.waypoints = np.array([[0, 0, 0], [2, 3, 0.5], [6, 4, 1], [10, 4, 1], [13, 3, 0.5],
                                       [15, 0, 0], [13, -3, -0.5], [10, -4, -1], [6, -4, -1], [2, -3, -0.5],
                                       [0, 0, 0], [-2, 3, 0.5], [-6, 4, 1], [-10, 4, 1], [-13, 3, 0.5],
                                       [-15, 0, 0], [-13, -3, -0.5], [-10, -4, -1], [-6, -4, -1], [-2, -3, -0.5],
                                       [0, 0, 0], [2, 3, 0.5], [6, 4, 1], [10, 4, 1]])
            for i, waypoint in enumerate(self.waypoints[1:]):
                dist = np.linalg.norm(waypoint - self.waypoints[i - 1])
                self.time.append(self.time[-1] + dist / self.v_des)
        elif waypoints == "hover":
            self.time = [0, self.traj_time]
            self.waypoints = np.array([self.config.traj_config.init_pos, self.config.traj_config.init_pos])
        elif waypoints == "track":
            filename = f"flying_sim/{waypoints}.yaml"
            with open(filename, 'r') as file:
                track = yaml.load(file, Loader=yaml.FullLoader)
            if 'gates' in track:
                self.waypoints = track['gates']
            else:
                print("No gates specified in " + filename)
            if 'initial' in track:
                initial = track['initial']
                x0 = np.zeros((12, ))
                if 'position' in initial:
                    x0[6:9] = initial['position']
                if 'attitude' in initial:
                    x0[0:3] = initial['attitude']
                if 'velocity' in initial:
                    x0[9:12] = initial['velocity']
                if 'omega' in initial:
                    x0[3:6] = initial['omega']
                self.drone.x0 = x0
                self.waypoints = np.vstack((x0[6:9], self.waypoints))
            if 'end' in track:
                end = track['end']
                if 'position' in end:
                    self.xf[6:9] = end['position']
                if 'attitude' in end:
                    self.xf[0:3] = end['attitude']
                if 'velocity' in end:
                    self.xf[9:12] = end['velocity']
                if 'omega' in end:
                    self.xf[3:6] = end['omega']
                self.waypoints = np.vstack((self.waypoints, self.xf[6:9]))
            self.time = [0]
            self.waypoint_idx = [0]
            for i in range(len(self.waypoints)-1):
                dist = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
                self.time.append(self.time[-1] + dist / self.v_des)
                self.waypoint_idx.append(int(self.time[-1] / self.drone.dt))

    def spline_trajectory(self):
        # Fit cubic spline waypoints
        self.traj_x = CubicSpline(self.time, self.waypoints[:, 0], bc_type="clamped")
        self.traj_y = CubicSpline(self.time, self.waypoints[:, 1], bc_type="clamped")
        self.traj_z = CubicSpline(self.time, self.waypoints[:, 2], bc_type="clamped")

        self.traj_dx = self.traj_x.derivative(nu=1)
        self.traj_dy = self.traj_y.derivative(nu=1)
        self.traj_dz = self.traj_z.derivative(nu=1)

        self.traj_ddx = self.traj_x.derivative(nu=2)
        self.traj_ddy = self.traj_y.derivative(nu=2)
        self.traj_ddz = self.traj_z.derivative(nu=2)

        time = np.arange(self.time[0], self.time[-1], self.drone.dt)
        trajectory = np.array([[self.traj_x(t), self.traj_y(t), self.traj_z(t)] for t in time])
        self.save_trajectory("spline", time, trajectory, self.time, self.waypoints)

    def optimized_trajectory(self, save_gif=False):
        # Initialize trajectory
        N = self.waypoint_idx[-1]

        s = np.loadtxt('flying_sim/initial_trajectory.csv')[:N+1]
        u = np.loadtxt('flying_sim/initial_inputs.csv')[1:N+1]
        time = np.linspace(self.time[0], self.time[-1], len(s))

        # Do SCP until convergence or maximum number of iterations is reached
        if save_gif:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(projection='3d')
            ax1.view_init(20, -45)
            ax1.set_xlabel("X-position [m]")
            ax1.set_ylabel("Y-position [m]")
            ax1.set_zlabel("Z-position [m]")
            ax1.scatter(self.waypoints[:, 0], self.waypoints[:, 1], -self.waypoints[:, 2])
            fig2, ax2 = plt.subplots(1, 3, figsize=(15, 8))
            [ax.grid() for ax in ax2]
            self.render_path(ax1, fig1, ax2, fig2, time, s, 0) if save_gif else None

        J = np.zeros(self.max_iter + 1)
        J[0] = np.inf
        for i in range(self.max_iter):
            s, u, J[i + 1] = self.scp_iteration(s, u)

            self.render_path(ax1, fig1, ax2, fig2, time, s, i+1) if save_gif else None

            dJ = np.abs(J[i + 1] - J[i])
            print(f"Completed SCP iteration {i} with change in objective of {dJ}")
            if dJ < self.eps:
                break

        if save_gif:
            paths = [Image.open(filename) for filename in self.filenames_3d]
            imageio.mimsave(f'SCP_3dpath_1.gif', paths, format='GIF', duration=1000)
            for filename in self.filenames_3d:
                os.remove(filename)
            self.filenames_3d = []

            paths = [Image.open(filename) for filename in self.filenames_2d]
            imageio.mimsave(f'SCP_2dpath_1.gif', paths, format='GIF', duration=1000)
            for filename in self.filenames_2d:
                os.remove(filename)
            self.filenames_2d = []

        # Plot objective over time
        plt.figure()
        plt.plot(J[1:])
        plt.grid()
        plt.xlabel("SCP Iterations")
        plt.ylabel(r"Objective J(x_0)")

        # Create reference trajectories
        lin_acc = np.zeros((len(time), 3))
        for i, (s_, u_) in enumerate(zip(s, u)):
            T = self.drone.rotationBodytoEarth(s_[0:3]) @ np.array([0, 0, u_[3]])
            G = np.array([0, 0, self.drone.m * self.drone.g])
            lin_acc[i] = (T + G) / self.drone.m
        self.time = 0.75 * np.array(self.time)
        time = np.linspace(self.time[0], self.time[-1], len(s))
        self.traj_x = interp1d(time, s[:, 6], kind='linear', bounds_error=False, fill_value=(self.waypoints[0, 0], self.waypoints[-1, 0]))
        self.traj_y = interp1d(time, s[:, 7], kind='linear', bounds_error=False, fill_value=(self.waypoints[0, 1], self.waypoints[-1, 1]))
        self.traj_z = interp1d(time, s[:, 8], kind='linear', bounds_error=False, fill_value=(self.waypoints[0, 2], self.waypoints[-1, 2]))
        self.traj_dx = interp1d(time, s[:, 9], kind='linear', bounds_error=False, fill_value=(s[0, 9], s[-1, 9]))
        self.traj_dy = interp1d(time, s[:, 10], kind='linear', bounds_error=False, fill_value=(s[0, 10], s[-1, 10]))
        self.traj_dz = interp1d(time, s[:, 11], kind='linear', bounds_error=False, fill_value=(s[0, 11], s[-1, 11]))
        self.traj_ddx = interp1d(time, lin_acc[:, 0], kind='linear', bounds_error=False, fill_value=(lin_acc[0, 0], lin_acc[-1, 0]))
        self.traj_ddy = interp1d(time, lin_acc[:, 1], kind='linear', bounds_error=False, fill_value=(lin_acc[0, 1], lin_acc[-1, 1]))
        self.traj_ddz = interp1d(time, lin_acc[:, 2], kind='linear', bounds_error=False, fill_value=(lin_acc[0, 2], lin_acc[-1, 2]))
        self.save_trajectory("SCP", time, s[:, 6:9], self.time, self.waypoints)
        return time, s, u, J

    def scp_iteration(self, x_prev, u_prev):
        """Solve a single SCP sub-problem for the obstacle avoidance problem."""
        n = x_prev.shape[-1]  # state dimension
        m = u_prev.shape[-1]  # control dimension
        N = u_prev.shape[0]  # number of steps

        x0 = x_prev[0, :]
        x_goal = self.xf

        A, B, c = self.drone.affinize(x_prev[:-1], u_prev)
        A, B, c = np.array(A), np.array(B), np.array(c)

        x_cvx = cvx.Variable((N + 1, n))
        u_cvx = cvx.Variable((N, m))

        objective = 0
        constraints = []
        # Build up the objective function
        for t in range(N):
            if t != 0 and t != N - 1:
                objective += cvx.atoms.quad_form((x_cvx[t + 1, 9:] - 2 * x_cvx[t, 9:] + x_cvx[t - 1, 9:]), self.Q_)  # state cost
                objective += cvx.atoms.quad_form((x_cvx[t + 2, 9:] - 3 * x_cvx[t + 1, 9:] + 3 * x_cvx[t, 9:] - x_cvx[t - 1, 9:]), self.Q_)  # state cost
            objective += cvx.atoms.quad_form((x_cvx[t+1, 6:9] - x_cvx[t, 6:9]), self.Q)     # state cost
            objective += cvx.atoms.quad_form(u_cvx[t, :], self.R)                           # input cost
        objective += cvx.atoms.quad_form((x_cvx[N] - x_goal), self.P)                       # terminal state cost

        # Build up the constrains
        constraints.append(x_cvx[0] == x0)  # initial constraint
        for t in range(N):
            # dynamics constraint
            constraints.append(A[t] @ (x_cvx[t] - x_prev[t]) + B[t] @ (u_cvx[t] - u_prev[t]) + c[t] == x_cvx[t + 1])
            # trust region
            constraints.append(cvx.norm(u_cvx[t] - u_prev[t], 'inf') <= self.uρ)
            constraints.append(cvx.norm(x_cvx[t] - x_prev[t], 'inf') <= self.xρ)
            # state constraints
            # constraints.append(x_cvx[t, 8] <= 0.)                       # Z-position
            constraints.append(x_cvx[t, :2] <= np.pi/2)                 # Attitude roll and pitch angles
            constraints.append(x_cvx[t, 3:6] <= 3/2 * np.pi)            # Angular velocity
        # waypoint constraint
        constraints.append(cvx.norm(x_cvx[self.waypoint_idx, 6:9] - self.waypoints, axis=1) <= self.r / 2)

        # Solve optimization problem
        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        prob.solve()
        if prob.status != "optimal":
            raise RuntimeError("SCP solve failed. Problem status: " + prob.status)
        x = x_cvx.value
        u = u_cvx.value
        J = prob.objective.value
        return x, u, J

    def CPC_trajectory(self):
        trajectory = pd.read_csv("flying_sim/out.csv")

        time = np.array(trajectory['t'])
        # Obtain waypoints
        waypoint_trackers = ['nu_0', 'nu_1', 'nu_2', 'nu_3', 'nu_4', 'nu_5', 'nu_6', 'nu_7']
        for i, tracker in enumerate(waypoint_trackers):
            self.time[i+1] = time[np.argmax(trajectory[tracker])]

        plt.show()
        # Position reference
        px = np.array(trajectory['p_x'])
        py = np.array(trajectory['p_y'])
        pz = -np.array(trajectory['p_z'])

        # Velocity reference
        vx = np.array(trajectory['v_x'])
        vy = np.array(trajectory['v_y'])
        vz = -np.array(trajectory['v_z'])

        # Acceleration reference
        ax = np.array(trajectory['a_lin_x'])
        ay = np.array(trajectory['a_lin_y'])
        az = -np.array(trajectory['a_lin_z'])

        # Create reference trajectories
        self.traj_x = interp1d(time, px, kind='linear', bounds_error=False, fill_value=(px[0], px[-1]))
        self.traj_y = interp1d(time, py, kind='linear', bounds_error=False, fill_value=(py[0], py[-1]))
        self.traj_z = interp1d(time, pz, kind='linear', bounds_error=False, fill_value=(pz[0], pz[-1]))
        self.traj_dx = interp1d(time, vx, kind='linear', bounds_error=False, fill_value=(vx[0], vx[-1]))
        self.traj_dy = interp1d(time, vy, kind='linear', bounds_error=False, fill_value=(vy[0], vy[-1]))
        self.traj_dz = interp1d(time, vz, kind='linear', bounds_error=False, fill_value=(vz[0], vz[-1]))
        self.traj_ddx = interp1d(time, ax, kind='linear', bounds_error=False, fill_value=(ax[0], ax[-1]))
        self.traj_ddy = interp1d(time, ay, kind='linear', bounds_error=False, fill_value=(ay[0], ay[-1]))
        self.traj_ddz = interp1d(time, az, kind='linear', bounds_error=False, fill_value=(az[0], az[-1]))

        trajectory = np.array([[self.traj_x(t), self.traj_y(t), self.traj_z(t)] for t in time])
        self.save_trajectory("CPC", time, trajectory, self.time, self.waypoints)

    def render_path(self, ax1, fig1, ax2, fig2, t, s, i):
        # 3d plot
        ax1.plot(s[:, 6], s[:, 7], -s[:, 8], label=i)
        filename = f'img_3d_{i}.png'
        fig1.savefig(filename)
        self.filenames_3d.append(filename)

        # 2d plot
        ax2[0].plot(t, s[:, 6])
        ax2[0].scatter(self.time, self.waypoints[:, 0])
        ax2[0].set_ylabel("X-position [m]")

        ax2[1].plot(t, s[:, 7])
        ax2[1].scatter(self.time, self.waypoints[:, 1])
        ax2[1].set_ylabel("Y-position [m]")

        ax2[2].plot(t, s[:, 8])
        ax2[2].scatter(self.time, self.waypoints[:, 2])
        ax2[2].set_ylabel("Z-position [m]")
        ax2[2].set_xlabel("Time [s]")

        for j in range(3):
            ax2[j].vlines(self.time, self.waypoints[:, j] - self.r, self.waypoints[:, j] + self.r, 'r')

        filename = f'img_2d_{i}.png'
        fig2.savefig(filename)
        self.filenames_2d.append(filename)

    def path_length(self):
        time = np.arange(0, self.time[-1], self.drone.dt)
        dist = 0
        for i in range(len(time)-1):
            dist += np.linalg.norm(self.position_ref(time[i+1]) - self.position_ref(time[i]))
        return dist

    def save_trajectory(self, name, time, trajectory, time_stamps, waypoints):
        waypoints = {'time_stamps': time_stamps,
                     'wp_x': waypoints[:, 0],
                     'wp_y': waypoints[:, 1],
                     'wp_z': waypoints[:, 2]}
        trajectory = {'time': time,
                      'px': trajectory[:, 0],
                      'py': trajectory[:, 1],
                      'pz': trajectory[:, 2]}
        wp_data = pd.DataFrame.from_dict(waypoints)
        traj_data = pd.DataFrame.from_dict(trajectory)
        wp_data.to_csv(f"wp_{name}")
        traj_data.to_csv(f"traj_{name}")

    def trajectory_xz(self):
        self.time = [0]
        self.waypoints = np.array([[0, 0, 0], [2, 0, 3], [6, 0, 4], [10, 0, 4], [13, 0, 3],
                                   [15, 0, 0], [13, 0, -3], [10, 0, -4], [6, 0, -4], [2, 0, -3],
                                   [0, 0, 0], [-2, 0, 3], [-6, 0, 4], [-10, 0, 4], [-13, 0, 3],
                                   [-15, 0, 0], [-13, 0, -3], [-10, 0, -4], [-6, 0, -4], [-2, 0, -3],
                                   [0, 0, 0], [2, 0, 3], [6, 0, 4], [10, 0, 4]])
        for i, waypoint in enumerate(self.waypoints[1:]):
            dist = np.linalg.norm(waypoint - self.waypoints[i-1])
            self.time.append(self.time[-1] + dist / self.v_des)

        # Fit cubic spline waypoints
        self.traj_x = CubicSpline(self.time, self.waypoints[:, 0], bc_type="clamped")
        self.traj_y = CubicSpline(self.time, self.waypoints[:, 1], bc_type="clamped")
        self.traj_z = CubicSpline(self.time, self.waypoints[:, 2], bc_type="clamped")

        self.traj_dx = self.traj_x.derivative(nu=1)
        self.traj_dy = self.traj_y.derivative(nu=1)
        self.traj_dz = self.traj_z.derivative(nu=1)

        self.traj_ddx = self.traj_x.derivative(nu=2)
        self.traj_ddy = self.traj_y.derivative(nu=2)
        self.traj_ddz = self.traj_z.derivative(nu=2)

    def trajectory_xy(self):
        self.time = [0]
        self.waypoints = np.array([[0, 0, 0], [2, 3, 0], [6, 4, 0], [10, 4, 0], [13, 3, 0],
                                   [15, 0, 0], [13, -3, 0], [10, -4, 0], [6, -4, 0], [2, -3, 0],
                                   [0, 0, 0], [-2, 3, 0], [-6, 4, 0], [-10, 4, 0], [-13, 3, 0],
                                   [-15, 0, 0], [-13, -3, 0], [-10, -4, 0], [-6, -4, 0], [-2, -3, 0],
                                   [0, 0, 0], [2, 3, 0], [6, 4, 0], [10, 4, 0]])
        for i, waypoint in enumerate(self.waypoints[1:]):
            dist = np.linalg.norm(waypoint - self.waypoints[i-1])
            self.time.append(self.time[-1] + dist / self.v_des)

        # Fit cubic spline waypoints
        self.traj_x = CubicSpline(self.time, self.waypoints[:, 0], bc_type="clamped")
        self.traj_y = CubicSpline(self.time, self.waypoints[:, 1], bc_type="clamped")
        self.traj_z = CubicSpline(self.time, self.waypoints[:, 2], bc_type="clamped")

        self.traj_dx = self.traj_x.derivative(nu=1)
        self.traj_dy = self.traj_y.derivative(nu=1)
        self.traj_dz = self.traj_z.derivative(nu=1)

        self.traj_ddx = self.traj_x.derivative(nu=2)
        self.traj_ddy = self.traj_y.derivative(nu=2)
        self.traj_ddz = self.traj_z.derivative(nu=2)

    def plot_trajectory(self):
        t = np.arange(0, self.time[-1], self.drone.dt)
        positions = np.array([0, 0, 0])
        velocities = np.array([0, 0, 0])
        accelerations = np.array([0, 0, 0])

        for time in t:
            positions = np.vstack((positions, self.position_ref(time)))
            velocities = np.vstack((velocities, self.velocity_ref(time)))
            accelerations = np.vstack((accelerations, self.acceleration_ref(time)))

        fig2, ax = plt.subplots(3, 3)
        ax[0, 0].plot(t, positions[1:, 0])
        ax[0, 0].scatter(self.time, self.waypoints[:, 0])
        ax[0, 0].set_ylabel("X-position [m]")
        ax[0, 0].grid()

        ax[1, 0].plot(t, positions[1:, 1])
        ax[1, 0].scatter(self.time, self.waypoints[:, 1])
        ax[1, 0].set_ylabel("Y-position [m]")
        ax[1, 0].grid()

        ax[2, 0].plot(t, positions[1:, 2])
        ax[2, 0].scatter(self.time, self.waypoints[:, 2])
        ax[2, 0].set_ylabel("Z-position [m]")
        ax[2, 0].set_xlabel("Time [s]")
        ax[2, 0].grid()

        # for i in range(3):
        #     ax[i, 0].vlines(self.time, self.waypoints[:, i] - self.r, self.waypoints[:, i] + self.r, 'r')

        ax[0, 1].plot(t, velocities[1:, 0])
        ax[0, 1].set_ylabel("X-velocity [m/s]")
        ax[0, 1].grid()

        ax[1, 1].plot(t, velocities[1:, 1])
        ax[1, 1].set_ylabel("Y-velocity [m/s]")
        ax[1, 1].grid()

        ax[2, 1].plot(t, velocities[1:, 2])
        ax[2, 1].set_ylabel("Z-velocity [m/s]")
        ax[2, 1].set_xlabel("Time [s]")
        ax[2, 1].grid()


        ax[0, 2].plot(t, accelerations[1:, 0])
        ax[0, 2].set_ylabel("X-acceleration [m/s^2]")
        ax[0, 2].grid()

        ax[1, 2].plot(t, accelerations[1:, 1])
        ax[1, 2].set_ylabel("Y-acceleration [m/s^2]")
        ax[1, 2].grid()

        ax[2, 2].plot(t, accelerations[1:, 2])
        ax[2, 2].set_ylabel("Z-acceleration [m/s^2]")
        ax[2, 2].set_xlabel("Time [s]")
        ax[2, 2].grid()

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(positions[1:, 0], positions[1:, 1], positions[1:, 2])
        ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2])
        ax.set_xlabel("X-position [m]")
        ax.set_ylabel("Y-position [m]")
        ax.set_zlabel("Z-position [m]")



def test_trajectory_optimization():
    config = configs.config.Config()
    drone = Drone(config)
    trajectory = Trajectory(config, drone, "track", "CPC")
    trajectory.plot_trajectory()

    # fig, axs = plt.subplots(1, 4)
    # axs[0].plot(t[:-1], u[:, 0])
    # axs[1].plot(t[:-1], u[:, 1])
    # axs[2].plot(t[:-1], u[:, 2])
    # axs[3].plot(t[:-1], u[:, 3])
    #
    # fig1, ax = plt.subplots(3, 2)
    #
    # ax[0, 0].plot(t, s[:, 0] * 180 / np.pi)
    # ax[0, 0].set_ylabel("Roll Angle [deg]")
    # ax[0, 0].grid()
    #
    # ax[1, 0].plot(t, s[:, 1] * 180 / np.pi)
    # ax[1, 0].set_ylabel("Pitch Angle [deg]")
    # ax[1, 0].grid()
    #
    # ax[2, 0].plot(t, s[:, 2] * 180 / np.pi)
    # ax[2, 0].set_ylabel("Yaw Angle [deg]")
    # ax[2, 0].grid()
    #
    # ax[0, 1].plot(t, s[:, 3] * 180 / np.pi)
    # ax[0, 1].set_ylabel("Roll Rate [deg/s]")
    # ax[0, 1].grid()
    #
    # ax[1, 1].plot(t, s[:, 4] * 180 / np.pi)
    # ax[1, 1].set_ylabel("Pitch Rate [deg/s]")
    # ax[1, 1].grid()
    #
    # ax[2, 1].plot(t, s[:, 5] * 180 / np.pi)
    # ax[2, 1].set_ylabel("Yaw Rate [deg/s]")
    # ax[2, 1].grid()
    #
    # plt.tight_layout()
    #
    # plt.figure()
    # plt.plot(J[1:])

    plt.show()

test_trajectory_optimization()
