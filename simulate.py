from flying_sim.drone import Drone
from flying_sim.controllers import AttitudeController, ControlAllocation, PositionController
from flying_sim.trajectory import Trajectory
from configs.config import Config

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse

matplotlib.use('TkAgg')


class Simulate:
    def __init__(self, seed, trajectory_type="spline", waypoints="random"):
        self.config = Config()

        # Initialize drone, controllers and trajectory
        self.drone = Drone(self.config)
        self.position_controller = PositionController(self.drone)
        self.attitude_controller = AttitudeController(self.drone)
        self.control_allocation = ControlAllocation(self.drone)
        self.trajectory = Trajectory(self.config, self.drone, waypoints, trajectory_type, seed=seed)

        self.reset()

    def reset(self):
        # Reset drone
        self.drone.reset()

        # Logging signals
        self.states = self.drone.state
        self.aux_states = np.array([0, 0, 0, 0, 0, 0])
        self.inputs = np.array([0, 0, 0, 0])
        self.angular_vel_ref = np.array([0, 0, 0])
        self.attitude_ref = np.array([0, 0, 0])
        self.acceleration_ref = np.array([0, 0, 0])
        self.acceleration_des = np.array([0, 0, 0])
        self.velocity_ref = self.trajectory.velocity_ref(0.)
        self.position_ref = self.trajectory.position_ref(0.)
        self.raw_inputs = np.array([0, 0, 0, 0])
        self.wp_completion = np.zeros((len(self.trajectory.waypoints), ))
        self.time = [0]

    def simulate(self):

        # Initialize time and control inputs
        control_moment = np.zeros((3,))
        thrust_des = -self.drone.m * self.drone.g

        # Simulation loop
        completed = False
        while self.time[-1] < self.trajectory.time[-1] * 1.1:
            # Reference trajectory
            pos_ref = self.trajectory.position_ref(self.time[-1])
            vel_ref = self.trajectory.velocity_ref(self.time[-1])
            acc_ref = self.trajectory.acceleration_ref(self.time[-1])

            # Position controller
            des_lin_acc = self.position_controller.get_desired_lin_acc(self.drone.position, self.drone.velocity_e, self.drone.lin_acc, pos_ref, vel_ref, acc_ref)
            att_des, thrust_des = self.position_controller.get_desired_attitude(self.drone.attitude, thrust_des, self.drone.lin_acc, des_lin_acc)
            yaw_des = 0     # np.arctan2(drone.state[10], drone.state[9])
            des_attitude = np.array([att_des[0], att_des[1], yaw_des])        # Desired attitude

            # Attitude controller
            des_angular_vel = self.attitude_controller.get_des_angular_vel(self.drone.attitude, des_attitude)
            control_moment = self.attitude_controller.get_control_moment(control_moment, self.drone.angular_velocity, des_angular_vel)
            control_input = self.control_allocation.get_control_input(control_moment, thrust_des)

            # Time step for drone
            self.drone.step(control_input)

            # Log states and time
            self.states = np.vstack((self.states, self.drone.state.copy()))
            self.aux_states = np.vstack((self.aux_states, np.hstack((self.drone.velocity_e, self.drone.lin_acc))))
            self.inputs = np.vstack((self.inputs, control_input))
            self.angular_vel_ref = np.vstack((self.angular_vel_ref, des_angular_vel))
            self.attitude_ref = np.vstack((self.attitude_ref, des_attitude))
            self.acceleration_des = np.vstack((self.acceleration_des, des_lin_acc))
            self.acceleration_ref = np.vstack((self.acceleration_ref, acc_ref))
            self.velocity_ref = np.vstack((self.velocity_ref, vel_ref))
            self.position_ref = np.vstack((self.position_ref, pos_ref))
            self.raw_inputs = np.vstack((self.raw_inputs, self.drone.inputs))
            self.time.append(self.time[-1] + self.drone.dt)

            # Waypoint completion condition
            for i, waypoint in enumerate(self.trajectory.waypoints):
                if np.linalg.norm(waypoint - self.drone.position) < self.trajectory.r and np.sum(self.wp_completion) == i:
                    self.wp_completion[i] = 1
            if np.all(self.wp_completion):
                completed = True
                break

        if completed:
            print(f"Drone reach last waypoint in {self.time[-1]} sec over path length of {self.trajectory.path_length()} m.")
        else:
            print(f"Drone did not complete all waypoints: {self.wp_completion}, path length is {self.trajectory.path_length()} m.")

        # np.savetxt("initial_trajectory.csv", self.states)
        # np.savetxt("initial_inputs.csv", self.raw_inputs)

    def evaluate(self, n=100):
        max_dev = []
        max_vel = []
        rmse_pos = []
        rms_vel = []
        for i in range(n):
            self.simulate()
            pos_dev = np.linalg.norm(self.states[:, 6:9] - self.position_ref, axis=1)
            velocity = np.linalg.norm(self.aux_states[:, 0:3], axis=1)

            max_dev.append(np.max(pos_dev))
            rmse_pos.append(np.sqrt((pos_dev ** 2).mean()))

            max_vel.append(np.max(velocity))
            rms_vel.append(np.sqrt((velocity ** 2).mean()))
            self.reset()
            print("Simulation Count: ", i+1)

        print("============================================")
        print("max(||x - x_ref||) :", np.array(max_dev).mean())
        print("RMSE(||x - x_ref||) :", np.array(rmse_pos).mean())
        print("max(||v||) :", np.array(max_vel).mean())
        print("RMS(||v||) :", np.array(rms_vel).mean())
        print("============================================")

    def plot(self):
        self.simulate()
        # np.savetxt("data/position_disturbance_conventional.csv", self.states[:, 6:9])

        # Plot states
        fig1, ax = plt.subplots(3, 2)

        ax[0, 0].plot(self.time, self.states[:, 0] * 180 / np.pi, label="state")
        ax[0, 0].plot(self.time, self.attitude_ref[:, 0] * 180 / np.pi, label="ref")
        ax[0, 0].set_ylabel("Roll Angle [deg]")
        ax[0, 0].grid()

        ax[1, 0].plot(self.time, self.states[:, 1] * 180 / np.pi, self.time, self.attitude_ref[:, 1] * 180 / np.pi)
        ax[1, 0].set_ylabel("Pitch Angle [deg]")
        ax[1, 0].grid()

        ax[2, 0].plot(self.time, self.states[:, 2] * 180 / np.pi, self.time, self.attitude_ref[:, 2] * 180 / np.pi)
        ax[2, 0].set_ylabel("Yaw Angle [deg]")
        ax[2, 0].grid()

        ax[0, 1].plot(self.time, self.states[:, 3] * 180 / np.pi, self.time, self.angular_vel_ref[:, 0] * 180 / np.pi)
        ax[0, 1].set_ylabel("Roll Rate [deg/s]")
        ax[0, 1].grid()

        ax[1, 1].plot(self.time, self.states[:, 4] * 180 / np.pi, self.time, self.angular_vel_ref[:, 1] * 180 / np.pi)
        ax[1, 1].set_ylabel("Pitch Rate [deg/s]")
        ax[1, 1].grid()

        ax[2, 1].plot(self.time, self.states[:, 5] * 180 / np.pi, self.time, self.angular_vel_ref[:, 2] * 180 / np.pi)
        ax[2, 1].set_ylabel("Yaw Rate [deg/s]")
        ax[2, 1].grid()

        ax[0, 0].legend()
        plt.tight_layout()


        fig2, ax = plt.subplots(3, 3)
        ax[0, 0].plot(self.time, self.states[:, 6], self.time, self.position_ref[:, 0])
        ax[0, 0].scatter(self.trajectory.time, self.trajectory.waypoints[:, 0])
        ax[0, 0].set_ylabel("X-position [m]")
        ax[0, 0].grid()

        ax[1, 0].plot(self.time, self.states[:, 7], self.time, self.position_ref[:, 1])
        ax[1, 0].scatter(self.trajectory.time, self.trajectory.waypoints[:, 1])
        ax[1, 0].set_ylabel("Y-position [m]")
        ax[1, 0].grid()

        ax[2, 0].plot(self.time, self.states[:, 8], self.time, self.position_ref[:, 2])
        ax[2, 0].scatter(self.trajectory.time, self.trajectory.waypoints[:, 2])
        ax[2, 0].set_ylabel("Z-position [m]")
        ax[2, 0].set_xlabel("Time [s]")
        ax[2, 0].grid()

        ax[0, 1].plot(self.time, self.aux_states[:, 0], self.time, self.velocity_ref[:, 0])
        ax[0, 1].set_ylabel("X-velocity [m/s]")
        ax[0, 1].grid()

        ax[1, 1].plot(self.time, self.aux_states[:, 1], self.time, self.velocity_ref[:, 1])
        ax[1, 1].set_ylabel("Y-velocity [m/s]")
        ax[1, 1].grid()

        ax[2, 1].plot(self.time, self.aux_states[:, 2], self.time, self.velocity_ref[:, 2])
        ax[2, 1].set_ylabel("Z-velocity [m/s]")
        ax[2, 1].set_xlabel("Time [s]")
        ax[2, 1].grid()

        ax[0, 2].plot(self.time, self.aux_states[:, 3], label="State")
        ax[0, 2].plot(self.time, self.acceleration_ref[:, 0], label="Ref")
        ax[0, 2].plot(self.time, self.acceleration_des[:, 0], label="Des")
        ax[0, 2].set_ylabel("X-acceleration [m/s2]")
        ax[0, 2].grid()

        ax[1, 2].plot(self.time, self.aux_states[:, 4], self.time, self.acceleration_ref[:, 1], self.time, self.acceleration_des[:, 1])
        ax[1, 2].set_ylabel("Y-acceleration [m/s2]")
        ax[1, 2].grid()

        ax[2, 2].plot(self.time, self.aux_states[:, 5], self.time, self.acceleration_ref[:, 2], self.time, self.acceleration_des[:, 2])
        ax[2, 2].set_ylabel("Z-acceleration [m/s2]")
        ax[2, 2].set_xlabel("Time [s]")
        ax[2, 2].grid()

        plt.tight_layout()


        fig3, ax = plt.subplots(4, 1)

        ax[0].plot(self.time, np.sqrt(np.abs(self.inputs[:, 0])) * 60 / (2 * np.pi))
        ax[0].plot(self.time, self.drone.max_rotor_speed * np.ones(len(self.time),) * 60 / (2 * np.pi))
        ax[0].set_ylabel("rotational velocity 1 [rpm]")
        ax[0].grid()

        ax[1].plot(self.time, np.sqrt(np.abs(self.inputs[:, 1])) * 60 / (2 * np.pi))
        ax[1].set_ylabel("rotational velocity 2 [rad/s]")
        ax[1].grid()

        ax[2].plot(self.time, np.sqrt(np.abs(self.inputs[:, 2])) * 60 / (2 * np.pi))
        ax[2].set_ylabel("rotational velocity 3 [rad/s]")
        ax[2].grid()

        ax[3].plot(self.time, np.sqrt(np.abs(self.inputs[:, 3])) * 60 / (2 * np.pi))
        ax[3].set_ylabel("rotational velocity 4 [rad/s]")
        ax[3].grid()

        # fig4, ax = plt.subplots(3, 1)
        #
        # ax[0].plot(self.time[:-1], self.drone.thrust[::4])
        # ax[0].set_ylabel("Drag force magnitude [N]")
        # ax[0].grid()
        #
        # ax[1].plot(self.time[:-1], self.drone.gravity[::4])
        # ax[1].set_ylabel("Gravity force magnitude [N]")
        # ax[1].grid()
        #
        # ax[2].plot(self.time[:-1], self.drone.drag[::4])
        # ax[2].set_ylabel("Drag force magnitude [N]")
        # ax[2].grid()

        plt.tight_layout()

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(self.states[:, 6], self.states[:, 7], self.states[:, 8])
        ax.plot(self.position_ref[:, 0], self.position_ref[:, 1], self.position_ref[:, 2])
        ax.scatter(self.trajectory.waypoints[:, 0], self.trajectory.waypoints[:, 1], self.trajectory.waypoints[:, 2])
        ax.set_xlabel("X-position [m]")
        ax.set_ylabel("Y-position [m]")
        ax.set_zlabel("Z-position [m]")

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Simulation',
        description='This program simulates the trajectory controller, either summarizing some metrics or plotting results',
        epilog='Text at the bottom of help')
    parser.add_argument("--action", default="plot", choices=["plot", "metrics"])
    parser.add_argument('--trajectory', default="spline", choices=["spline", "optimize", "CPC"])
    parser.add_argument('--waypoints', default="random", choices=["random", "track", "figure-8", "hover"])
    parser.add_argument("--num_eval", default=100, type=int)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if type(args.seed) is int:
        print(f"Set seed to {args.seed}")
        np.random.seed(args.seed)

    # Run simulations
    simulate = Simulate(args.seed, trajectory_type=args.trajectory, waypoints=args.waypoints)
    if args.action == "plot":
        print(f"Plot simulation results on {args.trajectory} trajectory.")
        simulate.plot()
    elif args.action == "metrics":
        print(f"Calculate evaluation metrics on {args.trajectory} trajectory over {args.num_eval} simulations")
        simulate.evaluate(args.num_eval)
