import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

traj_spline = pd.read_csv('traj_spline')
wp_spline = pd.read_csv('wp_spline')
traj_SCP = pd.read_csv('traj_SCP')
wp_SCP = pd.read_csv('wp_SCP')
traj_CPC = pd.read_csv('traj_CPC')
wp_CPC = pd.read_csv('wp_CPC')

fig2, ax = plt.subplots(1, 3)
ax[0].plot(traj_spline['time'], traj_spline['px'], label="Baseline")
ax[0].scatter(wp_spline['time_stamps'], wp_spline['wp_x'])
ax[0].plot(traj_SCP['time'], traj_SCP['px'], label="SCP")
ax[0].scatter(wp_SCP['time_stamps'], wp_SCP['wp_x'])
ax[0].plot(traj_CPC['time'], traj_CPC['px'], label="CPC")
ax[0].scatter(wp_CPC['time_stamps'], wp_CPC['wp_x'])
ax[0].set_ylabel("X-position [m]")
ax[0].set_xlabel("Time [s]")
ax[0].grid()
ax[0].legend()

ax[1].plot(traj_spline['time'], traj_spline['py'])
ax[1].scatter(wp_spline['time_stamps'], wp_spline['wp_y'])
ax[1].plot(traj_SCP['time'], traj_SCP['py'])
ax[1].scatter(wp_SCP['time_stamps'], wp_SCP['wp_y'])
ax[1].plot(traj_CPC['time'], traj_CPC['py'])
ax[1].scatter(wp_CPC['time_stamps'], wp_CPC['wp_y'])
ax[1].set_ylabel("Y-position [m]")
ax[1].set_xlabel("Time [s]")
ax[1].grid()

ax[2].plot(traj_spline['time'], traj_spline['pz'])
ax[2].scatter(wp_spline['time_stamps'], wp_spline['wp_z'])
ax[2].plot(traj_SCP['time'], traj_SCP['pz'])
ax[2].scatter(wp_SCP['time_stamps'], wp_SCP['wp_z'])
ax[2].plot(traj_CPC['time'], traj_CPC['pz'])
ax[2].scatter(wp_CPC['time_stamps'], wp_CPC['wp_z'])
ax[2].set_ylabel("Z-position [m]")
ax[2].set_xlabel("Time [s]")
ax[2].grid()

plt.show()


