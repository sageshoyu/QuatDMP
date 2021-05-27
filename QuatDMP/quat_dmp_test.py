from quat_dmp import QuatDMP, directed_slerp
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

T = 4.416
n = 1000
dt = T / n
# demo = np.load('/home/seiji/Documents/BROWN_STUFF/ROBOTICS_LAB/skills_kin/sim/os_traj.npy').T
# demo_qs = demo[3:, :]
demo_qs = directed_slerp(R.random(3, random_state=1).as_quat().T, np.array([0,1,2]), np.linspace(0,2,n))
demo_len = demo_qs.shape[1]
key_times = np.arange(demo_len)

times = np.linspace(0, T, num=n)

dmp = QuatDMP(T, dt, n_bfs=30, K=1000 * np.eye(3), D=40*np.eye(3))
dmp.fit(demo_qs, tau=1.0)


q, qd, omega, omegad = dmp.run_sequence(start=demo_qs[:,0].reshape(-1,1), goal=demo_qs[:,-1].reshape(-1,1))

plt.plot(demo_qs[0], label='x', color='b', linestyle='-')
plt.plot(q[0], color='b', linestyle='--')

plt.plot(demo_qs[1], label='y', color='r', linestyle='-')
plt.plot(q[1], color='r', linestyle='--')

plt.plot(demo_qs[2], label='z', color='g', linestyle='-')
plt.plot(q[2], color='g', linestyle='--')

plt.plot(demo_qs[3], label='w', color='m', linestyle='-')
plt.plot(q[3], color='m', linestyle='--')
plt.legend()
plt.title('quat demo and dmp \nstart=' + str(demo_qs[:, 0]) + '\n end=' + str(demo_qs[:, -1]))
plt.show()

# also do euler conversion for more easily interpretable units
demo_rots = R.from_quat(demo_qs.T)
euler_demo = R.as_euler(demo_rots, 'XYZ').T
dmp_rots = R.from_quat(q.T)
euler_roll = R.as_euler(dmp_rots, 'XYZ').T

plt.plot(euler_demo[0], label='euler x', color='b', linestyle='-')
plt.plot(euler_roll[0], color='b', linestyle='--')

plt.plot(euler_demo[1], label='euler y', color='r', linestyle='-')
plt.plot(euler_roll[1], color='r', linestyle='--')

plt.plot(euler_demo[2], label='euler z', color='g', linestyle='-')
plt.plot(euler_roll[2], color='g', linestyle='--')
plt.legend()
plt.title('euler demo and dmp \n start=' + str(euler_demo[:,0])
          + '\n end=' + str(euler_demo[:,-1]))
plt.show()

# assume all rotations are relative to a global frame. We will be rotating
# a single point point at [0, 0, 1] on unit circle to plot the paths
start = np.array([0,1,0])
demo_path = demo_rots.apply(start)
dmp_path = dmp_rots.apply(start)
axis = omega[:, 0] / np.linalg.norm(omega[:, 0])

# change rotation axis to world frame
axis = R.inv(demo_rots[0]).apply(axis)

# plotting paths on sphere
fig = plt.figure()
ax = fig.gca(projection='3d')

# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="r", alpha=0.1)
ax.plot(demo_path[:, 0], demo_path[:, 1], demo_path[:, 2], color='b')
ax.plot(dmp_path[:, 0], dmp_path[:, 1], dmp_path[:, 2], color='orange')
ax.plot([demo_path[0, 0]], [demo_path[0, 1]], [demo_path[0, 2]], marker='o')
ax.plot([axis[0]], [axis[1]], [axis[2]], marker='o')
ax.plot([demo_path[-1, 0]], [demo_path[-1, 1]], [demo_path[-1, 2]], marker='o')
plt.show()