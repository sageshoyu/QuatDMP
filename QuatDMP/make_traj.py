from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np

start = np.array([0.0,0.0,0.0,1.0])
end = np.array([1.0,1.0,1.0,1.0]) / 2

key_times = [0, 1]
key_rots = R.from_quat(np.stack((start, end)))
slerp = Slerp(key_times, key_rots)
times = np.linspace(0.0, 1.0, num=100)
interp_rots = slerp(times)
traj = interp_rots.as_quat()
np.save('ex_traj.npy', traj.T)