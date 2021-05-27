from scipy.spatial.transform.rotation import Slerp, Rotation as R
from cs import CanonicalSystem
import numpy as np


# implemented from:
# P. Pastor, L. Righetti, M. Kalakrishnan and S. Schaal,
# "Online movement adaptation based on previous sensor experiences"

# note: there was little information on how to extend basis-functions
# from scalar DMPs to quaternion DMPs

# Here, we naively construct 3 independent DMP basis functions + weights and couple them together with the
# system documented above

class QuatDMP(object):
    def __init__(self, T, dt, n_bfs=10, K=np.eye(3), D=np.eye(3)):
        self.K = K
        self.D = D
        self.T = T
        self.dt = dt
        self.n_bfs = n_bfs

        # canonical system
        a = 1.0
        self.cs = CanonicalSystem(a, T, dt)

        # set up basis functions and weights
        self.all_w = [np.zeros(n_bfs)] * 3

        self.all_centers = [None] * 3

        self.all_widths = [None] * 3
        self.set_basis_functions()

        # setup start and goal values 
        self.g = np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)
        self.q0 = np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)

        self.q = np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)
        self.qd = np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)
        self.omega = np.zeros((3, 1))
        self.omegad = np.zeros((3, 1))

    def reset(self):
        if self.q0 is not None:
            self.q = self.q0
        else:
            self.q0 = np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)
            self.q = np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)
        self.qd = np.zeros((4, 1))
        self.omega = np.zeros((3, 1))
        self.omegad = np.zeros((3, 1))
        self.cs.reset()

    def set_basis_functions(self):
        # this is exact basis function procedure done in Tsitsimis's dmpling
        time = np.linspace(0, self.T, self.n_bfs)
        for i in range(3):
            self.all_centers[i] = np.zeros(self.n_bfs)
            self.all_centers[i] = np.exp(-self.cs.a * time)
            self.all_widths[i] = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.all_centers[i] / self.cs.a
        

    def psi(self, theta):
        all_psi = []
        for i in range(3):

            # print(theta.shape)
            # print(self.all_widths[i].shape)
            # print(self.all_centers[i].shape)
            if isinstance(theta, np.ndarray):
                theta = theta.reshape(-1,1)
            all_psi.append(np.exp(-self.all_widths[i] * (theta - self.all_centers[i]) ** 2))

        return np.array(all_psi)

    def step(self, tau=1.0, start=None, goal=None, q=None):
        # this pretty much mirrors original DMP, except it replaces transformation system
        # with equivalent for quaternions
        if goal is None:
            g = self.g
        else:
            g = goal

        if start is None:
            q0 = self.q0
        else:
            q0 = start

        # do closed-loop orientation control if current orientation given
        if q is not None:
            # ensure that quats are within 90 degrees of each other
            # (so error term doesn't suddenly reverse direction)
            if np.sum(self.q * q) >= 0:
                self.q = q
            else:
                self.q = -q

        theta = self.cs.step(tau)
        psi = self.psi(theta)
        ws = np.array(self.all_w)

        # note: forcing term excludes error dependence in this dmp
        f = np.sum(ws * psi, axis=1, keepdims=True) * theta / np.sum(psi, axis=1, keepdims=True)

        # transformation system with quaternion error
        self.omegad = - np.dot(self.K, q_err(g, self.q)) \
                      - np.dot(self.D, self.omega) \
                      + np.dot(self.K, q_err(g, q0)) * theta \
                      + np.dot(self.K, f)

        self.omegad /= tau
        self.omega += self.omegad * self.dt

        # quaternion integration rule
        wd, xd, yd, zd = 0.5 * np.dot(np.block([
            [0, -self.omega.T],
            [self.omega, -q_cross(self.omega)]]),
            np.concatenate(([self.q[3]], self.q[:3])))

        self.qd = np.array([xd, yd, zd, wd]).reshape(-1, 1) / tau

        # integrate off of omega (easier integration formula, more accurate than linear approx)
        rq = R.from_quat(self.q.flatten()) * R.from_rotvec((self.omega * self.dt).flatten())
        self.q = R.as_quat(rq).reshape(-1, 1)

        return self.q, self.qd, self.omega, self.omegad

    def fit(self, q_demo, tau=1.0):
        assert q_demo.ndim == 2 and q_demo.shape[0] == 4

        self.q0 = q_demo[:, 0].copy().reshape(-1, 1)
        self.g = q_demo[:, -1].copy().reshape(-1, 1)

        # interpolate using SLERP
        key_times = np.linspace(0, self.T, num=q_demo.shape[1])
        dmp_times = np.array([i * self.dt for i in range(self.cs.N)])
        demo_interp = directed_slerp(q_demo, key_times, dmp_times)

        rs = R.from_quat(demo_interp.T)

        # compute omega (angular velocity)
        rs1_inv = R.inv(rs[:-1])
        rs2 = rs[1:]
        drs = rs2 * rs1_inv  # displacement rotation from t to t+1

        omega = (R.as_rotvec(drs) / self.dt).T
        omega = np.concatenate((omega[:, 0].reshape(-1, 1), omega), axis=1)  # trying to keep same length

        # angular acceleration
        omegad = np.diff(omega) / self.dt
        omegad = np.concatenate((omega[:, 1].reshape(-1, 1), omegad), axis=1)
        theta_seq = self.cs.all_steps()

        # solve for f_target
        f_target = tau * omegad + np.dot(self.K, q_err(self.g, demo_interp)) \
                   + np.dot(self.D, omega) - np.dot(self.K, q_err(self.g, self.q0)) * theta_seq
        f_target = np.dot(np.linalg.inv(self.K), f_target)  # K is likely a diagonal matrix, so this is okay

        psi_funs = self.psi(theta_seq)

        # Locally-weighted regression
        # to avoid confusion, this operation is done separately for each dmp
        # LWR is a single-shot procedure, so not much of a loss in performance anyway

        for i in range(3):
            w_aa = np.multiply(theta_seq.reshape((1, theta_seq.shape[0])), psi_funs[i].T)
            w_aa = np.multiply(w_aa, f_target[i].reshape((1, theta_seq.shape[0])))
            w_aa = np.sum(w_aa, axis=1)

            w_bb = np.multiply(theta_seq.reshape((1, theta_seq.shape[0])) ** 2, psi_funs[i].T)
            w_bb = np.sum(w_bb, axis=1)
            self.all_w[i] = w_aa / w_bb


        self.reset()

    def run_sequence(self, tau=1.0, start=None, goal=None):
        q = np.zeros((4, self.cs.N))
        qd = np.zeros((4, self.cs.N))
        omega = np.zeros((3, self.cs.N))
        omegad = np.zeros((3, self.cs.N))
        for i in range(self.cs.N):
            q_i, qd_i, omega_i, omegad_i = self.step(tau=tau, start=start, goal=goal)
            q[:, i] = q_i.flatten()
            qd[:, i] = qd_i.flatten()
            omega[:, i] = omega_i.flatten()
            omegad[:, i] = omegad_i.flatten()

        return q, qd, omega, omegad



def q_err(q1, q2):
    return q1[3] * q2[:3] - q2[3] * q1[:3] - np.dot(q_cross(q1), q2[:3])


def q_cross(q):
    # consistent with Scipy (scalar is last)
    return np.array([[0, -q[2, 0], q[1, 0]],
                     [q[2, 0], 0, -q[0, 0]],
                     [-q[1, 0], q[0, 0], 0]])


# implementation of slerp algorithm
# without the dot product check, so that the path can take the 'long way'
# around if desired (because the error term will force the dmp
# to do that anyway)
# NOTE: this is an incredibly inefficient implementation.
# HOWEVER - this is only supposed to be run once during
# DMP fitting at the start of the experiment, so it's okay
# may have to think about euler vectors for a more efficient version if necessary...
def directed_slerp(qs, key_times, desired_times):
    """Given a list of quaternions and a list of associated timestamps,
       and a list of desired timestamps, give a list of slerp-interpolated
       quaternions at those specific timestamps

       NOTE: this doesn't check if SLERP goes the 'long way' around the
       sphere (on purpose) to provide correct demonstration data
       that matches the way the Schaal's error term work with DMP
    """
    assert qs.shape[0] == 4
    assert qs.shape[1] == key_times.size
    assert desired_times[0] >= key_times[0] and desired_times[-1] <= key_times[-1]

    res = np.zeros((4, len(desired_times)))
    t_inds = np.searchsorted(key_times, desired_times, side='left')
    for i in range(len(desired_times)):
        t = desired_times[i]
        t_ind = t_inds[i]

        # if the earliest time is equal to the start time, then push the index by one to stay within bounds
        if i == 0 and t == key_times[0]:
            t_ind += 1

        t_norm = (t - key_times[t_ind - 1]) / (key_times[t_ind] - key_times[t_ind - 1])
        res[:, i] = directed_single_slerp(qs[:, t_ind-1].reshape(-1,1),
                                              qs[:, t_ind].reshape(-1,1), t_norm).flatten()

    return res



# formula and algorithm modified from
# https://en.wikipedia.org/wiki/Slerp
def directed_single_slerp(q0, q1, t):
    """Given two quaternions and a time t between [0, 1], compute interpolation at
    intermediate time t, where t=0 is q0 and t=1 is 11"""
    assert t <= 1.0 or t >= 0.0
    assert q0.shape == (4, 1) and q1.shape == (4, 1)

    dot = np.sum(q0 * q1)
    DOT_THRESHOLD = 0.9995

    # if rotations are super close, just linearly interpolate
    if dot > DOT_THRESHOLD:
        res = (q1 - q0) * t + q0
        return res / np.linalg.norm(res)

    # if not, use spherical interpolation formula
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    res = q0 * np.sin((1 - t) * theta) / sin_theta + q1 * np.sin(t * theta) / sin_theta
    return res / np.linalg.norm(res)

