import numpy as np
import control as ct
from scipy.linalg import expm
from scipy.optimize import fsolve


kts2mps_ = 1852. / 3600.
mps2kts_ = 1 / kts2mps_


def match_sos(steady_delay, overshoot):
    """
    second order system
    :param steady_delay:
    :param overshoot:
    :return:
    """
    def sos_mag(omega_, g_, w0_):
        return 1 / np.sqrt((1 - (omega_ / w0_) ** 2) ** 2 + (2 * g_ * omega_ / w0_) ** 2)

    def sos_phase(omega_, g_, w0_):
        return np.arctan2(-2 * g_ * omega_ / w0_, 1 - (omega_ / w0_) ** 2)

    def sosd_tf(omega_, g_, w0_, b_):
        num = np.array([0, 2j * b_ / w0_, 1])
        den = np.array([-1 / w0_ ** 2, 2j * g_ / w0_, 1])
        wvec = np.array([omega_ ** 2, omega_, 1])

        tf = (num @ wvec) / (den @ wvec)
        mag_ = np.abs(tf)
        phase_ = np.angle(tf)

        return mag_, phase_, num, den

    wzero = 1e-5
    # tzero = 0.1
    # mp = 0.1
    sol = fsolve(lambda x: np.array([sos_phase(wzero, x[0], x[1]) / wzero + steady_delay,
                                     # sos_mag(wstar, x[0], x[1]) - mstar]),
                                     np.exp(-x[0] * np.pi / (np.sqrt(1 - x[0] ** 2))) - overshoot]),
                 [1 / np.sqrt(2), 1])

    g, w0 = list(sol)
    a = np.array([[0, 1], [-w0**2, -2*g*w0]])
    b = np.array([0, w0**2])
    c = np.array([1, 0])
    d = np.array([0])

    return g, w0, a, b, c, d


def discretize(ac, bc, dt):
    """! Discretize a linear state space representation given a sample time. Equivalent to
    control.sample_system(ss_system, dt, method='zoh')
    @param ac: Continuous state matrix
    @param bc: Continuous input matrix
    @param dt: Sample time
    @returns Discretized state and input matrices"""

    #  Digital Control of Dynamic Systems, Gene F. Franklin, page 106, expression 4.58:
    ad = expm(ac * dt)
    expat = quad5(lambda t_: expm(ac * t_), dt)
    bd = expat @ bc
    return ad, bd


def kts2mps(x):
    """! Multiplies the input by the knots to m/s conversion factor
    @param x: Speed expressed in knots
    @returns: Speed expressed in m/s
    """
    return x * kts2mps_


def mps2kts(x):
    """! Multiplies the input by the m/s to knots conversion factor
    @param x: Speed expressed in m/s
    @returns: Speed expressed in knots
    """
    # return x / kts2mps_
    return x * mps2kts_


def dlqri(a, b, q, r, n=None, ni=0, dofs_i_local=[], dt=1, bryson=False):
    """! Computes the optimal linear quadratic state feedback controller for a given linear system.
    The discretization of the optimization problem is managed internally
    @param a: Discrete state space representation dynamics  matrix
    @param b: Discrete state space representation input matrix
    @param q: State weight matrix
    @param r: Input weight matrix
    @param ni: Number of integrators
    @param dofs_i_local: Indexes of the states that will be integrated
    @param dt: Sample time
    @param bryson: Select whether to use Bryson's rule for weight matrices or not
    """

    if n is None:
        n = np.zeros((len(q), len(r)))
    if bryson:
        q = np.diag(1 / np.array(q) ** 2)
        r = np.diag(1 / np.array(r) ** 2)
        n = np.array([[1/el if el != 0 else 0 for el in row] for row in n]) # Avoid division by zero
        
    # Digital Control of Dynamic Systems, Gene F. Franklin, page 324, expression 8.84:
    c = np.eye(a.shape[0])
    if dofs_i_local is None:
        dofs_i_local = np.arange(ni)
    else:
        ni = len(dofs_i_local)
        
    ci = c[dofs_i_local, :] # Modified to allow for any dof to be integrated

    aa = np.block([[np.eye(ni), ci],
                   [np.zeros((a.shape[0], ni)), a]])
    ba = np.block([[np.zeros((ni, b.shape[1]))], [b]])

    qd, rd, nd = discretize_cost(aa, ba, q, r, n, dt=dt)

    # print("Qd: ", qd)
    # print("Rd: ", rd)
    # print("Nd: ", nd) 
    
    
    def print_matlab_matrix(matrix, label="matrix"):
      rows, cols = matrix.shape
      print(f"{label} = [")
      for i in range(rows):
          row_str = ", ".join([f"{elem:.15e}" for elem in matrix[i]])
          print(f"{row_str};" if i < rows - 1 else f"{row_str}")
      print("];")
        
    try:
        (k, s, e) = ct.dlqr(aa, ba, qd, rd, nd)
        # (k, s, e) = ct.matlab.dlqr(aa, ba, qd, rd, nd)
        # print(f"Worked:")
        # print_matlab_matrix(aa, "aa")
        # print_matlab_matrix(ba, "ba")
        # print_matlab_matrix(qd, "qd")
        # print_matlab_matrix(rd, "rd")
        # print_matlab_matrix(nd, "nd")
        # print_matlab_matrix(k, "k")

    except Exception as e:
        print("Error in dlqri: ", e)
        print_matlab_matrix(aa, "aa")
        print_matlab_matrix(ba, "ba")
        print_matlab_matrix(qd, "qd")
        print_matlab_matrix(rd, "rd")
        print_matlab_matrix(nd, "nd")
        # print(f"Det(A) = {np.linalg.det(aa)}")
        # print(f"Det(Q) = {np.linalg.det(qd)}")
        # print(f"Det(R) = {np.linalg.det(rd)}")
        raise e
    return (k, s, e)


class Settings:
    """! Control Settings class (struct).
    Defines the control configuration parameters of the pyAiry.Control class."""

    def __init__(self, k, dt, sp, u0, ulims, input_limits, mts, dofs, integrator_idxs, sim_data={}, 
                 setpoint_steps: dict={}, input_deadband=0.0, adaptive_lqr=None,
                 sp_max_rates_of_change=None, anti_windup=[], cascade_roll_gain=1, cascade_z_gain=0,
                 step_smoothing='linear', sp_jump_factor=1.0, symmetric_turn_mask=None):
        """! Initializes all settings members.
        @param k: Gain matrix of the control loop
        @param dt: Sample time of the control loop
        @param sp: Setpoint array of the control
        @param u0: Initial control outputs
        @param ulims: Output range in the form of [min, max] (not used but kept for backwards compatibility)
        @param input_limits: Input limits in the form of [[min, max, ui_min, ui_max, max_rate], ...]
        @param mts: Mode transition speed array in the form of [takeoff, cruise]
        @param dofs: Degrees of freedom willing to control. For height, roll and pitch control
        @param integrator_idx: Indexes of the states that will be integrated
        @param sim_data: Simulation data dictionary
        @param setpoint_steps: Setpoint steps dictionary
        @param input_deadband: Input deadband
        @param adaptive_lqr: Adaptive LQR object
        """
        self.dt = dt
        self.sp = sp
        self.setpoint_steps = setpoint_steps
        self.setpoint_times = [float(t) for t in setpoint_steps.keys()]
        self.sp_max_rates_of_change = np.array([np.inf] * len(sp) if sp_max_rates_of_change is None else sp_max_rates_of_change)
        self.u0 = u0
        self.ulims = ulims,
        self.input_limits = input_limits
        self.mts = mts
        self.dofs = dofs
        self.k = k
        self.integrator_idxs = integrator_idxs
        self.sim_data = sim_data
        self.input_deadband = input_deadband
        self.adaptive_lqr = adaptive_lqr
        self.anti_windup = anti_windup
        self.cascade_roll_gain = cascade_roll_gain
        self.cascade_z_gain = cascade_z_gain 
        self.step_smoothing = step_smoothing
        self.sp_jump_factor = sp_jump_factor
        self.symmetric_turn_mask = symmetric_turn_mask


class Control:
    """! PyAiry Control class.
    Constructor (controller setup method) should be called only once. After that, step() method
    must be called every sample time interval during a simulation."""
    def __init__(self, s: Settings):
        """! Control constructor.
        Fill it with a Settings instance.
        @param s: PyAiry Settings instance"""
        self.settings = s
        self.x_error = np.zeros(self.settings.k.shape[1])
        self.x_ref = np.zeros(len(self.settings.dofs))
        self.x_ref[self.settings.integrator_idxs] = self.settings.sp
        self.ui = self.settings.u0 
        self.ux = np.zeros(self.settings.u0.size)
        self.u = self.ui
        self.transitioning = False
        self.was_clipped = False
        

    def step(self, x_current, v):
        """! Propagates the control forward in time when called deterministically every sample
        time.
        @param x: State vector of the vehicle (only dofs of interest)
        @param v: Speed along x-axis in local frame [m/s]
        @updates: Control.u class attribute member
        """

        sf = self._sf(v) # Speed factor
        k = self.get_k_matrix(x_current)
            
        integrated_states_idxs = self.settings.integrator_idxs
        self.x_ref = np.zeros(len(self.settings.dofs))
        self.x_ref[integrated_states_idxs] = self.settings.sp
        
        # Cascade control
        if bool(self.settings.cascade_roll_gain): # Feed roll angle error to roll rate setpoint
            roll_idx = self.settings.sim_data['dofs'].index('roll') # Index of roll angle
            p_idx = self.settings.sim_data['dofs'].index('p') # Index of roll rate
            self.x_ref[p_idx] += -1 * self.settings.cascade_roll_gain * self.x_error[roll_idx]
        if bool(self.settings.cascade_z_gain): # Feed altitude error to pitch angle setpoint
            z_idx = self.settings.sim_data['dofs'].index('z') # Index of altitude
            pitch_idx = self.settings.sim_data['dofs'].index('pitch')
            unit_scaling = np.deg2rad(1) * 1e2 # So that gain can be expressed in deg/cm (more inuitive than rad/m)
            self.x_ref[pitch_idx] += self.settings.cascade_z_gain * self.x_error[z_idx] * unit_scaling
        
        self.x_error = (x_current - self.x_ref)
        
        # Integral control
        ui_delta = -k[:, integrated_states_idxs] @ self.x_error[integrated_states_idxs] * sf
        self.ui += ui_delta
        self.ui = self._anti_windup(self.ui)
        
        # Proportional control
        self.ux = -k[:, len(integrated_states_idxs):] @ self.x_error * sf
        
        # Sum of integral and proportional control
        u = self.ui + self.ux
        u = np.where(abs(u) < self.settings.input_deadband, 0, u) # Apply deadband
        self.u = u

    
    def _anti_windup(self, ui):
        
        ui = ui.copy()
        
        if 'bc' in self.settings.anti_windup: # Anti-windup back-calculation
            t_bc = self.settings.anti_windup[self.settings.anti_windup.index('bc')+1] # Back-calculation time constant (larger -> slower ui recovery)
            ui += (self.u - ui) * self.settings.dt / t_bc
            
        if 'cl' in self.settings.anti_windup: # Anti-windup clamping
            ui, _ = self.clip_inputs(ui, self.settings.input_limits, ui=True)
            
        return ui
    
       
    def update_setpoints(self, t):
        """! Updates the setpoint of the control loop.
        @param t: Current time of the simulation
        @updates: Control.settings.sp class attribute member
        """

        if round(t, 2) in self.settings.setpoint_times: # New target setpoint
            self.new_setpoint = self.settings.setpoint_steps[str(t)]
            self.t0 = t
            self.old_setpoint = self.settings.sp
            self.delta_setpoint = self.new_setpoint - self.old_setpoint
            transition_times = np.array(abs(self.delta_setpoint)) / self.settings.sp_max_rates_of_change
            self.transition_time = max(transition_times) # Time to reach the new setpoint
            self.transitioning = True
            # print(f"Transition times: {transition_times.round(2)} s")
            # print(f"\nStarting setpoint transition to {self.new_setpoint} at {t:.2f} s")
        
        if self.transitioning: # Smooth transition
            # If current setpoint close enough to target setpoint, end transition
            if any(abs(self.settings.sp - self.new_setpoint) > abs(self.delta_setpoint) * self.settings.dt):
                if self.settings.step_smoothing == 'linear': # Linear transition
                    self.settings.sp += self.delta_setpoint * self.settings.dt / self.transition_time
                elif self.settings.step_smoothing == 'sigmoid': # Sigmoid transition
                    self.settings.sp = self.old_setpoint + self.delta_setpoint / (1 + np.exp(-4 * (t - self.t0 - self.transition_time / self.settings.sp_jump_factor) / self.transition_time))
                else: # No step smoothing
                    self.settings.sp = self.new_setpoint
                # print(f"\nCurrent setpoint: {self.settings.sp}")
            else: # End of transition
                self.settings.sp = self.new_setpoint
                self.transitioning = False
                # print(f"\nTransition ended at {t:.2f} s")
                

    def _sf(self, speed):
        """! Internal method. Calculates the speed factor of the inner loop of the control.
        @param speed: Speed in [m/s]
        @returns: Speed factor
        """
        return (self.settings.mts[1] / max(self.settings.mts[0], speed)) ** 2
    
    def get_k_matrix(self, x):
        """! Internal method. Calculates the gain matrix of the control loop.
        Handles adaptive LQR if specified in the settings, and applies symmetric turn mask if needed.
        @param x: Current state vector
        @returns: Gain matrix
        """
        if self.settings.adaptive_lqr is not None:
            self.settings.k, _ = self.settings.adaptive_lqr._k(x, setpoints=self.settings.sp)
            
        roll = x[self.settings.sim_data['dofs'].index('roll')]
        sp_roll = self.settings.sp[self.settings.sim_data['dofs_i'].index('roll')]
        invert_gains = (sp_roll < 0) or (sp_roll == 0 and roll < 0)
        k = self.settings.k.copy()
        if self.settings.symmetric_turn_mask is not None and invert_gains: # If turning left, invert some gains to make the controller symmetric
            k *= self.settings.symmetric_turn_mask
            print("Inverted gains")
        return k
    
    @staticmethod
    def clip_inputs(inputs, limits, ui=False):
        """
        Clips the inputs to the limits specified
        @param inputs: array-like, values of the inputs
        @param limits: list of lists, lower and upper limits for each input
        param ui: boolean, True if the it's the integrated inputs that are being clipped (used by anti-windup clamping)
        @return: array-like, clipped values of the inputs and a boolean indicating if any input was clipped
        """
        if len(limits) != len(inputs):
            raise ValueError("The number of inputs and limits must be the same")
                
        (min_idx, max_idx) = (2, 3) if ui else (0, 1) # Use different limits for integrated and non-integrated inputs
        
        was_clipped = False
        for i in range(len(inputs)):
            if limits[i] is None:
                continue
            if inputs[i] < limits[i][min_idx]:
                inputs[i] = limits[i][min_idx]
                was_clipped = True
            elif inputs[i] > limits[i][max_idx]:
                inputs[i] = limits[i][max_idx]
                was_clipped = True

        return inputs, was_clipped
  

class NoControl:
    """! NoControl class should be used when no control action is required."""
    def __init__(self, u0):
        """! No control constructor.
        @param u0: Fixed input array
        """
        self.u = u0

    def step(self, x, v):
        """! Does nothing. It has the same arguments as Control.step() method just to keep it
        consistent
        @param x: State vector of the vehicle
        @param v: Speed magnitude in [m/s]
        """
        pass
       
            
def quad5(f, b: float, a: float = 0):
    """! Integrate up to a bidimensional array function using a Gauss-Legendre quadrature of 5th
    order.
    @param f: Integrand function
    @param b: Endpoint of the interval
    @param a: Initial point of the interval
    @returns: Calculated quadrature"""
    def cv(x, x0, x1):
        return (x1 - x0) / 2 * x + (x0 + x1) / 2

    p0 = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
    p = cv(p0, a, b)
    w = np.array([5, 8, 5]) / 9

    q = []
    for i in range(3):
        q.append(w[i] * f(p[i]))

    return sum(q) * (b - a) / 2


def discretize_cost(a, b, q, r, n=None, dt=0.01):
    """! Discretizes the cost function of an LQR optimal control scheme.
    @param a: Discrete state space representation matrix, A, of a linearized version of the
    system of interest, that is: xk+1 = A @ xk + B @ uk
    @param b: Discrete state space representation matrix, B, of a linearized version of the
    system of interest, that is: xk+1 = A @ xk + B @ uk
    @param q: Weights of the Model Based Control (i+LQR) optimal control tuning method
    corresponding to state weighting, that is: J = integral(xT @ diag(q) @ x + uT @ diag(r) @ u)
    for all time. The discretization of the optimization problem is managed automatically
    @param r: Weights of the Model Based Control (i+LQR) optimal control tuning method
    corresponding to input weighting
    @param n: Weights of the Model Based Control (i+LQR) optimal control tuning method
    corresponding to cross weighting, that is: J = integral(xT @ diag(q) @ x + uT @ diag(r) @ u + 2 * xT @ n @ u)
    @param dt: Sample time of the control loop
    @returns: Tuple of discretized Q, R and N matrices
    """
    #  Digital Control of Dynamic Systems, Gene F. Franklin, page 379, expression 9.53:
    n_ab = a.shape[1] + b.shape[1]
    na = a.shape[0]
    m = np.eye(n_ab)
    m[:na, :na] = a
    m[:na, na:] = b

    cc = np.zeros((n_ab, n_ab))
    cc[:na, :na] = q
    cc[na:, na:] = r
    if n is not None:
        cc[:na, na:] = n  
        cc[na:, :na] = n.T

    im = quad5(lambda t: m.T @ cc @ m, dt)
    qd = im[:na, :na]
    rd = im[na:, na:]
    nd_1 = im[:na, na:]
    nd_2 = im[na:, :na]
    return (qd + qd.T) / 2, (rd + rd.T) / 2, (nd_1 + nd_2.T) / 2


def refi(a, b, hr, k):
    """! Computes command input matrices of an LQR optimal control scheme with feedforward and
    steady state input estimation. The number of contol inputs (u) must be equal to the
    number of outputs (y), i.e., if shape of b = n x m, then shape of hr must be m x n.
    @param a: State space representation matrix, A, of a linearized version of the system of
    interest
    @param b: State space representation matrix, B, of a linearized version of the system of
    interest
    @param hr: Accesible substate distribution matrix
    @param k: Gain matrix
    @returns: Tuple of reference input to reference state matrix, input reference to steady state
    input matrix and the matrix that condenses the latter
    """

    #  Digital Control of Dynamic Systems, Gene F. Franklin, page 313, expression 8.73:
    eye_a = np.eye(a.shape[0])
    m, n = hr.shape
    a = np.block([[a - eye_a, b], [hr, np.zeros((m, m))]])
    b = np.block([[np.zeros((n, m))], [np.eye(m)]])
    nn = np.linalg.solve(a, b)

    nx = nn[:n, :]
    nu = nn[n:n + m, :]
    nb = nu + k @ nx

    return nx, nu, nb