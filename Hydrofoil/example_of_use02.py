import time
import numpy as np
import src.core.pyairy as pyairy

from src.core.Foilcart import Foilcart


if __name__ == "__main__":

    ###############################################################################################
    # Get an instance of Foilcart class
    ###############################################################################################
    print("Initializing Foilcart class...")
    foilcart = Foilcart()
    print("Foilcart class has been instantiated")

    ###############################################################################################
    # Set boat state parameters
    ###############################################################################################
    # Define initial values of the variables at time zero, before simulation starts
    # ´attitude´ contains the position of the COG (x, y, z) in ground/horizon co-ordinates and
    #  the Euler angles of the boat (yaw, pitch, roll)
    attitude = np.array([0.0, 0.0, -0.8, 0.0, 0.0, 0.0])
    # For an initial attitude we can calculate the transformation that will turn any vector in
    #  ground co-ordinates to the boat reference frame
    ground2boat = foilcart.get_ground2boat_matrix(attitude)
    # We initialize the boat speed. We set it in ground frame (which is easy to understand) and
    #  turn it into boat co-ordinates
    boat_speed = ground2boat.dot(np.array([10, 0.0, 0.0]))
    # Boat linear and angular speeds are merged into ´attitude_dot´, which is the
    #  concatenation of the (x, y, z) components of the linear speed and the (x, y, z) components
    #  of the angular speed.
    # Note that, although the name may be misleading, ´attitude_dot´ is not the derivative
    #  w.r.t. time of ´attitude´
    attitude_dot = np.array([boat_speed[0], boat_speed[1], boat_speed[2], 0.0, 0.0, 0.0])
    # We now set the angles of the wings and the flaps
    wing_angles = {"main wing": 0, "elevator": 0}
    flap_angles = {"main wing": {"stbd": 0, "port": 0}, "strut": {"upper": 0, "lower": 0}}
    # The total thrust provided by the propellers. If a scalar is inputted, it is the value of the
    #  total thrust provided by the two propellers, in Newtons. If None, total thrust is
    #  calculated so the forces over the boat along the ground x-axis are in equilibrium.
    total_thrust = None
    # The additional thrust of the starboard propeller compared to the port one. If `T` is the
    #  total thrust of the boat, the starboard propeller provides
    #  `T/2 * (1 + differential_thrust)´ while the port propeller provides
    #  `T/2 * (1 - differential_thrust)´
    differential_thrust = 0.
    # The state and input vector are generated using the previously defined values
    state_vector = foilcart.get_state(attitude, attitude_dot)
    input_vector = np.array([wing_angles["main wing"],
                             flap_angles["main wing"]["stbd"],
                             flap_angles["main wing"]["port"],
                             wing_angles["elevator"],
                             flap_angles["strut"]["lower"],
                             differential_thrust])

    ###############################################################################################
    # Set environmental parameters
    ###############################################################################################
    # We also set the speeds of the wind and the water (currents) w.r.t. the ground reference frame
    ground_water_speed = np.array([0, 0, 0])
    ground_wind_speed = np.array([0, 0, 0])
    # Definition of the modes od the wave to be used
    wave_direction = np.array([-1 / np.sqrt(2), -1 / np.sqrt(2), 0])
    wave_parameters = {"distribution": "complete",
                       "crest trough height [m]": np.array([0.05, 0.0]) * 0.8,
                       "wave number [m]": np.array([2 * np.pi / 40 * wave_direction,
                                                    2 * np.pi / 4.1 * wave_direction]),
                       "phase shift [rad]": np.array([0, 0]),
                       "depth [m]": None}
    # wave_parameters = None

    ###############################################################################################
    # Set control parameters
    ###############################################################################################
    # Set time step
    dt = 0.01
    # Choose the indices of which degrees of freedom (dofs) must be calculated during
    #  simulation. Indices refer to their positions in the state vector, e.g., 2 is height of CoG
    #  in the reference frame, 4 is pitch, 8 is the vertical speed in boat frame (in general, not
    #  the derivative of the second component of the state vector!!), and 10 is the angular speed
    #  in the y, boat frame axis.
    dofs = [2, 4, 5, 8, 9, 10]
    # Set variables that must be kept fixed during simulation
    fixed_dofs = [1, 3, 7, 11]
    # All the variables we can set to control the boat are enumerated here, although not all of
    #  them may be used
    input_labels = ["main wing", "flap stbd", "flap port", "elevator",
                    "rudder", "differential thrust"]
    foilcart.set_inputs_for_this_case(input_labels)
    # Indices of the input vector that will be modified by the control
    controlled_inputs = [0, 3, 4]
    ni = len(controlled_inputs)
    labels_of_controlled_inputs = [input_labels[pp] for pp in controlled_inputs]
    foilcart.set_inputs_for_this_case(labels_of_controlled_inputs)
    # Q and R matrices of the LQRI scheme
    q = [0.1 * 10, np.deg2rad(2) * 10, np.deg2rad(2) * 10,  # I
         0.1, np.deg2rad(3), np.deg2rad(3),  # P
         0.1 * 3, np.deg2rad(3) * 3, np.deg2rad(3) * 3]  # D
    r = np.deg2rad([1]) * 1
    # Variables to monitor and how often they must be showed
    # monitoring_parameters = {"dt": 0.2,
    #                          "dt dynamics": 0.1,
    #                          "variables": [[["actual z"],
    #                                         ["actual pitch"],
    #                                         ["strut wave height"],
    #                                         ["total x force", "total y force", "total z force"]],
    #                                        [["actual w"],
    #                                         ["actual q"],
    #                                         ["strut wave vertical speed"],
    #                                         ["total x moment", "total y moment", "total z moment"]],
    #                                        [["commanded main wing", "actual main wing"],
    #                                         ["strut span"]]]}

    monitoring_parameters = {"dt plot": 2.0,
                             "dt dynamics": 0.1,
                             "variables": [[["actual z"],
                                            ["actual pitch"],
                                            ["actual roll"],
                                            ["total x force", "total y force", "total z force"]],
                                           [["strut wave height"],
                                            ["strut wave vertical speed"],
                                            ["strut span"],
                                            ["total x moment", "total y moment", "total z moment"]],
                                           [["commanded main wing", "actual main wing",
                                             "main wing effective aoa center"],
                                            ["commanded elevator", "actual elevator",
                                             "elevator effective aoa center"],
                                            ["commanded rudder", "actual rudder",
                                             "strut effective aoa center"]]]}

    ###############################################################################################
    # Set actuator parameters
    ###############################################################################################
    actuator_parameters = {"max speed": 180e6, "max acceleration": 180e6, "overshoot": 0.05,
                           "steady delay": 1}

    ###############################################################################################
    # Calculate stationary solution
    ###############################################################################################
    # We are going to find an initial point for the simulation. We will calculate a stationary
    #  solution for the steady case (no wave, nor any  other unsteady perturbations) and use it
    #  as the initial values for time zero.
    # We first set which equations must be solved for this case. All the other equations or
    #  relations will be ignored, but it is recommended that they are zero for our initial guess.
    equations_to_solve = ["force z", "moment y", "moment x"]
    # We then define which input variables will be modified to get the solution. In general,
    #  the number of equations to solve must be equal to the variables to solve for.
    variables_to_find = ["main wing", "elevator", "rudder"]
    # # Set inputs to control for this case
    # stationary_controlled_inputs = [0, 3]
    # labels_of_controlled_inputs = [input_labels[pp] for pp in stationary_controlled_inputs]
    # foilcart.set_inputs_for_this_case(labels_of_controlled_inputs)
    # The stationary solution is found
    print("Calculating stationary solution...")
    st = time.process_time()
    results = foilcart.find_stationary_solution(state_vector, input_vector[controlled_inputs],
                                                equations_to_solve, variables_to_find,
                                                ground_wind_speed=ground_wind_speed,
                                                ground_water_speed=ground_water_speed)
    et = time.process_time()
    stationary_movement_time = et - st
    print('Stationary solution found in :', time.strftime("%H:%M:%S",
                                                          time.gmtime(stationary_movement_time)))

    # labels_of_controlled_inputs = [input_labels[pp] for pp in controlled_inputs]
    # foilcart.set_inputs_for_this_case(labels_of_controlled_inputs)

    ###############################################################################################
    # Initialize control scheme
    ###############################################################################################
    # Set solution of the search of the stationary case as new reference state
    state_vector = results[0].copy()
    input_vector = results[1].copy()

    # Calculate linear approximation of the equations
    f0, a, b = foilcart.linearize(state_vector=state_vector, input_vector=input_vector,
                                  ground_water_speed=ground_water_speed,
                                  ground_wind_speed=ground_wind_speed)

    print(f"Residuals of the equations = {f0:}")
    print("Initial state = {}".format(state_vector))
    print("Initial input = {}".format(input_vector))

    # Calculate discrete version of the matrices of the linearized equations
    ad, bd = pyairy.discretize(a[dofs, :][:, dofs], b[dofs, :], dt=dt)

    # Calculate LQRI control scheme
    k = pyairy.dlqri(ad, bd, q=q, r=r,
                     ni=ni, dt=dt, bryson=True)[0]

    # Set control scheme
    if len(dofs) % 2 == 0:
        m = int(len(dofs) / 2)
    else:
        raise ValueError("Number of degrees of freedom must be an even number")
    settings = pyairy.Settings(k=k, dt=dt, sp=state_vector[dofs[:m]],
                               u0=input_vector,
                               ulims=[-10, 10],
                               mts=np.array([pyairy.kts2mps(10), pyairy.kts2mps(20)]),
                               dofs=dofs,
                               integrator_idxs=[pp for pp in range(ni)])
    foilcart_control = pyairy.Control(settings)

    ###############################################################################################
    # Perform simulation
    ###############################################################################################

    initial_state_vector = state_vector.copy()
    # initial_state_vector[2] += 0.05
    initial_state_vector[4] += 0.05
    # initial_state_vector[5] += 0.05

    print("Starting simulation...")
    st = time.process_time()
    results = foilcart.simulate(simulation_time=10.0, initial_state=initial_state_vector,
                                initial_inputs=input_vector,
                                control=foilcart_control, fixed_dofs=fixed_dofs,
                                monitoring_parameters=monitoring_parameters,
                                actuator_parameters=actuator_parameters,
                                wave_parameters=wave_parameters,
                                noise_parameters=None, results_path='wave_test_00.pickle',
                                do_not_recalculate_preproccesed_prantl=True)
    et = time.process_time()
    stationary_movement_time = et - st
    print('Simulation run in :', time.strftime("%H:%M:%S", time.gmtime(stationary_movement_time)))