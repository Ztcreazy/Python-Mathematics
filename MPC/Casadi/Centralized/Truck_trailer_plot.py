import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def transform_points(points, T):
    """Transform the points using the transformation matrix T."""
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    transformed_points = T @ points_homogeneous.T
    return transformed_points[:2, :].T

def get_rectangle_vertices(L, W):
    """Get rectangle vertices given length L and width W."""
    return np.array([
        [L/2, W/2],
        [L/2, -W/2],
        [-L/2, -W/2],
        [-L/2, W/2],
        [L/2, W/2]
    ])

def expand_to_full_state(q_partial, params):
    """Expand the partial state to the full state of the truck-trailer."""
    L1 = params['L1']
    M = params['M']
    L2 = params['L2']

    x2, y2, theta2, beta = q_partial

    theta1 = theta2 + beta
    x1 = x2 + M * np.cos(theta1) + L2 * np.cos(theta2)
    y1 = y2 + M * np.sin(theta1) + L2 * np.sin(theta2)

    # Return as a 1D array
    return np.array([x1, y1, theta1, x2, y2, theta2, beta])

def animate_tractor_trailer(ax, t_out, q_out, alpha, params):
    """Animate the tractor-trailer."""
    W1, L1 = params['W1'], params['L1']
    W2, L2 = params['W2'], params['L2']
    W3, L3 = params['Wwheel'], params['Lwheel']

    truck_points = transform_points(get_rectangle_vertices(L1, W1), np.array([[1, 0, L1/2], [0, 1, 0], [0, 0, 1]]))
    trailer_points = transform_points(get_rectangle_vertices(L2, W2), np.array([[1, 0, L2/2], [0, 1, 0], [0, 0, 1]]))
    wheel_points = get_rectangle_vertices(L3, W3)

    h_truck, = ax.plot([], [], color=[1, 0.5, 0], linewidth=1)
    h_trailer, = ax.plot([], [], color='m', linewidth=1)
    h_wheel_fl = ax.fill([], [], color='k')[0]
    h_wheel_fr = ax.fill([], [], color='k')[0]
    h_wheel_rl = ax.fill([], [], color='k')[0]
    h_wheel_rr = ax.fill([], [], color='k')[0]
    h_wheel_tl = ax.fill([], [], color='k')[0]
    h_wheel_tr = ax.fill([], [], color='k')[0]
    h_hinge, = ax.plot([], [], 'b-', linewidth=2)

    for i in range(len(t_out)):
        x2, y2, theta2, beta = q_out[i]
        q_partial = np.array([x2, y2, theta2, beta])
        q_full = expand_to_full_state(q_partial, params)

        # Correctly unpack q_full
        x1, y1, theta1 = q_full[0], q_full[1], q_full[2]

        # Draw Truck
        T_truck = np.array([
            [np.cos(theta1), -np.sin(theta1), x1],
            [np.sin(theta1), np.cos(theta1), y1],
            [0, 0, 1]
        ])
        truck_transformed = transform_points(truck_points, T_truck)
        h_truck.set_data(truck_transformed[:, 0], truck_transformed[:, 1])

        # Draw Trailer
        T_trailer = np.array([
            [np.cos(theta2), -np.sin(theta2), x2],
            [np.sin(theta2), np.cos(theta2), y2],
            [0, 0, 1]
        ])
        trailer_transformed = transform_points(trailer_points, T_trailer)
        h_trailer.set_data(trailer_transformed[:, 0], trailer_transformed[:, 1])

        # Draw wheels and hinge, similar process as above
        # Wheel transformations would go here...

        # Draw Hinge (Line connecting the truck and trailer)
        hinge_points = np.array([[0, 0], [-params['M'], 0]])
        hinge_transformed = transform_points(hinge_points, T_truck)
        h_hinge.set_data(hinge_transformed[:, 0], hinge_transformed[:, 1])

        # Make all elements visible
        h_truck.set_visible(True)
        h_trailer.set_visible(True)
        h_hinge.set_visible(True)
        h_wheel_fl.set_visible(True)
        h_wheel_fr.set_visible(True)
        h_wheel_rl.set_visible(True)
        h_wheel_rr.set_visible(True)
        h_wheel_tl.set_visible(True)
        h_wheel_tr.set_visible(True)

        # Pause for animation
        plt.pause(0.3)

    plt.show()

# Example usage:
# params = {
#     'W1': 4.0, 'L1': 8.0, 'W2': 3.0, 'L2': 12.0, 'Wwheel': 0.4, 'Lwheel': 1, 'M': 2
# }

# fig, ax = plt.subplots()
# ax.set_xlim(-50, 50)
# ax.set_ylim(-30, 30)
# ax.set_xlabel('x')
# ax.set_ylabel('y')

# # Example call to animate the tractor-trailer:
# t_out = np.linspace(0, 10, 100)
# q_out = np.random.rand(100, 4)  # Example data, replace with actual simulation results
# alpha = np.random.rand(100)  # Example steering angle data

# animate_tractor_trailer(ax, t_out, q_out, alpha, params)


