import numpy as np
import matplotlib.pyplot as plt

class Rectangle:
    def __init__(self, center, width, height, angle=0):
        self.center = np.array(center)
        self.width = width
        self.height = height
        self.angle = angle  # Angle in radians
    
    def vertices(self):
        hw, hh = self.width / 2, self.height / 2
        corners = np.array([
            [hw, hh],
            [hw, -hh],
            [-hw, -hh],
            [-hw, hh]
        ])
        
        rotation_matrix = np.array([
            [np.cos(self.angle), -np.sin(self.angle)],
            [np.sin(self.angle), np.cos(self.angle)]
        ])
        
        return [self.center + rotation_matrix @ corner for corner in corners]
    
    def furthest_point(self, direction):
        direction = np.array(direction)
        vertices = self.vertices()
        max_dot = -np.inf
        furthest = None
        for v in vertices:
            dot_product = np.dot(v, direction)
            if dot_product > max_dot:
                max_dot = dot_product
                furthest = v
        return furthest

class GJK:
    def __init__(self, rect1, rect2):
        self.rect1 = rect1
        self.rect2 = rect2

    def support(self, direction):
        p1 = self.rect1.furthest_point(direction)
        p2 = self.rect2.furthest_point(-direction)
        return p1 - p2

    def gjk_distance(self):
        direction = np.array([1, 0])  # Initial arbitrary direction
        simplex = [self.support(direction)]
        direction = -simplex[0]

        while True:
            new_point = self.support(direction)
            if np.dot(new_point, direction) <= 0:
                return self.distance_to_origin(simplex)

            simplex.append(new_point)
            if self.contains_origin(simplex, direction):
                return 0  # Collision detected

    def distance_to_origin(self, simplex):
        if len(simplex) == 1:
            return np.linalg.norm(simplex[0]), simplex[0], np.zeros(2)
        elif len(simplex) == 2:
            return self.distance_to_line_segment(simplex[0], simplex[1])
        elif len(simplex) == 3:
            return self.distance_to_triangle(simplex[0], simplex[1], simplex[2])

    def distance_to_line_segment(self, A, B):
        AB = B - A
        AO = -A
        AB_AB = np.dot(AB, AB)
        AO_AB = np.dot(AO, AB)
        t = np.clip(AO_AB / AB_AB, 0, 1)
        closest_point = A + t * AB
        return np.linalg.norm(closest_point), closest_point, np.zeros(2)

    def distance_to_triangle(self, A, B, C):
        # Not needed for this application
        pass

    def contains_origin(self, simplex, direction):
        if len(simplex) == 1:
            return False
        elif len(simplex) == 2:
            A, B = simplex
            AB = B - A
            AO = -A
            AB_perp = np.array([-AB[1], AB[0]])
            if np.dot(AB_perp, AO) > 0:
                direction[:] = AB_perp
            else:
                direction[:] = -AB_perp
            return False
        elif len(simplex) == 3:
            A, B, C = simplex
            AB = B - A
            AC = C - A
            AO = -A
            AB_perp = np.array([-AB[1], AB[0]])
            AC_perp = np.array([-AC[1], AC[0]])
            if np.dot(AB_perp, AO) > 0:
                simplex[:] = [A, B]
                direction[:] = AB_perp
            elif np.dot(AC_perp, AO) > 0:
                simplex[:] = [A, C]
                direction[:] = AC_perp
            else:
                return True
            return False

def plot_rectangles(rect1, rect2, min_distance, closest_point, direction_vector):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    def plot_rectangle(rect, color):
        vertices = rect.vertices()
        vertices.append(vertices[0])  # Close the rectangle
        xs, ys = zip(*vertices)
        ax.plot(xs, ys, color=color)
        ax.fill(xs, ys, color=color, alpha=0.3)
        # Plot the vertices
        ax.scatter(xs[:-1], ys[:-1], color=color)

    plot_rectangle(rect1, 'blue')
    plot_rectangle(rect2, 'red')

    # Plot the line representing the minimal distance
    p1, p2 = closest_point
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--')
    ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, f'Min Distance: {min_distance:.2f}', color='green')

    # Plot the direction vector
    origin = p1 + 0.5 * direction_vector
    ax.quiver(*origin, *direction_vector, color='purple', angles='xy', scale_units='xy', scale=1)

    # Set axis limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    plt.show()

def main():
    # Define two rectangles with center coordinates, width, height, and angle (in radians)
    rect1 = Rectangle(center=[0, 0], width=4, height=2, angle=np.radians(90))
    rect2 = Rectangle(center=[5, 0], width=3, height=3, angle=np.radians(45))

    # Create GJK instance and calculate the minimum distance
    gjk = GJK(rect1, rect2)
    min_distance, p1, p2 = gjk.gjk_distance()
    closest_points = (p1, p2)
    
    # Calculate the direction vector of the minimum distance
    direction_vector = p2 - p1

    print(f"Minimum distance: {min_distance:.2f}")
    
    # Plot the rectangles and the minimum distance
    plot_rectangles(rect1, rect2, min_distance, closest_points, direction_vector)

if __name__ == "__main__":
    main()
