from typing import List, Any
import numpy as np
import math, random
from sklearn.cluster import OPTICS
from scipy.spatial import ConvexHull, Delaunay

def find_closest_points(cluster_files, cluster, points, percent=10) -> List[Any]:

    n = int(len(cluster) * percent/100)

    # Calculate the convex hull of the points using the ConvexHull function from scipy
    hull = ConvexHull(points)
    tri = Delaunay(points)
    
    # Get the vertices and edges of the convex hull
    vertices = hull.vertices
    edges = [(points[vertices[i-1]], points[vertices[i]]) for i in range(len(vertices))]
    
    # Initialize a list to store the closest points
    closest_points = []
    
    # Iterate through the points in the cluster
    for p, file in zip(cluster, cluster_files):
        min_distance = float('inf')

        if tri.find_simplex(p) >= 0:
            closest_points.append((file, 0))
            continue
        
        # Iterate through the edges of the convex hull
        for a, b in edges:
            # Calculate the distance from p to the edge
            distance = distance_to_line(p, a, b)
            
            # Update the minimum distance and the closest point if the distance to the edge is smaller
            min_distance = min(min_distance, distance)
        
        # Add the closest point to the list of closest points
        closest_points.append((file, min_distance))
    
    # Sort the list of closest points by distance from the boundary of the convex hull
    closest_points.sort(key=lambda x: x[1])
    
    # Return the first n closest points
    return [x[0] for x in closest_points[:n]]

def distance_to_line(p, a, b):
    # Calculate the distance from p to the line defined by a and b
    distance = abs((b[1] - a[1])*p[0] - (b[0] - a[0])*p[1] + b[0]*a[1] - b[1]*a[0]) / math.sqrt((b[1] - a[1])**2 + (b[0] - a[0])**2)
    
    # Calculate the projection of p onto the line defined by a and b
    t = ((p[0] - a[0])*(b[0] - a[0]) + (p[1] - a[1])*(b[1] - a[1])) / ((b[0] - a[0])**2 + (b[1] - a[1])**2)
    
    # Check if the projection of p is outside the line segment
    if t < 0:
        # Return the distance from p to a
        return math.sqrt((p[0] - a[0])**2 + (p[1] - a[1])**2)
    elif t > 1:
        # Return the distance from p to b
        return math.sqrt((p[0] - b[0])**2 + (p[1] - b[1])**2)
    else:
        # Return the distance from p to the line
        return distance

def find_centroid(points):
    # Find the mean of the points along each dimension
    points = np.array(points)
    centroid = []
    for i in range(points.shape[1]):
        centroid.append(points[:, i].mean())

    return np.array(centroid)

def min_max_convex_hull(points1, points2):
    # Find the minimum and maximum x and y values among the two sets of points
    min_x = min(min(x for x, _ in points1), min(x for x, _ in points2))
    max_x = max(max(x for x, _ in points1), max(x for x, _ in points2))
    min_y = min(min(y for _, y in points1), min(y for _, y in points2))
    max_y = max(max(y for _, y in points1), max(y for _, y in points2))
    
    # Construct the min-max convex hull from the min-max points
    convex_hull = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
    
    return convex_hull

def align_centroids(centroid1, centroid2, inverse=False):
    # Unpack the centroids
    x1, y1 = centroid1
    x2, y2 = centroid2
    
    x_mid, y_mid = centroid1 - centroid2 / 2

    x1 = x1 - x_mid
    x2 = x2 - x_mid
    y1 = y1 - y_mid
    y2 = y2 - y_mid

    if inverse:
        x_mid = -x_mid
        y_mid = -y_mid

    translate = np.array([[1, 0, -x_mid],
                         [0, 1, -y_mid],
                         [0, 0, 1]])

    # Calculate the angle between the vector connecting the centroids and the x-axis
    angle = math.atan2(y2 - y1, x2 - x1)
    
    # Calculate the rotation matrix that aligns the vector with the x-axis
    rot = np.array([[math.cos(-angle), -math.sin(-angle), 0],
                    [math.sin(-angle), math.cos(-angle), 0],
                    [0, 0, 1]])
    
    # If the inverse flag is set, calculate the inverse of the rotation matrix
    if inverse:
        rot = np.linalg.inv(rot)
    
    # Define a function that uses the rotation matrix to transform a set of points
    def transform(points):
        if not inverse:
            rotated = [(rot @ translate @ np.array([x, y, 1]))[:2] for x, y in points]
        else:
            rotated = [(translate @ rot @ np.array([x, y, 1]))[:2] for x, y in points]
        return rotated
    
    return transform

def finding_narrow_min_max(points, x_value):
    # Initialize the nearest positive and negative points to None
    nearest_positive = None
    nearest_negative = None
    
    # Iterate through the points
    for point in points:
        x, y = point
        
        # If the x value of the point is closest to v so far and y is positive, update the nearest positive point
        if y >= 0 and (nearest_positive is None or abs(x - x_value) < abs(nearest_positive[0] - x_value)):
            nearest_positive = point
        
        # If the x value of the point is closest to v so far and y is negative, update the nearest negative point
        if y < 0 and (nearest_negative is None or abs(x - x_value) < abs(nearest_negative[0] - x_value)):
            nearest_negative = point
    
    return nearest_positive, nearest_negative

def midpoint_centroids(points1, points2):
    # Calculate the centroids of the two sets of points
    x1 = sum(x for x, _ in points1) / len(points1)
    y1 = sum(y for _, y in points1) / len(points1)
    x2 = sum(x for x, _ in points2) / len(points2)
    y2 = sum(y for _, y in points2) / len(points2)
    
    # Calculate the midpoint of the centroids
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2
    
    return (x_mid, y_mid)

def get_narrow_bounds(points1, points2):
    # Assumes the points are aligned with x axis

    x_target, _ = midpoint_centroids(points1, points2)

    np1, nn1 = finding_narrow_min_max(points1, x_target)
    np2, nn2 = finding_narrow_min_max(points2, x_target)

    bounds = [np1, nn1, nn2, np2]
    return bounds
    
def transform_spaces(points1, points2):
    # Calculate the transformation function

    centroid1 = find_centroid(points1)
    centroid2 = find_centroid(points2)

    transform = align_centroids(centroid1, centroid2)

    # Transform the points using the transformation function
    points1_aligned = transform(points1)
    points2_aligned = transform(points2)

    # border = min_max_convex_hull(points1_aligned, points2_aligned)
    border = get_narrow_bounds(points1_aligned, points2_aligned)

    # To perform the inverse transformation, call the transformation function with the inverse flag set to True
    transform_inv = align_centroids(centroid1, centroid2, inverse=True)
    border_to_draw = transform_inv(border)

    return border_to_draw

def find_points_near_border(clusters, points) -> List[List[Any]]:

    closest_for_each = []

    border_points = transform_spaces(points[0], points[1])

    for c, p in zip(clusters, points):
        closest_for_each.append(find_closest_points(c, p, border_points, 15))

    return closest_for_each

def get_border_points(embeddings, names):
    clf = OPTICS(min_samples=len(embeddings)//5)
    predictions = clf.fit_predict(embeddings)
    clusters = [[] for _ in set(predictions)]

    for p, v, name in zip(embeddings, predictions, names):
        clusters[v].append(p)

    hull_border = transform_spaces(clusters[0], clusters[1])
    
    near_border = find_points_near_border(clusters)
    return hull_border, near_border

def test():
    # Now broken
    x = [random.uniform(-1, -0.2) for _ in range(100)] + [random.uniform(0.2, 1) for _ in range(100)]
    y = [random.uniform(-1, -0.2) for _ in range(100)] + [random.uniform(0.2, 1) for _ in range(100)]
    embeddings = list(zip(x, y))
    try:
        hull_border, near_border = get_border_points(embeddings)
        print(hull_border)
        print(near_border)
    except:
        print("Failure to run gap detection")