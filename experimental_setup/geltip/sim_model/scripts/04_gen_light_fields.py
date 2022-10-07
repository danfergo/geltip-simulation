import math
import os
from math import pi, sin, cos

import numpy as np
from pygeodesic import geodesic

from trimesh.exchange.load import load
from trimesh.proximity import ProximityQuery
import trimesh.intersections as intersections

import potpourri3d as pp3d

from experimental_setup.geltip.sim_model.scripts.utils.camera import circle_mask


# (start) the "planes" method and geometry utils

def dist(x, y):
    return math.sqrt(np.sum((x - y) ** 2))


# the vector norm
def norm(x):
    return math.sqrt(np.sum(x ** 2))


# the farthest vertex of a segment/edge w.r.t a pt
def closest(edge, pt):
    return edge[0] if dist(edge[0], pt) < dist(edge[1], pt) else edge[1]


# the farthest vertex of a segment/edge w.r.t a pt
def farthest(edge, pt):
    return edge[1] if dist(edge[0], pt) < dist(edge[1], pt) else edge[0]


# computes the index of the path whose closest point is nearest the pt
def closest_of(path, pt):
    return int(np.argmin(np.array([dist(closest(e, pt), pt) for e in path])))


# given a set of line segments, computes the path from source to target, following the direction of v
def sort_path(path, source, v, target, ahead=0):
    a_min_target = closest_of(path, target)
    min_dist = dist(closest(path[a_min_target], target), target)
    a_min_source = closest_of(path, source)

    found_target = False
    ahead_target = 0
    remaining = [e for e in path]
    new_path = []

    # picking the first pivot
    # from the vertexes of the closest edge
    # based on the an angle between v and (source, pX)
    p0 = path[a_min_source][0]
    p1 = path[a_min_source][1]
    cos0 = np.dot(v, p0 - source) / norm(v) * norm(p0 - source)
    cos1 = np.dot(v, p1 - source) / norm(v) * norm(p1 - source)
    pivot = path[a_min_source][0 if cos0 > cos1 else 1]
    new_path.append(remaining.pop(a_min_source))

    while len(remaining) > 0:
        a_current_edge = closest_of(remaining, pivot)
        current_dist = dist(closest(remaining[a_current_edge], target), target)

        head = farthest(remaining[a_current_edge], pivot)
        tail = closest(remaining[a_current_edge], pivot)

        new_path.append((tail, head))
        remaining.pop(a_current_edge)

        pivot = head

        if current_dist <= min_dist:
            found_target = True

        if found_target:
            if ahead_target >= ahead:
                break

            ahead_target += 1

    return new_path


def light_ray_vector(mesh, led, pt, orientation):
    led_pt_v = pt - led
    # dot = np.dot(orientation, led_pt_v) / abs(norm(led_pt_v) * norm(orientation))
    # intersection mesh with plane
    plane_v = np.cross(orientation, led_pt_v)
    path = intersections.mesh_plane(mesh, plane_v, led)
    # abs(dot) < 0.001 or
    if len(path) == 0:
        print('dot: ', abs(0), 'len: ', len(path))
        return led_pt_v / norm(led_pt_v), (None, led + orientation, (None, None))

    sub_path = sort_path(path, led, orientation, pt)
    sub_path.reverse()

    # light vector
    lv = sub_path[0][1] - sub_path[0][0]
    lv /= norm(lv)

    return lv, (sub_path, led + orientation, sub_path[0])


def planes_light_field(mesh,
                       source_pos,
                       cloud_map,
                       n_rays=3,
                       starting_angle=pi / 2 - pi / 3,
                       end_angle=pi / 2 + pi / 3,
                       log_progress=True):
    # all_viz = []
    # proximity_query = proximity_query or ProximityQuery(mesh)
    #
    # computing the vector
    # p = ProximityQuery(mesh)
    # p_dist, p_idx = p.vertex(pt)
    # n = (mesh.vertex_normals[p_idx] / norm(mesh.vertex_normals[p_idx])) * 0.01
    # lrv, viz = light_ray_vector(mesh, led, pt, n)
    # aggregate = lrv
    # all_viz.append(viz)
    # computing the vector based on multiple plane-intersections
    # if method == 'planes':

    field = np.zeros(cloud_map.shape)

    for i in range(cloud_map.shape[0]):
        for j in range(cloud_map.shape[1]):

            if log_progress and j == (cloud_map.shape[1] - 1):
                progress = ((i * cloud_map.shape[1] + j) / (cloud_map.shape[0] * cloud_map.shape[1]))
                print('progress... ' + str(round(progress * 100, 2)) + '%')

            pt = cloud_map[i, j]

            delta_angle = (end_angle - starting_angle) / (n_rays - 1)
            vectors = []  # np.array([0.0, 0.0, 0.0])
            for k in range(0, n_rays):
                angle = starting_angle + (k * delta_angle)
                orientation = np.array([0.0, 0.005 * cos(angle), 0.005 * sin(angle)])
                lrv, viz = light_ray_vector(mesh, source_pos, pt, orientation)
                vectors.append(lrv)
            lm = np.mean(np.array(vectors), axis=0)
            lm /= norm(lm)
            field[i, j] = lm

            # all_viz.append(viz)
    #     return vectors, None, None
    return field


# (end) the "planes" method and geometry utils


def compute_light_field(mesh,
                        source_pos,
                        cloud_map,
                        method='geodesic',
                        log_progress=True):
    m = circle_mask(np.array([cloud_map.shape[1], cloud_map.shape[0]]))

    if method == 'linear':

        field = np.zeros(cloud_map.shape)

        for i in range(cloud_map.shape[0]):
            for j in range(cloud_map.shape[1]):

                if log_progress and j == (cloud_map.shape[1] - 1):
                    progress = ((i * cloud_map.shape[1] + j) / (cloud_map.shape[0] * cloud_map.shape[1]))
                    print('progress... ' + str(round(progress * 100, 2)) + '%')

                if m[i, j] > 0.5:
                    lm = cloud_map[i, j] - source_pos
                    lm /= norm(lm)
                    field[i, j] = lm
        return field

    elif method == 'planes':
        return planes_light_field(mesh, source_pos, cloud_map)
    elif method == 'geodesic':
        proximity_query = ProximityQuery(mesh)
        _, sourceIndex = proximity_query.vertex(source_pos)
        field = np.zeros(cloud_map.shape)
        M = m.flatten()
        P = cloud_map.reshape([-1, 3])

        # Mf_idx = np.where(M > 0.5)
        # Pf = np.array([pp for pp in range(len(P)) if M[pp] > 0.5])
        # print(Mf_idx)
        # print(Pf.shape, Mf_idx.shape)
        # print(np.where(P > 0.5))

        for i in range(cloud_map.shape[0]):
            for j in range(cloud_map.shape[1]):
                if log_progress and j == (cloud_map.shape[1] - 1):
                    progress = ((i * cloud_map.shape[1] + j) / (cloud_map.shape[0] * cloud_map.shape[1]))
                    print('progress... ' + str(round(progress * 100, 2)) + '%')

                if m[i, j] > 0.5:
                    pt = cloud_map[i, j]

                    __, targetIndex = proximity_query.vertex(pt)

                    # Compute the geodesic distance and the path
                    points = mesh.vertices
                    faces = mesh.faces
                    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)
                    distance, path = geoalg.geodesicDistance(sourceIndex, targetIndex)

                    if len(path) < 2:
                        return [0, 0, 0], path, 'FAILED'
                        # return path[0], path, False

                    lrv = [path[0][i] - path[1][i] for i in range(3)]
                    field[i, j] = lrv

                    # aggregate = lrv
                    # all_viz.append((path, None, None))

        return field

    elif method == 'logmap':
        proximity_query = ProximityQuery(mesh)
        P = cloud_map.reshape([-1, 3])  # a Nx3 numpy array of points
        solver = pp3d.MeshVectorHeatSolver(mesh.vertices, mesh.faces)
        # solver = pp3d.PointCloudHeatSolver(P)

        basisX, basisY, basisN = solver.get_tangent_frames()

        _, sourceV = proximity_query.vertex(source_pos)

        logmap = solver.compute_log_map(sourceV)
        logmap3D = logmap[:, 0, np.newaxis] * basisX + logmap[:, 1, np.newaxis] * basisY

        logmap3D = np.array([logmap3D[proximity_query.vertex(P[i])[1]] for i in range(len(P))])
        # print(proximity_query.vertex(P[0]))
        # print(logmap.shape)
        # print(mesh.vertices.shape)

        # dx = normalize_vectors(derivative(cloud_map, 'x'))
        # dy = normalize_vectors(derivative(cloud_map, 'y'))
        # N = np.cross(dx, dy).reshape((-1, 3))

        # v = np.cross(N, logmap3D, axis=1)
        v = logmap3D
        return v.reshape((cloud_map.shape[0], cloud_map.shape[1], 3))


assets_path = os.path.dirname(os.path.abspath(__file__)) + '/../assets/'

method = 'planes'
cloud_size = (160, 120)
# cloud_size = (640, 480)
# fields = (0, 1)
# fields = (1, 2)
fields = (2, 3)
# fields = (0, 3)

prefix = str(cloud_size[0]) + 'x' + str(cloud_size[1])
cloud = np.load(assets_path + '/' + prefix + '_ref_cloud.npy')
mesh = load(assets_path + '/' + prefix + '_aligned_mesh_voxel_e-6.stl', 'stl', force='mesh')

# new_vertices, new_faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, 0.01)
# hr_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

# path_solver = pp3d.EdgeFlipGeodesicSolver(mesh.vertices, mesh.faces)

scale = 0.00105
z_trans = -0.015
led_radius = 12
leds = [np.array([cos(a * (2 * pi / 3)) * scale * led_radius, sin(a * (2 * pi / 3)) * scale * led_radius, z_trans]) for
        a in range(3)]

# leds_colors = ['red', 'green', 'blue']
# spheres = [sphere(l, 'green') for l in leds]

cloud_map = cloud.reshape((cloud_size[1], cloud_size[0], 3))

for l in range(*fields):
    print('generating light field ' + str(l))
    field = compute_light_field(mesh,
                                leds[l],
                                cloud_map,
                                method)
    np.save(assets_path + '/' + method + '_' + prefix + '_field_' + str(l) + '.npy', field)
