import math
import os
import time
from math import pi, sin, cos

import cv2
import numpy as np
from pygeodesic import geodesic
from trimesh.exchange.load import load
from trimesh.proximity import ProximityQuery
import trimesh.intersections as intersections
from trimesh.proximity import nearby_faces

import potpourri3d as pp3d

from sim_model.utils.maths import normalize_vectors, partial_derivative, proj_vectors, normals
from sim_model.utils.camera import circle_mask, get_camera_matrix, depth2cloud


# (start) the "planes" method and geometry utils


def dist(x, y):
    return math.sqrt(np.sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))


# the vector norm
def norm(x):
    return math.sqrt(np.sum(x ** 2))


# normalize a vector
def normalize(x):
    return x / norm(x)


# the farthest vertex of a segment/edge w.r.t a pt
def closest(edge, pt):
    return edge[0] if dist(edge[0], pt) < dist(edge[1], pt) else edge[1]


# the farthest vertex of a segment/edge w.r.t a pt
def farthest(edge, pt):
    return edge[1] if dist(edge[0], pt) < dist(edge[1], pt) else edge[0]


# computes the index of the path whose closest point is nearest the pt
def closest_of(path, pt):
    return int(np.argmin(np.array([dist(closest(e, pt), pt) for e in path])))


def proj(u, n_):
    norm_ = norm(n_)
    if norm_ == 0:
        return 0 * u
    # print(u.dot(n))
    return u.dot(n_) * normalize(n_)


def angle2d(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.lianalg.norm(a)
    norm_b = np.linalg.norm(b)
    angle_rad = np.arccs(dot_product / (norm_a * norm_b))


# given a set of line segments, computes the path from source to target, following the direction of v
def sort_path(path, source, target, v, ahead=0):
    """
    :param path: list of line segments
    :param source:
    :param v: vector at the origin
    :param target:
    :param ahead:
    :return:
    """
    a_min_source = closest_of(path, source)
    a_min_target = closest_of(path, target)
    min_dist = dist(closest(path[a_min_target], target), target)

    found_target = False
    ahead_target = 0
    remaining = [e for e in path]
    new_path = []

    # picking the vertex from the first segment as the pivot
    # based on the an angle between v and (source, pX)
    p0 = path[a_min_source][0]
    p1 = path[a_min_source][1]
    cos0 = np.dot(v, p0 - p1) / norm(v) * norm(p0 - p1)
    cos1 = np.dot(v, p1 - p0) / norm(v) * norm(p1 - p0)
    pivot = path[a_min_source][0 if cos0 > cos1 else 1]  # higher cos means lower angle between vectors
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


def translate_vector(mesh, source, target, v, n):
    # v = target - source
    # dot = np.dot(orientation, led_pt_v) / abs(norm(led_pt_v) * norm(orientation))
    # intersection mesh with plane
    plane_v = np.cross(n, v)
    intersection_path = intersections.mesh_plane(mesh, plane_v, source)
    # abs(dot) < 0.001 or
    # if len(s_path) == 0:
    #     print('dot: ', abs(0), 'len: ', len(s_path))
    #     return v / norm(v), (None, source + n, (None, None))

    sub_path = sort_path(intersection_path, source, target, v)

    vi = normalize(sub_path[0][1] - sub_path[0][0])

    e = len(sub_path) - 1
    ve = normalize(sub_path[e][1] - sub_path[e][0])

    return vi, ve, sub_path


def rotation_matrix_from_vectors(vec1, vec2):
    vec1_unit = vec1 / np.linalg.norm(vec1)
    vec2_unit = vec2 / np.linalg.norm(vec2)
    axis = np.cross(vec1_unit, vec2_unit)
    angle = np.arccos(np.dot(vec1_unit, vec2_unit))
    kmat = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = (
            np.eye(3) + kmat + np.dot(kmat, kmat) * (1 - np.cos(angle))
    )
    return rotation_matrix


def angle_between(v1, v2):
    """Returns the angle between two 2D vectors in radians."""
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = dot_product / norm_product
    return np.arccos(cos_theta)

def rotate_vector(v, theta):
    """Rotates a 2D vector v by the angle theta (in radians)."""
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return np.dot(R, v)

# def planes_light_field(mesh,
#                        source_pos,
#                        cloud_map,
#                        log_progress=True):
# n_rays=3,
# starting_angle=pi / 2 - pi / 3,
# end_angle=pi / 2 + pi / 3,
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
# dx = normalize_vectors(derivative(cloud_map, 'x'))
# dy = normalize_vectors(derivative(cloud_map, 'y'))
# normals = np.cross(dx, dy)


# pt = cloud_map[10, 10]
# pt = cloud_map[100, 100]
# pt = cloud_map[0, 100]
# pt = cloud_map[119, 159]

# n = normals[i, j]
# angle between planes at origin
# delta_angle = (end_angle - starting_angle) / (n_rays - 1)
# vectors = []  # np.array([0.0, 0.0, 0.0])
# orientations = []
# arrows = [(pt, ptv), (pt, n)],
# subsample = 25

# show_field(
#     cloud_map=cloud_map,
#     field=normals,
#     field_color='red',
#     mesh=mesh,
#     paths=[s_path],
#     pts=[pt, source_pos],
#     arrows=[
#         # (source_pos, v),
#         # (source_pos, source_pos, 'blue'),
#         (source_pos, vi, 'green'),
#         (pt, ve, 'blue'),
#     ]
# )

# m = circle_mask((cloud_map.shape[1], cloud_map.shape[0]))
# if m[i, j] > 0.5 and (not subsample or (i * cloud_map.shape[1] + j) % subsample == 0):
# , arrows = [(pt, n)]

# for k in range(0, 1):
# angle = starting_angle + (k * delta_angle)

# orientation = np.array([0.0, cos(angle), sin(angle)])
# orientation = np.array([0, 0, 1])
# orientations.append(orientation)

# np.cross(n,)
# lrv, viz = light_ray_vector(mesh, source_pos, pt, source_pos)
# s_path, _, __ = viz

# vectors.append(lrv)
# for o in orientations
# paths = [s_path],
# (pt, np.array([0, 0, 1])),
# (pt, n),
# ptv = pt - source_pos

# lm = np.mean(np.array(vectors), axis=0)
# lm /= norm(lm)
# field[i, j] = lm

# all_viz.append(viz)
#     return vectors, None, None
# return field


# (end) the "planes" method and geometry utils


def closest_vertex(m, face, p):
    i = np.argmin([dist(m.vertices[vi], p) for vi in range(len(face))])
    return m.vertices[face[i]]


def closest_faces(m, P):
    nb_faces = nearby_faces(m, P)

    def closest_face(p, nb_faces_p):
        i = np.argmin([dist(closest_vertex(m, m.faces[nb_faces_p[i]], p), p) for i in range(len(nb_faces_p))])
        return nb_faces_p[i]

    return [closest_face(P[i], nb_faces[i]) for i in range(len(nb_faces))]


def map_field_vertices2points(field, m, P):
    # proximity_query = ProximityQuery(m)
    # return np.array([field[proximity_query.vertex(P[i])[1]] for i in range(len(P))])

    # cl_faces = closest_faces(m, P)
    nb_faces = nearby_faces(m, P)

    def interpolate(faces, point):
        nb_vertices = [vi for face in faces for vi in face]
        return np.sum(np.array([dist(m.vertices[vi], point) * field[vi] for vi in nb_vertices]), axis=0) / \
               np.sum(np.array([dist(m.vertices[vi], point) for vi in nb_vertices]))

    return np.array([interpolate(m.faces[nb_faces[i]], P[i]) for i in range(len(nb_faces))])


__location__ = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(__location__, '../../experimental_setup/geltip/sim_assets')


def compute_light_field(mesh,
                        source_pos,
                        cloud_map,
                        method='geodesic',
                        log_progress=True,
                        n_processes=None,
                        process_i=None,
                        field_i=None):
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

    elif method == 'plane':
        field = np.zeros(cloud_map.shape)

        for i in range(cloud_map.shape[0]):
            for j in range(cloud_map.shape[1]):

                if log_progress and j == (cloud_map.shape[1] - 1):
                    progress = ((i * cloud_map.shape[1] + j) / (cloud_map.shape[0] * cloud_map.shape[1]))
                    print('progress... ' + str(round(progress * 100, 2)) + '%')

                if n_processes is None or j % n_processes == process_i:
                    pt = cloud_map[i, j]

                    n = np.array([source_pos[0], source_pos[1], 0])  # normal vector to the surface, at the source led.
                    # this works because the mesh is cylinder (on the sides)
                    # todo this should be generalized for more general cases
                    v = pt - source_pos  # we are going to translate this vector
                    vi, ve, s_path = translate_vector(mesh, source_pos, pt, v, n)
                    field[i, j] = ve
        return field
    # elif method == 'plane2':
    #     prefix = str(cloud_map.shape[1]) + 'x' + str(cloud_map.shape[0])
    #     field = np.zeros(cloud_map.shape)
    #
    #     dx = normalize_vectors(partial_derivative(cloud_map, 'x'))
    #     dy = normalize_vectors(partial_derivative(cloud_map, 'y'))
    #     normals = -np.cross(dx, dy)
    #
    #     plane = np.load(assets_path + '/transport_' + prefix + '_field_' + str(field_i) + '.npy')
    #     for i in range(cloud_map.shape[0]):
    #         for j in range(cloud_map.shape[1]):
    #
    #             if log_progress and j == (cloud_map.shape[1] - 1):
    #                 progress = ((i * cloud_map.shape[1] + j) / (cloud_map.shape[0] * cloud_map.shape[1]))
    #                 print('progress... ' + str(round(progress * 100, 2)) + '%')
    #
    #             if m[i, j] > 0.5:
    #                 # lm = cloud_map[i, j] - source_pos
    #                 # d = norm(lm)
    #                 # lm /= (d * d * 50)
    #                 print(norm(plane[i, j]), norm(normals[i, j]))
    #                 field[i, j] = normalize(plane[i, j]) + 0.1 * normals[i, j]
    #     return field

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

    elif method == 'transport' or method == 'rtransport':

        # depth-map Point-Cloud
        solver = pp3d.MeshVectorHeatSolver(mesh.vertices, mesh.faces)

        # vector that we are transporting in x,y,z
        v = np.array([0.0, 0.0, 1.0])

        # vector that we are transporting in u,v
        # projected to the mesh at the source point
        proximity_query = ProximityQuery(mesh)
        _, sourceV = proximity_query.vertex(source_pos)
        basisX, basisY, basisN = solver.get_tangent_frames()
        vx = basisX[sourceV]
        vy = basisY[sourceV]
        v_surface = [norm(proj(v, vx)), norm(proj(v, vy))]

        if method == 'rtransport':
            # vector transport
            ext = solver.transport_tangent_vector(sourceV, v_surface)

            print('1. rtransport')
            ext3D = ext[:, 0, np.newaxis] * basisX + ext[:, 1, np.newaxis] * basisY
            ext3D = map_field_vertices2points(ext3D, mesh, cloud_map.reshape([-1, 3]))
            field = ext3D.reshape((cloud_map.shape[0], cloud_map.shape[1], 3))

            for i in range(cloud_map.shape[0]):
                for j in range(cloud_map.shape[1]):
                    if m[i, j] > 0.5:
                        # vector at origin.
                        lm = cloud_map[i, j] - source_pos
                        lm /= norm(lm)

                        # projected vector on the surface, near origin
                        # vv = [norm(proj(lm, vx)), norm(proj(lm, vy))]
                        rot_matrix = rotation_matrix_from_vectors(lm, v)

                        # alpha = angle between vv and v
                        # alfa = angle2d(v, vv)

                        inv_rot_matrix = np.linalg.inv(rot_matrix)

                        field[i, j] = np.dot(inv_rot_matrix, field[i, j])

                        # rotate field[i,j] by alpha
                        # print(v_surface)
        else:
            ext = solver.transport_tangent_vector(sourceV, v_surface)
            ext3D = ext[:, 0, np.newaxis] * basisX + ext[:, 1, np.newaxis] * basisY
            ext3D = map_field_vertices2points(ext3D, mesh, cloud_map.reshape([-1, 3]))
            field = ext3D.reshape((cloud_map.shape[0], cloud_map.shape[1], 3))

        return field


def compute_field(method=None, cloud_size=None, field_i=None, n_processes=None, process_i=None, z_trans=None):
    prefix = str(cloud_size[0]) + 'x' + str(cloud_size[1])

    depth = np.load(assets_path + '/bkg.npy')
    depth = cv2.resize(depth, cloud_size, interpolation=cv2.INTER_LINEAR)
    cam_matrix = get_camera_matrix(depth.shape[::-1], fov_deg=90)
    cloud = depth2cloud(cam_matrix, depth)

    mesh = load(assets_path + '/' + prefix + f'_aligned_mesh.stl', 'stl', force='mesh')

    led_radius = 0.012
    led_height = 0

    leds = [
        np.array([cos(a * (2 * pi / 3)) * led_radius, sin(a * (2 * pi / 3)) * led_radius, led_height])
        for a in range(3)
    ]

    cloud_map = cloud.reshape((cloud_size[1], cloud_size[0], 3))

    print(f'computing field :{method} field:{str(field_i)} process:{"" if n_processes is None else process_i}')
    t_start = time.time()
    field = compute_light_field(mesh,
                                leds[field_i],
                                cloud_map,
                                method,
                                field_i=field_i,
                                n_processes=n_processes,
                                process_i=process_i)

    p_suffix = '' if n_processes is None else f'_{process_i}'
    np.save(f'{assets_path}/{method}_{prefix}_field_{str(field_i)}{p_suffix}.npy', field)

    t_end = time.time()
    delta = t_end - t_start
    print('duration seconds:', str(delta))
    print('duration minutes:', str(delta / 60))
    print('duration hours:', str(delta / 3600))
    print(f'end computing field :{method} field:{str(field_i)} process:{"" if n_processes is None else process_i}')

    # if method != 'linear':
    #     n = normals(cloud_map)
    #     r_field = field - proj_vectors(field, n)
    #     np.save(f'{assets_path}/r{method}_{prefix}_field_{str(field_i)}{p_suffix}.npy', r_field)


def merge_fields(method, cloud_size, n_processes, field_i):
    prefix = str(cloud_size[0]) + 'x' + str(cloud_size[1])

    field_partials = [
        np.load(f'{assets_path}/{method}_{prefix}_field_{str(field_i)}_{str(p)}.npy')
        for p in range(n_processes)
    ]

    shape = field_partials[0].shape
    field = np.zeros(shape)
    print('merging fields ...')
    for i in range(shape[0]):
        for j in range(shape[1]):
            field[i, j] = field_partials[j % n_processes][i, j]
    np.save(f'{assets_path}/{method}_{prefix}_field_{str(field_i)}.npy', field)
    print('end merging fields.')


def main():
    field_start_i = 0
    n_fields = 3

    # method = 'linear'
    # method = 'plane'
    # method = 'geodesic'
    method = 'rtransport'
    # method = 'plane2'
    cloud_size = (160, 120)
    z_trans = 0.010
    # cloud_size = (640, 480)
    # fields = (0, 1)
    # fields = (1, 2)

    n_processes_map = {
        'linear': None,
        'plane': 8,
        'geodesic': 8,
        'transport': None,
        'rtransport': None,
        'plane2': None,
    }

    n_processes = n_processes_map[method]

    for field_i in range(field_start_i, field_start_i + n_fields):

        if n_processes is None:
            compute_field(
                method=method,
                cloud_size=cloud_size,
                field_i=field_i,
                n_processes=n_processes,
                z_trans=z_trans
            )

        else:
            # children = []
            # for process_i in range(n_processes):
            #     child = os.fork()
            #     if child:
            #         children.append(child)
            #     else:
            #         compute_field(
            #             method=method,
            #             cloud_size=cloud_size,
            #             field_i=field_i,
            #             n_processes=n_processes,
            #             process_i=process_i,
            #             z_trans=z_trans
            #         )
            #         exit()
            #
            # for child in children:
            #     os.waitpid(child, 0)
            #
            # if len(children) > 0:
            merge_fields(method, cloud_size, n_processes, field_i)


if __name__ == '__main__':
    main()
