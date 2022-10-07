import numpy as np

from trimesh.exchange.load import load
from trimesh.transformations import scale_matrix, translation_matrix
from trimesh import Scene

from experimental_setup.geltip.sim_model.scripts.utils.vis import sphere

geltip_path = '/experimental_setup/geltip/'
mesh_path = geltip_path + 'meshes/elastomer_very_long_voxel_e-6.stl'
assets_path = geltip_path + 'sim_model/assets'

# cloud_size = (32, 24)
# cloud_size = (64, 48)
# cloud_size = (320, 240)
# cloud_size = (160, 120)
cloud_size = (640, 480)

prefix = str(cloud_size[0]) + 'x' + str(cloud_size[1])
cloud = np.load(assets_path + '/' + prefix + '_ref_cloud_clean.npy')
mesh = load(mesh_path, 'stl', force='mesh')

# TODO: mesh is visually aligned.
# this should be improved in the future.
# trans, cost = mesh.register(cloud); didn't work
mesh = mesh.apply_transform(scale_matrix(0.00105))
mesh = mesh.apply_transform(translation_matrix(np.array([0, 0, -0.015])))

mesh.export(assets_path + '/' + prefix + '_aligned_mesh_voxel_e-6.stl')
print('aligned.')

mesh.visual.face_colors = [127, 127, 127, 127]
spheres = [sphere(cloud[i]) for i in range(len(cloud)) if i % 100 == 0]
scene = Scene([mesh, *spheres])
scene.show()
