import open3d as o3d
import numpy as np
import copy
import time

# Crear nube de puntos
pcd = o3d.geometry.PointCloud()

points = []
for i in range(100):
	for j in range(100):
		points.append([i,j,0])
pcd.points = o3d.utility.Vector3dVector(np.array(points))

# Leer nube de puntos
pcd = o3d.io.read_point_cloud("snap_0point.pcd")

# distance_threshold --> rango max/menos plano dominante por eso al poner 1 pilla la mesa (en m) 
# ransac_n ---> al tratarse de un plano son 3 puntos
# num_iteration ---> num veces ejecuta ransac

# El threshol es la grosor del plano dominante
plane_model, inliers = pcd.segment_plane(distance_threshold = 0.025,ransac_n=3,num_iterations=1000)
# [a,b,c,d] = plane_model
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers,invert=True)
inlier_cloud.paint_uniform_color([1.0,0,0])

pcd2 = outlier_cloud

# El threshol es la grosor del plano dominante
plane_model, inliers = pcd2.segment_plane(distance_threshold = 0.025,ransac_n=3,num_iterations=1000)
# [a,b,c,d] = plane_model
inlier_cloud = pcd2.select_by_index(inliers)
outlier_cloud = pcd2.select_by_index(inliers,invert=True)
inlier_cloud.paint_uniform_color([0,1.0,0])

pcd3 = outlier_cloud

# El threshol es la grosor del plano dominante
plane_model, inliers = pcd3.segment_plane(distance_threshold = 0.025,ransac_n=3,num_iterations=1000)
# [a,b,c,d] = plane_model
inlier_cloud = pcd3.select_by_index(inliers)
outlier_cloud = pcd3.select_by_index(inliers,invert=True)
inlier_cloud.paint_uniform_color([0,1.0,0])

pcd3 = outlier_cloud
# o3d.visualization.draw_geometries([outlier_cloud])

# tambien se puede usar UNIFORM SAMPLING PARA RESUMIR PUNTOS

# print('input')
# N = 2000
# pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(N)
# fit to unit cube
# pcd3.scale(1 / np.max(pcd3.get_max_bound() - pcd3.get_min_bound()), # no es necesario para estae problema 
# center=pcd.get_center())
# pcd3.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
# o3d.visualization.draw_geometries([pcd3])
# print('voxelization')

# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd3, voxel_size=0.01)
# o3d.visualization.draw_geometries([voxel_grid])

# Crea un voxel grid e formato de nube de puntos
voxel_grid = pcd3.voxel_down_sample(0.01)
o3d.visualization.draw_geometries([voxel_grid])
print(voxel_grid)

# Detectamos los keypoints
tic = time.time()
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(voxel_grid)
toc = 1000 * (time.time() - tic)
print("ISS Computation took {:.0f} [ms]".format(toc))
print(keypoints) 
# pcd3.paint_uniform_color([0.5, 0.5, 0.5])
# keypoints.paint_uniform_color([255.0, 0, 0.0])
o3d.visualization.draw_geometries([keypoints])

# Descriptores
radius_normal = 0.01*2
voxel_grid.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
radius_feature = 0.01*5
desc1 = o3d.pipelines.registration.compute_fpfh_feature(voxel_grid, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
o3d.visualization.draw_geometries([desc1])

# result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(desc2, desc1
