import open3d as o3d
import numpy as np
import copy
import time

# def preprocess_point_cloud(pcd, voxel_size):
#     pcd_voxel = pcd.voxel_down_sample(voxel_size)
#     pcd_voxel.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2.0, max_nn=30))
#     pcd_desc = o3d.pipelines.registration.compute_fpfh_feature(pcd_voxel, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5.0, max_nn=100))
#     return (pcd_voxel, pcd_desc)

def preprocess_point_cloud(pcd, voxel_size):
    # Filtra las nubes para reducir su tamaño
    pcd_voxel = pcd.voxel_down_sample(voxel_size)

    # Extrae puntos característicos
    tic = time.time()
    pcd_key = o3d.geometry.keypoint.compute_iss_keypoints(pcd_voxel)
    # pcd_key = o3d.geometry.keypoint.compute_iss_keypoints(pcd_voxel, salient_radius=0.005, non_max_radius=0.005, gamma_21=?, gamma12=?)
    toc = 1000*(time.time() - tic)

    # print("ISS Computation took {:.0f} [ms]".format(toc))
    # print(keypoints)
    # print(pcd_voxel)

    # pcd3.paint_uniform_color([0.5, 0.5, 0.5])
    # keypoints.paint_uniform_color([255.0, 0, 0.0])

    # Calcula los descriptores para los puntos característicos
    pcd_key.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2.0, max_nn=30))
    pcd_desc = o3d.pipelines.registration.compute_fpfh_feature(pcd_key, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5.0, max_nn=100))

    return (pcd_voxel, pcd_desc, pcd_key)

# distance_threshold --> rango max/menos plano dominante por eso al poner 1 pilla la mesa (en m) 
# ransac_n ---> al tratarse de un plano son 3 puntos
# num_iteration ---> num veces ejecuta ransac

def plane_elimination(pcd, threshold, ransac, it):
    _, inliers = pcd.segment_plane(distance_threshold=threshold, ransac_n=ransac, num_iterations=it)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([1.0,0,0])
    return (outlier_cloud)

# Crea una nube de puntos
pcd = o3d.geometry.PointCloud()

points = []
for i in range(100):
	for j in range(100):
		points.append([i,j,0])
pcd.points = o3d.utility.Vector3dVector(np.array(points))

# Lee la nube de puntos creada
pcd = o3d.io.read_point_cloud("snap_0point.pcd")

# Elimina los planos dominantes de grosor (threshold) de 0,025
pcd2 = plane_elimination(pcd, 0.025, 3, 1000)
pcd3 = plane_elimination(pcd2, 0.025, 3, 1000)
pcd3 = plane_elimination(pcd3, 0.025, 3, 1000)
# o3d.visualization.draw_geometries([pcd3])

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

# Filtra la nube de puntos reducida y detecta sus descriptores
# ESCENA
src_voxel, src_desc, src_key = preprocess_point_cloud(pcd3, 0.01)
o3d.visualization.draw_geometries([src_voxel])
o3d.visualization.draw_geometries([src_key])
# o3d.visualization.draw_geometries([src_desc])

# TODO: OBJETO
# dst_voxel, dst_desc, dst_key = preprocess_point_cloud(???, 0.01)
# o3d.visualization.draw_geometries([dst_voxel])
# o3d.visualization.draw_geometries([dst_key])
# o3d.visualization.draw_geometries([dst_desc])

# TODO: Computa los emparajamientos entre los descriptores
# result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#     src_down,
#     dst_down,
#     src_fpfh,
#     dst_fpfh,
#     mutual_filter=args.mutual_filter,
#     max_correspondence_distance=distance_threshold,
#     estimation_method=o3d.pipelines.registration.
#     TransformationEstimationPointToPoint(False),
#     ransac_n=3,
#     checkers=[
#         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
#             0.9),
#         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
#             distance_threshold)
#     ],
#     criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
#         args.max_iterations, args.confidence))