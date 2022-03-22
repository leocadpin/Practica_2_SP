import open3d as o3d
import numpy as np
import copy


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


# Mostrar nube de puntos
o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud])
