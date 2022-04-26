import open3d as o3d
import numpy as np
import copy
import time

# def preprocess_point_cloud(pcd, voxel_size):
#     pcd_voxel = pcd.voxel_down_sample(voxel_size)
#     pcd_voxel.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2.0, max_nn=30))
#     pcd_desc = o3d.pipelines.registration.compute_fpfh_feature(pcd_voxel, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5.0, max_nn=100))
#     return (pcd_voxel, pcd_desc)

# FILTRADO Y KEYPOINTS
def preprocess_point_cloud(pcd, voxel_size):
    # Filtramos las nubes para reducir su tamaño
    pcd_voxel = pcd.voxel_down_sample(voxel_size)
    print(pcd_voxel)

    # TODO: Extraemos puntos característicos
    tic = time.time()
    pcd_key = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd_voxel,                                          # Nube de puntos filtrada
        salient_radius=voxel_size*2,                        # TODO
        non_max_radius=voxel_size*2,                        # TODO
        gamma_21=0.5,                                       # TODO
        gamma_32=0.5  )                                     # TODO
    toc = 1000*(time.time() - tic)
    print(pcd_key)
    print("ISS Computation took {:.0f} [ms]".format(toc))

    # pcd3.paint_uniform_color([0.5, 0.5, 0.5])
    # keypoints.paint_uniform_color([255.0, 0, 0.0])

    # Calculamos los descriptores para los puntos característicos
    radius_normal = voxel_size*2
    pcd_voxel.estimate_normals(                                                     # Estimación de normales
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))      # Buscamos vecinos cercanos (como máximo 30)
    
    radius_feature = voxel_size*5
    pcd_desc = o3d.pipelines.registration.compute_fpfh_feature(                     # TODO
        pcd_voxel,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))    # Buscamos vecinos cercanos (como máximo 100)

    return pcd_voxel, pcd_desc, pcd_key

# PLANOS DOMINANTES
def plane_elimination(pcd, threshold, ransac, it):
    _, inliers = pcd.segment_plane(
        distance_threshold=threshold,                           # Rango max/menos plano dominante. Al poner 1 pilla la mesa (en m) 
        ransac_n=ransac,                                        # Al tratarse de un plano, son 3 puntos
        num_iterations=it )                                     # Número de veces que se ejecuta RANSAC
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([1.0,0,0])                 # TODO
    return outlier_cloud

def draw_registration_result(src, dst, transformation):
    src_temp = copy.deepcopy(src)
    dst_temp = copy.deepcopy(dst)
    # src_temp.paint_uniform_color([1, 0.706, 0])
    dst_temp.paint_uniform_color([0, 0.651, 0.929])
    src_temp.transform(transformation)
    o3d.visualization.draw_geometries([src_temp, dst_temp])

# Creamos una nube de puntos
pcd = o3d.geometry.PointCloud()
points = []
for i in range(100):
	for j in range(100):
		points.append([i,j,0])
pcd.points = o3d.utility.Vector3dVector(np.array(points))

# Leemos la nube de puntos creada
pcd = o3d.io.read_point_cloud("snap_0point.pcd")        # Nube de puntos de la mesa
# planta = o3d.io.read_point_cloud("s0_plant_corr.pcd")   # Nube de puntos de la planta
planta = o3d.io.read_point_cloud("s0_piggybank_corr.pcd")   # Nube de puntos de la planta

# Definimos parámetros para las fucniones
distance_threshold = 0.025
ransac_n = 3
num_iterations = 1000
voxel_size = 0.01

# Eliminamos los planos dominantes de grosor threshold = 0,025
pcd1 = plane_elimination(pcd, distance_threshold, ransac_n, num_iterations)
pcd2 = plane_elimination(pcd1, distance_threshold, ransac_n, num_iterations)
mesa = plane_elimination(pcd2, distance_threshold, ransac_n, num_iterations)
# o3d.visualization.draw_geometries([pcd3])

# (Tambien se puede usar UNIFORM SAMPLING para resumir puntos)

# print('input')
# N = 2000
# pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(N)
# fit to unit cube
# mesa.scale(1 / np.max(mesa.get_max_bound() - mesa.get_min_bound()), # no es necesario para este problema 
# center=pcd.get_center())
# mesa.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N,3)))
# o3d.visualization.draw_geometries([mesa])
# print('voxelization')
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(mesa, voxel_size=0.01)
# o3d.visualization.draw_geometries([voxel_grid])

# Filtramos la nube de puntos reducida y detectamos sus descriptores
# TODO: MESA (origen)
src_voxel, src_desc, src_key = preprocess_point_cloud(mesa, voxel_size)
# o3d.visualization.draw_geometries([src_voxel])
# o3d.visualization.draw_geometries([src_key])
# o3d.visualization.draw_geometries([src_desc])

# TODO: OBJETO (destino)
dst_voxel, dst_desc, dst_key = preprocess_point_cloud(planta, voxel_size)
# o3d.visualization.draw_geometries([dst_voxel])
# o3d.visualization.draw_geometries([dst_key])
# o3d.visualization.draw_geometries([dst_desc])

# TODO: Computa los emparajamientos entre los descriptores
distance_threshold = voxel_size*1.5
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    src_voxel,                                                                                  # Nude de puntos de origen
    dst_voxel,                                                                                  # Nube de puntos de destino
    src_desc,                                                                                   # Descriptores de la nube de puntos de origen
    dst_desc,                                                                                   # Descriptores de la nube de puntos de destino
    mutual_filter=True,                                                                         # TODO
    max_correspondence_distance=distance_threshold,                                             # Distancia máxima de pares de puntos de correspondencia
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),   # Método de estimación de los puntos
    ransac_n=3,                                                                                 # Ajuste RANSAC con ransac_n correspondencias
    checkers=[                                                                                  # TODO
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),                 # Comprueba si son similares las longitudes de cualquiera de los 2 bordes arbitrarios extraídos individualmente de las correspondencias de origen y destino
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)     # Comprueba si las nubes de puntos alineadas están cerca (menos del umbral especificado)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 100))                 # Criterios de convergencia (por defecto, max_iteration=100000 y max_validaion=100)
print(result_ransac)

draw_registration_result(mesa, planta, result_ransac.transformation)

# planta.paint_uniform_color([1, 0, 0])
# mesa.paint_uniform_color([0, 1, 0])
# o3d.visualization.draw([planta.transform(result_ransac.transformation), mesa])

# Refinamiento local de la registración de emparejamientos
distance_threshold = voxel_size*0.4

src_temp = copy.deepcopy(pcd)
dst_temp = copy.deepcopy(planta)

radius_normal = voxel_size*2
src_temp.estimate_normals(                                                          # Estimación de normales
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))      # Buscamos vecinos cercanos (como máximo 30)
dst_temp.estimate_normals(                                                          # Estimación de normales
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))      # Buscamos vecinos cercanos (como máximo 30)

result_icp = o3d.pipelines.registration.registration_icp(
    src_temp,
    dst_temp,
    distance_threshold,
    result_ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
print(result_icp)

draw_registration_result(pcd, planta, result_icp.transformation)

# ERROR MEDIO
# Calculamos las distancias entre los vecinos más cercanos )objeto respecto a la escena)
# Acumlamos las distancias
# Dividimos entre la cantidad de puntos

# ¿como tener normales fiables?
# calculamos las normales de todos los puntos con radio chiquito
# luego sacamos keypoints as usual
# luego le decimos a los keypoints normales les corresponden

# RESULTADO del ransac
# fitness = inliers / total %
# Inliers = p. dentro de umbral

src_temp = copy.deepcopy(mesa)                      # Copia de la escena
dst_temp = copy.deepcopy(planta)                    # Copia del objeto

src_temp.transform(result_ransac.transformation)    # Transformación de la escena
dst_temp.transform(result_ransac.transformation)    # Transformación del objeto

pcd_tree = o3d.geometry.KDTreeFlann(src_temp)

# TODO: Hay que tener un bucle para el objeto que busca coincidencias (vecinos) con los puntos de la escena
# Obtener num de puntos de la nube del objetp !!!