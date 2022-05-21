import open3d as o3d
import numpy as np
import copy
import time

# FUNCIÓN PARA FILTRAR LA NUBE DE PUNTOS Y SACAR SUS KEYPOINTS Y DESCRIPTORES
def preprocess_point_cloud(pcd, voxel_size):
    # Estimación de normales
    tic = time.time()
    radius_normal = 0.05                                                        # TODO: Radio para las normales
    pcd.estimate_normals(                                                       
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))  # Buscamos vecinos cercanos (como máximo 30)
    toc = 1000*(time.time() - tic)
    print("Tiempo de las normales: {:.0f} [ms]".format(toc))

    # Filtramos las nubes para reducir su tamaño
    tic = time.time()
    pcd_voxel = pcd.voxel_down_sample(voxel_size)
    # print(pcd_voxel)
    toc = 1000*(time.time() - tic)
    print("Tiempo de los voxels: {:.0f} [ms]".format(toc))
       
    # Extraemos los puntos característicos
    tic = time.time()
    pcd_key = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd_voxel,                                          # Nube de puntos filtrada
        salient_radius=0.01,                               # TODO: Radio que determina cuanto de grande será la vecindad para los puntos a estudiar
        non_max_radius=0.005,                               # TODO: 
        gamma_21=0.6,                                       # TODO: Ratio entre autovalor 2 y autovalor 1
        gamma_32=0.4  )                                     # TODO: Ratio entre autovalor 3 y autovalor 2
    print(pcd_key)
    toc = 1000*(time.time() - tic)
    print("Tiempo de los keypoints: {:.0f} [ms]".format(toc))
    
    # Calculamos los descriptores para los puntos característicos
    tic = time.time()
    radius_feature = 0.01                                                          # TODO: Radio para FPFH
    pcd_desc = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_key,                                                                    # Nube de puntos de keypoints
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))    # Buscamos vecinos cercanos (como máximo 100)
    toc = 1000*(time.time() - tic)
    print("Tiempo de los descriptores: {:.0f} [ms]".format(toc))

    return pcd_voxel, pcd_desc, pcd_key

# FUNCIÓN PARA ELIMINAR LOS PLANOS DOMINANTES
def plane_elimination(pcd, threshold, ransac, it):
    # Busca un plano dominante que tiene mayor cantidad de puntos
    _, inliers = pcd.segment_plane(                             # Devuelve los puntos que están dentro del plano dominante
        distance_threshold=threshold,                           # Grueso máximo que puede tener el plano (en m) 
        ransac_n=ransac,                                        # TODO: Al tratarse de un plano, son 3 puntos
        num_iterations=it )                                     # Número de veces que se ejecuta RANSAC
    inlier_cloud = pcd.select_by_index(inliers)                 # Nube de puntos (dominantes)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)   # Nube de puntos (no dominantes)
    inlier_cloud.paint_uniform_color([1.0,0,0])
    return outlier_cloud

# FUNCIÓN PARA REPRESENTAR EL RESULTADO DE LA TRANSFORMACIÓN 
def draw_registration_result(src, dst, transformation):
    src_temp = copy.deepcopy(src)                           # Copia de la nube de origen
    dst_temp = copy.deepcopy(dst)                           # Copia de la nube de destino
    # src_temp.paint_uniform_color([1, 0.706, 0])
    dst_temp.paint_uniform_color([0, 0.651, 0.929])
    src_temp.transform(transformation)                      # Transformación de la nube origen
    o3d.visualization.draw_geometries([src_temp, dst_temp]) # Visualizamos la unión de las nubes

# FUNCIÓN PARA CALCULAR EL ERROR DE MATCHING
def matching_error(src, dst, transformation):
    src_temp = copy.deepcopy(src)                               # Copia de la nube de origen
    dst_temp = copy.deepcopy(dst)                               # Copia de la nube de destino
    src_temp.transform(transformation)                          # Transformación de la nube origen

    pcd_tree = o3d.geometry.KDTreeFlann(src_temp)               # Creamos un árbol de puntos de la nube origen (escena)
    # print(pcd_tree)

    dist_tot = 0
    num_points = len(np.asarray(dst_temp.points))               # Número de puntos de la nube destino (objeto)
    for i in range (num_points):                                # Para cada punto del objeto
        p = dst_temp.points[i]                                  # Cogemos un punto
        [k, idx, dist] = pcd_tree.search_knn_vector_3d(p, 1)    # (k) Buscamos el vecino más cercano para este punto dentro de la escena origen
                                                                # (dist) Sacamos la distancia entre el punto p y su vecino
        dist_tot = dist_tot + dist[0]                           # Acumulamos las distancias encontradas
    error = dist_tot/float(num_points)                          # Calculamos el error como la media de las distancias entre las nubes (para el total de puntos del objeto)

    return error

# FUNCIÓN QUE CALCULA LA DISTANCIA DE LA TRANSFORMACIÓN RESPECTO A UNA REFERENCIA 
def error_referencia(src, t1, t2):                      # (nube de la escena, tranformación referencia, transformación calculada actual)
    src_ref = copy.deepcopy(src)                        # Copiamos la escena dos veces
    src_actual = copy.deepcopy(src)
    src_ref.transform(t1)                               # Transformamos la nube aplicando la referencia
    src_actual.transform(t2)                            # Transformamos la nube aplicando la transformación actual
    num_points = len(np.asarray(src_actual.points))     # Número de puntos de ambas nubes
    dist_tot = 0

    # print(t1)
    # print(t2)

    for i in range (num_points):                        # Para cada par de puntos i de ambas nubes
        p1 = src_actual.points[i]        
        p2 = src_ref.points[i]                                          
        dist = np.linalg.norm(p1-p2)                    # Calculamos la distancia euclídea entre ambos puntos
        dist_tot = dist_tot + dist                      # Acumulamos las distancias encontradas
    error = dist_tot/float(num_points)                  # Calculamos el error como la media de las distancias entre las nubes (para el total de puntos del objeto)

    # dist = src_actual.compute_point_cloud_distance(src_ref)
    # # print(dist)
    # s_dist = sum(dist)
    # l_dist = len(dist)
    # error = s_dist/l_dist

    return error

start = time.time()

# Cargamos las transformaciones guardadas
icp_ref = np.load('icp.npy')
ransac_ref =  np.load('ransac.npy')

# Creamos una nube de puntos
pcd = o3d.geometry.PointCloud()

# Leemos la nube de puntos creada
pcd = o3d.io.read_point_cloud("snap_0point.pcd")            # Nube de puntos de la escena
objeto = o3d.io.read_point_cloud("s0_plant_corr.pcd")       # Nube de puntos de la planta
# objeto = o3d.io.read_point_cloud("s0_piggybank_corr.pcd")   # Nube de puntos del peluche
# objeto = o3d.io.read_point_cloud("s0_plc_corr.pcd")         # Nube de puntos del plc
# objeto = o3d.io.read_point_cloud("s0_mug_corr.pcd")         # Nube de puntos del plc

# Definimos los parámetros para las fucniones
distance_threshold = 0.025
ransac_n = 3
num_iterations = 70

# Eliminamos los planos dominantes de grosor threshold = 0,025 m
tic = time.time()
pcd1 = plane_elimination(pcd, distance_threshold, ransac_n, num_iterations)
pcd2 = plane_elimination(pcd1, distance_threshold, ransac_n, num_iterations)
mesa = plane_elimination(pcd2, distance_threshold, ransac_n, num_iterations)
# o3d.visualization.draw_geometries([pcd3])
toc = 1000*(time.time() - tic)
print("Tiempo de eliminación de planos: {:.0f} [ms]".format(toc))

# (Tambien se puede usar UNIFORM SAMPLING para resumir puntos)

# Filtramos la nube de puntos reducida y detectamos sus descriptores
voxel_size = 0.001

tic = time.time()
src_voxel, src_desc, src_key = preprocess_point_cloud(mesa, voxel_size)     # MESA (origen)
# o3d.visualization.draw_geometries([src_voxel])
# o3d.visualization.draw_geometries([src_key])
toc = 1000*(time.time() - tic)
print("Tiempo de filtrado y procesamiento de la escena: {:.0f} [ms]".format(toc))

tic = time.time()
dst_voxel, dst_desc, dst_key = preprocess_point_cloud(objeto, voxel_size)   # OBJETO (destino)
# o3d.visualization.draw_geometries([dst_voxel])
# o3d.visualization.draw_geometries([dst_key])
toc = 1000*(time.time() - tic)
print("Tiempo de filtrado y procesamiento del objeto: {:.0f} [ms]".format(toc))

# Computamos los emparejamientos entre los descriptores
tic = time.time()
distance_threshold = 0.0015
# dd = 0.0005*1.5
# print("max_correspondence_distance:", distance_threshold)                                                             # TODO: Umbral de aceptación para RANSAC
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    src_key,                                                                                    # Nude de puntos de origen (con kyepoints)
    dst_key,                                                                                    # Nube de puntos de destino (con kyepoints)
    src_desc,                                                                                   # Descriptores de la nube de puntos de origen
    dst_desc,                                                                                   # Descriptores de la nube de puntos de destino
    mutual_filter=True,                                                                         # TODO: 
    max_correspondence_distance=distance_threshold,                                             # Distancia máxima de pares de puntos de correspondencia
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),   # Método de estimación de los puntos
    ransac_n=4,                                                                                 # Ajuste RANSAC con 4 correspondencias
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),                 # TODO: Comprueba si son similares las longitudes de cualquiera de los 2 bordes arbitrarios extraídos individualmente de las correspondencias de origen y destino
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)     # Comprueba si las nubes de puntos alineadas están cerca (menos del umbral especificado)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 100))                 # TODO: Criterios de convergencia (por defecto, max_iteration=100000 y max_validaion=100)
# print(result_ransac)
toc = 1000*(time.time() - tic)
print("Tiempo de RANSAC: {:.0f} [ms]".format(toc))
draw_registration_result(mesa, objeto, result_ransac.transformation)
# t_ransac = toc

# Refinamiento local de la registración de emparejamientos
# tic = time.time()
# src_temp = copy.deepcopy(pcd)                                                   # Copia de la nube de la escena
# dst_temp = copy.deepcopy(objeto)                                                # Copia de la nube del objeto

# radius_normal = voxel_size*2                                                    # TODO: Radio para las normales
# src_temp.estimate_normals(                                                      # Estimación de normales
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))  # Buscamos vecinos cercanos (como máximo 30)
# dst_temp.estimate_normals(                                                      # Estimación de normales
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))  # Buscamos vecinos cercanos (como máximo 30)

distance_threshold = 0.0002                                             # TODO: Umbral de aceptación para ICP
result_icp = o3d.pipelines.registration.registration_icp(
    pcd,                                                                   # Nube de puntos del origen
    objeto,                                                                   # Nube de puntos del destino
    distance_threshold,                                                         # TODO: 
    result_ransac.transformation,                                               # Transformación con RANSAC
    o3d.pipelines.registration.TransformationEstimationPointToPlane())          # TODO: 
# print(result_icp)
toc = 1000*(time.time() - tic)
print("Tiempo de ICP: {:.0f} [ms]".format(toc))
draw_registration_result(pcd, objeto, result_icp.transformation)

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

# Determinamos el tiempo que tarda el algoritmo entero
finish = 1000*(time.time() - start)
print("Tiempo total del algoritmo: {:.0f} [ms]".format(finish))

# Calculamos el error de matching de las nubes
error_ransac = matching_error(pcd, objeto, result_ransac.transformation)
error_icp = matching_error(pcd, objeto, result_icp.transformation)
error_ref_ransac = error_referencia(pcd, ransac_ref, result_ransac.transformation)
error_ref_icp = error_referencia(pcd, icp_ref, result_icp.transformation)

print("Error de RANSAC:", error_ransac)
print("Error de ICP:", error_icp)
print("Error de referencia para Ransac:", error_ref_ransac)
print("Error de referencia para ICP:", error_ref_icp, "\n", "\n")

# print("aawaga", dd, "|---|", result_ransac.fitness, "|---|", result_ransac.inlier_rmse, "|---|", len(result_ransac.correspondence_set), "|---|", error_ransac, "|---|", error_icp, "|---|", error_ref_ransac, "|---|", error_ref_icp)
# print("Tiempo de RANSAC: {:.0f} [ms]".format(t_ransac))
# print("Tiempo total del algoritmo: {:.0f} [ms]".format(finish))

# # Guardamos los parámetros de lad transformaciones
# np.save('ransac', result_ransac.transformation)
# np.save('icp', result_icp.transformation)