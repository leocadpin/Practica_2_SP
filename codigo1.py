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

# Mostrar nube de puntos
o3d.visualization.draw_geometries([pcd])

# Leer nube de puntos
pcd = o3d.io.read_point_cloud("snap_0point.pcd")

o3d.visualization.draw_geometries([pcd])

# Acceder a sus componentes
shape = np.array(pcd.points).shape
print("Nube:", pcd)
print("Shape del tensor que contiene la nube:", shape)

point = pcd.points[200]
print("Posición del punto:", point)
color = pcd.colors[200]
print("Color del punto:", color) # RGB en el rango [0...1]

# Aplicar translación
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame() # Creamos una malla que representa los ejes de coordenadas
mesh_trans = copy.deepcopy(mesh).translate((2, 2, 2), relative=False)
print("Centro del SC:", mesh.get_center())
print("Centro del SC transladado:", mesh_trans.get_center())
o3d.visualization.draw_geometries([mesh, mesh_trans])

#Aplicar rotación
mesh_rot = copy.deepcopy(mesh).translate((4, 2, 2), relative=False)
R = mesh_rot.get_rotation_matrix_from_xyz((np.pi/2.0, 0, 0)) # 90 grados eje x
mesh_rot.rotate(R, center=(0, 0, 0)) # La rotación se ejecuta con respecto del 0,0,0, no con respecto al propio objecto!
o3d.visualization.draw_geometries([mesh, mesh_trans, mesh_rot])

# Aplicar subsampling con voxelgrid
pcd_sub = pcd.voxel_down_sample(0.1) # Tamaño de la hoja de 0.05
o3d.visualization.draw_geometries([pcd_sub])
print("Shape del tensor que contiene la nube:", np.array(pcd_sub.points).shape)


# Calcular los 100 vecinos más cercano de un punto
pcd_tree = o3d.geometry.KDTreeFlann(pcd_sub)
p = [-0.76023054, -0.63303238,  1.55300009]
[k, idx, _] = pcd_tree.search_knn_vector_3d(p, 100)
np.asarray(pcd_sub.colors)[idx[1:], :] = [0, 0, 1] # Los pinto de azul
o3d.visualization.draw_geometries([pcd_sub])
