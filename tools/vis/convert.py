import pcl
import numpy as np
import open3d as o3d

fwd_cam_coods = np.load("/Users/quiescence/Desktop/fwd_cam_coords_2011_09_26_drive_0013_sync_02_0000000045.npy")
fwd_rigid_warp = np.load("/Users/quiescence/Desktop/fwd_rigid_warp_2011_09_26_drive_0013_sync_02_0000000045.npy")

points = fwd_cam_coods[0].reshape([4,-1])
p_rgb = fwd_rigid_warp[0].reshape([3,-1])

def rgb2c(r,g,b):
    return r<<16 | g<<8 | b

p_xyz = []
p_color = []

for i in range(points.shape[1]):
    x = points[0, i]
    y = points[1, i]
    z = points[2, i]
    
    r = (p_rgb[0, i]+1)/2
    g = (p_rgb[1, i]+1)/2
    b = (p_rgb[2, i]+1)/2

    p_xyz.append([x,y,z])
    p_color.append([r,g,b])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(p_xyz)
pcd.colors = o3d.utility.Vector3dVector(p_color)
o3d.io.write_point_cloud("test.pcd", pcd, write_ascii=True)