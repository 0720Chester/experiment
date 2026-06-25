import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# 添加pytorch3d路径
sys.path.insert(0, '/Users/Flower/Documents/trae_projects/ke wei xuan ran/pytorch3d')

from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

# 设置设备
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

def load_target_mesh(obj_path):
    """加载目标网格模型"""
    verts, faces, aux = load_obj(obj_path)
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)
    
    # 归一化到单位球内
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    
    return Meshes(verts=[verts], faces=[faces_idx]), center, scale

def create_cameras(num_views=12, radius=3.0):
    """创建多视角摄像机"""
    cameras = []
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        R = torch.tensor([
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)]
        ], device=device).unsqueeze(0)
        T = torch.tensor([0.0, 0.0, radius], device=device).unsqueeze(0)
        cameras.append(PerspectiveCameras(device=device, R=R, T=T))
    return cameras

def render_silhouettes(mesh, cameras, image_size=512):
    """渲染剪影图"""
    silhouettes = []
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=1e-4,
        faces_per_pixel=1,
    )
    
    for camera in cameras:
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader()
        )
        silhouette = renderer(mesh)
        silhouettes.append(silhouette[..., 3])  # 只取alpha通道
    return torch.stack(silhouettes)

def optimize_mesh(source_mesh, target_silhouettes, cameras, num_iterations=1000, lr=0.1):
    """优化网格使其拟合目标剪影"""
    # 初始化可微变形参数
    deform_verts = torch.full(source_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    
    # 设置优化器
    optimizer = torch.optim.Adam([deform_verts], lr=lr)
    
    # 正则化权重
    w_chamfer = 1.0
    w_edge = 1.0
    w_normal = 0.01
    w_laplacian = 0.1
    
    # 光栅化设置
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=np.log(1. / 1e-4 - 1.) * 1e-1,
        faces_per_pixel=10,
    )
    
    loop = tqdm(range(num_iterations))
    
    for i in loop:
        optimizer.zero_grad()
        
        # 变形网格
        deformed_mesh = source_mesh.offset_verts(deform_verts)
        
        # 计算剪影损失
        silhouette_loss = 0.0
        for j, camera in enumerate(cameras):
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=camera,
                    raster_settings=raster_settings
                ),
                shader=SoftSilhouetteShader()
            )
            pred_silhouette = renderer(deformed_mesh)[0, ..., 3]
            target_silhouette = target_silhouettes[j]
            silhouette_loss += torch.nn.functional.mse_loss(pred_silhouette, target_silhouette)
        
        silhouette_loss /= len(cameras)
        
        # 计算正则化损失
        edge_loss = mesh_edge_loss(deformed_mesh)
        normal_loss = mesh_normal_consistency(deformed_mesh)
        laplacian_loss = mesh_laplacian_smoothing(deformed_mesh, method="uniform")
        
        # 总损失
        total_loss = (
            silhouette_loss +
            w_edge * edge_loss +
            w_normal * normal_loss +
            w_laplacian * laplacian_loss
        )
        
        loop.set_description(f'Loss: {total_loss.item():.6f}')
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 每100次迭代保存一次结果
        if i % 100 == 0:
            final_verts, final_faces = deformed_mesh.get_mesh_verts_faces(0)
            save_obj(f'output/iter_{i}.obj', final_verts, final_faces)
    
    return deformed_mesh

def main():
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 加载目标奶牛模型
    print("Loading target cow mesh...")
    cow_mesh, center, scale = load_target_mesh('/Users/Flower/Documents/trae_projects/ke wei xuan ran/pytorch3d/docs/tutorials/data/cow_mesh/cow.obj')
    
    # 创建多视角摄像机
    print("Creating cameras...")
    cameras = create_cameras(num_views=12)
    
    # 渲染目标剪影
    print("Rendering target silhouettes...")
    target_silhouettes = render_silhouettes(cow_mesh, cameras)
    
    # 初始化球体网格
    print("Initializing source sphere...")
    source_mesh = ico_sphere(4, device)
    
    # 执行优化
    print("Starting optimization...")
    final_mesh = optimize_mesh(source_mesh, target_silhouettes, cameras)
    
    # 保存最终结果
    print("Saving final result...")
    final_verts, final_faces = final_mesh.get_mesh_verts_faces(0)
    final_verts = final_verts * scale + center
    save_obj('output/final_cow.obj', final_verts, final_faces)
    
    print("Done!")

if __name__ == "__main__":
    main()