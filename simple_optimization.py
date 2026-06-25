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
from pytorch3d.ops import sample_points_from_meshes
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

def optimize_mesh(source_mesh, target_mesh, num_iterations=2000, lr=1.0):
    """优化网格使其拟合目标"""
    # 初始化可微变形参数
    deform_verts = torch.full(source_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    
    # 设置优化器
    optimizer = torch.optim.SGD([deform_verts], lr=lr, momentum=0.9)
    
    # 正则化权重
    w_chamfer = 1.0
    w_edge = 1.0
    w_normal = 0.01
    w_laplacian = 0.1
    
    loop = tqdm(range(num_iterations))
    
    # 预采样目标点云
    target_points = sample_points_from_meshes(target_mesh, 5000)
    
    for i in loop:
        optimizer.zero_grad()
        
        # 变形网格
        deformed_mesh = source_mesh.offset_verts(deform_verts)
        
        # 采样点云
        source_points = sample_points_from_meshes(deformed_mesh, 5000)
        
        # 计算chamfer距离损失
        loss_chamfer, _ = chamfer_distance(target_points, source_points)
        
        # 计算正则化损失
        loss_edge = mesh_edge_loss(deformed_mesh)
        loss_normal = mesh_normal_consistency(deformed_mesh)
        loss_laplacian = mesh_laplacian_smoothing(deformed_mesh, method="uniform")
        
        # 总损失
        total_loss = (
            w_chamfer * loss_chamfer +
            w_edge * loss_edge +
            w_normal * loss_normal +
            w_laplacian * loss_laplacian
        )
        
        loop.set_description(f'Loss: {total_loss.item():.6f}')
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 每200次迭代保存一次结果
        if i % 200 == 0:
            final_verts, final_faces = deformed_mesh.get_mesh_verts_faces(0)
            save_obj(f'output/iter_{i}.obj', final_verts, final_faces)
            print(f'Saved iter_{i}.obj')
    
    return deformed_mesh

def main():
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 加载目标奶牛模型
    print("Loading target cow mesh...")
    cow_mesh, center, scale = load_target_mesh('/Users/Flower/Documents/trae_projects/ke wei xuan ran/pytorch3d/docs/tutorials/data/cow_mesh/cow.obj')
    
    # 初始化球体网格
    print("Initializing source sphere (ico_sphere level 4)...")
    source_mesh = ico_sphere(4, device)
    
    # 执行优化
    print("Starting optimization...")
    final_mesh = optimize_mesh(source_mesh, cow_mesh)
    
    # 保存最终结果
    print("Saving final result...")
    final_verts, final_faces = final_mesh.get_mesh_verts_faces(0)
    final_verts = final_verts * scale + center
    save_obj('output/final_cow.obj', final_verts, final_faces)
    
    print("Done!")

if __name__ == "__main__":
    main()