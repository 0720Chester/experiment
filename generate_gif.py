"""
创建优化过程的 GIF 动画
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_sphere_mesh(num_lat=8, num_lon=8):
    """创建球体网格"""
    vertices = []
    faces = []
    
    for i in range(num_lat + 1):
        lat = np.pi * i / num_lat - np.pi / 2
        for j in range(num_lon):
            lon = 2 * np.pi * j / num_lon
            
            x = np.cos(lat) * np.cos(lon)
            y = np.sin(lat)
            z = np.cos(lat) * np.sin(lon)
            
            vertices.append([x, y, z])
    
    for i in range(num_lat):
        for j in range(num_lon):
            v0 = i * num_lon + j
            v1 = i * num_lon + (j + 1) % num_lon
            v2 = (i + 1) * num_lon + j
            v3 = (i + 1) * num_lon + (j + 1) % num_lon
            
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    return torch.tensor(vertices, device=device), torch.tensor(faces, device=device)

def create_cow_like_target():
    """创建目标形状"""
    theta = np.linspace(0, 2*np.pi, 20)
    phi = np.linspace(0, np.pi, 10)
    body_verts = []
    
    for t in theta:
        for p in phi:
            x = 1.5 * np.sin(p) * np.cos(t)
            y = 0.8 * np.cos(p)
            z = 0.8 * np.sin(p) * np.sin(t)
            body_verts.append([x, y, z])
    
    faces = []
    n_phi = len(phi)
    n_theta = len(theta)
    for i in range(n_phi - 1):
        for j in range(n_theta - 1):
            v0 = i * n_theta + j
            v1 = i * n_theta + (j + 1) % n_theta
            v2 = (i + 1) * n_theta + j
            v3 = (i + 1) * n_theta + (j + 1) % n_theta
            
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    return torch.tensor(body_verts, device=device), torch.tensor(faces, device=device)

def sample_points_from_mesh(vertices, faces, num_points=1000):
    """从网格表面采样点"""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    cross = torch.cross(v1 - v0, v2 - v0)
    areas = torch.norm(cross, dim=1)
    
    total_area = torch.sum(areas)
    probs = areas / total_area
    
    num_faces = len(faces)
    selected_faces = torch.multinomial(probs, num_points, replacement=True)
    
    r1 = torch.rand(num_points, device=device)
    r2 = torch.rand(num_points, device=device)
    
    mask = (r1 + r2 > 1.0)
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]
    
    p0 = vertices[faces[selected_faces, 0]]
    p1 = vertices[faces[selected_faces, 1]]
    p2 = vertices[faces[selected_faces, 2]]
    
    points = p0 + r1.unsqueeze(1) * (p1 - p0) + r2.unsqueeze(1) * (p2 - p0)
    
    return points

def chamfer_distance(points1, points2):
    """计算Chamfer距离"""
    dist_matrix = torch.cdist(points1, points2)
    dist1 = torch.min(dist_matrix, dim=1)[0]
    dist2 = torch.min(dist_matrix, dim=0)[0]
    chamfer_dist = torch.mean(dist1) + torch.mean(dist2)
    return chamfer_dist

def laplacian_smoothing_loss(vertices, faces):
    """拉普拉斯平滑损失"""
    loss = 0.0
    num_vertices = vertices.shape[0]
    
    adjacency = [[] for _ in range(num_vertices)]
    for face in faces:
        v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
        adjacency[v0].extend([v1, v2])
        adjacency[v1].extend([v0, v2])
        adjacency[v2].extend([v0, v1])
    
    for i in range(num_vertices):
        if len(adjacency[i]) > 0:
            neighbor_indices = torch.tensor(adjacency[i], device=device, dtype=torch.long)
            neighbors = vertices[neighbor_indices]
            avg_neighbor = torch.mean(neighbors, dim=0)
            loss += torch.norm(vertices[i] - avg_neighbor)
    
    return loss / num_vertices

def edge_length_loss(vertices, faces):
    """边长一致性损失"""
    loss = 0.0
    edges = set()
    
    for face in faces:
        v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
        edges.add(tuple(sorted([v0, v1])))
        edges.add(tuple(sorted([v1, v2])))
        edges.add(tuple(sorted([v2, v0])))
    
    target_length = 0.3
    
    for edge in edges:
        v0_idx, v1_idx = edge
        length = torch.norm(vertices[v0_idx] - vertices[v1_idx])
        loss += torch.abs(length - target_length)
    
    return loss / len(edges)

def normal_consistency_loss(vertices, faces):
    """法线一致性损失"""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    normals = torch.cross(v1 - v0, v2 - v0)
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
    
    face_adjacency = {}
    for i, face in enumerate(faces):
        for v in face:
            v = int(v)
            if v not in face_adjacency:
                face_adjacency[v] = []
            face_adjacency[v].append(i)
    
    loss = 0.0
    num_comparisons = 0
    
    for v, adjacent_faces in face_adjacency.items():
        for i, f1 in enumerate(adjacent_faces):
            for f2 in adjacent_faces[i+1:]:
                loss += torch.abs(torch.dot(normals[f1], normals[f2]))
                num_comparisons += 1
    
    return loss / max(num_comparisons, 1)

def plot_mesh(vertices, faces, iteration, loss, output_dir='frames'):
    """绘制单帧图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    points = sample_points_from_mesh(vertices, faces, num_points=2000)
    points_np = points.cpu().numpy()
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=1, alpha=0.5)
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Iteration {iteration} | Loss: {loss:.4f}")
    
    plt.savefig(f'{output_dir}/frame_{iteration:04d}.png', dpi=100)
    plt.close()

def optimize_and_render():
    """优化并生成GIF"""
    # 创建网格
    source_vertices, source_faces = create_sphere_mesh(num_lat=10, num_lon=10)
    target_vertices, target_faces = create_cow_like_target()
    
    # 可优化参数
    deform_verts = torch.zeros_like(source_vertices, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([deform_verts], lr=0.1)
    
    # 预采样目标点云
    target_points = sample_points_from_mesh(target_vertices, target_faces, num_points=2000)
    
    # 权重
    w_chamfer = 1.0
    w_laplacian = 0.5
    w_edge = 0.1
    w_normal = 0.05
    
    print("优化并生成GIF帧...")
    
    # 保存初始帧
    plot_mesh(source_vertices, source_faces, 0, float('inf'))
    
    for i in range(300):
        optimizer.zero_grad()
        
        vertices = source_vertices + deform_verts
        current_points = sample_points_from_mesh(vertices, source_faces, num_points=2000)
        
        loss_chamfer = chamfer_distance(current_points, target_points)
        loss_laplacian = laplacian_smoothing_loss(vertices, source_faces)
        loss_edge = edge_length_loss(vertices, source_faces)
        loss_normal = normal_consistency_loss(vertices, source_faces)
        
        total_loss = (
            w_chamfer * loss_chamfer +
            w_laplacian * loss_laplacian +
            w_edge * loss_edge +
            w_normal * loss_normal
        )
        
        total_loss.backward()
        optimizer.step()
        
        # 每10次迭代保存一帧
        if (i + 1) % 10 == 0:
            plot_mesh(vertices.detach(), source_faces, i + 1, total_loss.item())
            print(f"Iteration {i+1}/300, Loss: {total_loss.item():.6f}")
    
    print("生成GIF...")
    
    # 读取所有帧
    frames = []
    for i in range(0, 301, 10):
        frame_path = f'frames/frame_{i:04d}.png'
        frames.append(Image.open(frame_path))
    
    # 保存GIF
    frames[0].save(
        'output/optimization.gif',
        save_all=True,
        append_images=frames[1:],
        duration=200,  # 每帧200ms
        loop=0        # 无限循环
    )
    
    print("GIF生成完成！")

if __name__ == "__main__":
    optimize_and_render()