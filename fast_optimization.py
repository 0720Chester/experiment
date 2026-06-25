"""
可微光栅化实验 - 快速优化版本
基于Chamfer Distance的网格优化，避免了像素级渲染的高计算复杂度
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_sphere_mesh(num_lat=8, num_lon=8):
    """
    创建球体网格
    :param num_lat: 纬度分辨率
    :param num_lon: 经度分辨率
    :return: 顶点和三角形面
    """
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
    
    # 创建三角形面
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
    """
    创建一个类似奶牛形状的目标网格
    使用椭球体组合来近似
    """
    # 身体 - 椭球体
    theta = np.linspace(0, 2*np.pi, 20)
    phi = np.linspace(0, np.pi, 10)
    body_verts = []
    
    for t in theta:
        for p in phi:
            # 身体是椭球体
            x = 1.5 * np.sin(p) * np.cos(t)
            y = 0.8 * np.cos(p)
            z = 0.8 * np.sin(p) * np.sin(t)
            body_verts.append([x, y, z])
    
    # 创建简单的三角形面
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
    """
    从网格表面采样点
    """
    # 计算三角形面积
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # 边的叉积
    cross = torch.cross(v1 - v0, v2 - v0)
    areas = torch.norm(cross, dim=1)
    
    # 归一化面积
    total_area = torch.sum(areas)
    probs = areas / total_area
    
    # 选择三角形
    num_faces = len(faces)
    selected_faces = torch.multinomial(probs, num_points, replacement=True)
    
    # 在选中的三角形上采样点
    r1 = torch.rand(num_points, device=device)
    r2 = torch.rand(num_points, device=device)
    
    # 确保 r1 + r2 <= 1
    mask = (r1 + r2 > 1.0)
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]
    
    p0 = vertices[faces[selected_faces, 0]]
    p1 = vertices[faces[selected_faces, 1]]
    p2 = vertices[faces[selected_faces, 2]]
    
    points = p0 + r1.unsqueeze(1) * (p1 - p0) + r2.unsqueeze(1) * (p2 - p0)
    
    return points

def chamfer_distance(points1, points2):
    """
    计算两组点云之间的Chamfer距离
    """
    # points1: (N, 3)
    # points2: (M, 3)
    
    # 计算每对点之间的距离矩阵
    dist_matrix = torch.cdist(points1, points2)  # (N, M)
    
    # 到最近点的距离
    dist1 = torch.min(dist_matrix, dim=1)[0]  # (N,)
    dist2 = torch.min(dist_matrix, dim=0)[0]  # (M,)
    
    # Chamfer距离
    chamfer_dist = torch.mean(dist1) + torch.mean(dist2)
    
    return chamfer_dist

def laplacian_smoothing_loss(vertices, faces):
    """
    拉普拉斯平滑损失
    """
    loss = 0.0
    num_vertices = vertices.shape[0]
    
    # 构建邻接表
    adjacency = [[] for _ in range(num_vertices)]
    for face in faces:
        v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
        adjacency[v0].extend([v1, v2])
        adjacency[v1].extend([v0, v2])
        adjacency[v2].extend([v0, v1])
    
    # 计算拉普拉斯损失
    for i in range(num_vertices):
        if len(adjacency[i]) > 0:
            neighbor_indices = torch.tensor(adjacency[i], device=device, dtype=torch.long)
            neighbors = vertices[neighbor_indices]
            avg_neighbor = torch.mean(neighbors, dim=0)
            loss += torch.norm(vertices[i] - avg_neighbor)
    
    return loss / num_vertices

def edge_length_loss(vertices, faces):
    """
    边长一致性损失
    """
    loss = 0.0
    edges = set()
    
    for face in faces:
        v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
        edges.add(tuple(sorted([v0, v1])))
        edges.add(tuple(sorted([v1, v2])))
        edges.add(tuple(sorted([v2, v0])))
    
    target_length = 0.3  # 目标边长
    
    for edge in edges:
        v0_idx, v1_idx = edge
        length = torch.norm(vertices[v0_idx] - vertices[v1_idx])
        loss += torch.abs(length - target_length)
    
    return loss / len(edges)

def normal_consistency_loss(vertices, faces):
    """
    法线一致性损失
    """
    # 计算每个面的法线
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    normals = torch.cross(v1 - v0, v2 - v0)
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
    
    # 构建邻接面
    face_adjacency = {}
    for i, face in enumerate(faces):
        for v in face:
            v = int(v)
            if v not in face_adjacency:
                face_adjacency[v] = []
            face_adjacency[v].append(i)
    
    # 计算法线一致性
    loss = 0.0
    num_comparisons = 0
    
    for v, adjacent_faces in face_adjacency.items():
        for i, f1 in enumerate(adjacent_faces):
            for f2 in adjacent_faces[i+1:]:
                loss += torch.abs(torch.dot(normals[f1], normals[f2]))
                num_comparisons += 1
    
    return loss / max(num_comparisons, 1)

def optimize_mesh(source_vertices, faces, target_vertices, target_faces, num_iterations=500, lr=0.1):
    """
    优化网格使其拟合目标形状
    """
    # 创建可优化的顶点偏移量参数
    deform_verts = torch.zeros_like(source_vertices, device=device, requires_grad=True)
    
    # 设置优化器
    optimizer = torch.optim.Adam([deform_verts], lr=lr)
    
    # 预采样目标点云
    target_points = sample_points_from_mesh(target_vertices, target_faces, num_points=2000)
    
    # 正则化权重
    w_chamfer = 1.0
    w_laplacian = 0.5
    w_edge = 0.1
    w_normal = 0.05
    
    print("开始优化...")
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # 计算变形后的顶点
        vertices = source_vertices + deform_verts
        
        # 采样当前网格的点云
        current_points = sample_points_from_mesh(vertices, faces, num_points=2000)
        
        # 计算Chamfer距离损失
        loss_chamfer = chamfer_distance(current_points, target_points)
        
        # 计算正则化损失
        loss_laplacian = laplacian_smoothing_loss(vertices, faces)
        loss_edge = edge_length_loss(vertices, faces)
        loss_normal = normal_consistency_loss(vertices, faces)
        
        # 总损失
        total_loss = (
            w_chamfer * loss_chamfer +
            w_laplacian * loss_laplacian +
            w_edge * loss_edge +
            w_normal * loss_normal
        )
        
        if i % 50 == 0:
            print(f"Iteration {i}/{num_iterations}, Total Loss: {total_loss.item():.6f}, "
                  f"Chamfer: {loss_chamfer.item():.6f}")
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 每100次迭代保存一次
        if i % 100 == 0:
            final_verts = (source_vertices + deform_verts).detach()
            save_mesh_as_obj(final_verts, faces, f'output/optimized_iter_{i}.obj')
    
    return (source_vertices + deform_verts).detach()

def save_mesh_as_obj(vertices, faces, filename):
    """保存网格为OBJ格式"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        # 写入顶点
        for v in vertices.cpu().numpy():
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # 写入面（OBJ索引从1开始）
        for face in faces.cpu().numpy() + 1:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

def plot_pointcloud(points, title="", filename=None):
    """绘制点云"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    points_np = points.cpu().numpy()
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=1, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_loss_history(losses, filename='output/loss_history.png'):
    """绘制损失历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Optimization Loss')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    print("=" * 60)
    print("可微光栅化实验 - 简化版")
    print("=" * 60)
    
    # 1. 创建初始球体网格
    print("\n1. 创建初始球体网格...")
    source_vertices, source_faces = create_sphere_mesh(num_lat=10, num_lon=10)
    print(f"   顶点数: {len(source_vertices)}, 面数: {len(source_faces)}")
    
    # 2. 创建目标形状（类似奶牛）
    print("\n2. 创建目标形状...")
    target_vertices, target_faces = create_cow_like_target()
    print(f"   顶点数: {len(target_vertices)}, 面数: {len(target_faces)}")
    
    # 3. 可视化初始状态
    print("\n3. 保存初始状态...")
    source_points = sample_points_from_mesh(source_vertices, source_faces, num_points=2000)
    target_points = sample_points_from_mesh(target_vertices, target_faces, num_points=2000)
    plot_pointcloud(source_points, "Source Sphere", "output/source_sphere.png")
    plot_pointcloud(target_points, "Target Shape", "output/target_shape.png")
    
    # 4. 执行优化
    print("\n4. 开始优化...")
    optimized_vertices = optimize_mesh(source_vertices, source_faces, target_vertices, target_faces)
    
    # 5. 可视化结果
    print("\n5. 保存最终结果...")
    optimized_points = sample_points_from_mesh(optimized_vertices, source_faces, num_points=2000)
    plot_pointcloud(optimized_points, "Optimized Mesh", "output/optimized_mesh.png")
    
    # 保存为OBJ
    save_mesh_as_obj(optimized_vertices, source_faces, 'output/final_optimized.obj')
    
    print("\n" + "=" * 60)
    print("优化完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - source_sphere.png: 初始球体")
    print("  - target_shape.png: 目标形状")
    print("  - optimized_mesh.png: 优化后的网格")
    print("  - final_optimized.obj: 优化后的OBJ模型")
    print("  - optimized_iter_*.obj: 优化过程中的中间结果")

if __name__ == "__main__":
    main()