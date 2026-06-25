"""
可微光栅化实验简化版
本代码展示了可微光栅化的核心原理，不依赖pytorch3d
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
    """Sigmoid函数，用于软光栅化的平滑过渡"""
    return 1.0 / (1.0 + torch.exp(-x))

def compute_distance_to_triangle(pixel, v0, v1, v2):
    """
    计算像素到三角形的距离
    :param pixel: 像素坐标 (x, y)
    :param v0, v1, v2: 三角形三个顶点
    :return: 像素到三角形的有符号距离
    """
    # 计算边向量
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2
    
    # 计算像素到各边的向量
    d0 = pixel - v0
    d1 = pixel - v1
    d2 = pixel - v2
    
    # 计算像素到各边的投影
    t0 = torch.clamp(torch.dot(d0, e0) / torch.dot(e0, e0), 0.0, 1.0)
    t1 = torch.clamp(torch.dot(d1, e1) / torch.dot(e1, e1), 0.0, 1.0)
    t2 = torch.clamp(torch.dot(d2, e2) / torch.dot(e2, e2), 0.0, 1.0)
    
    # 计算最近点
    closest0 = v0 + t0 * e0
    closest1 = v1 + t1 * e1
    closest2 = v2 + t2 * e2
    
    # 计算距离
    dist0 = torch.norm(pixel - closest0)
    dist1 = torch.norm(pixel - closest1)
    dist2 = torch.norm(pixel - closest2)
    
    # 返回最小距离
    return torch.min(torch.stack([dist0, dist1, dist2]))

def soft_rasterize_triangle(v0, v1, v2, image_size=64, sigma=1.0):
    """
    软光栅化一个三角形
    :param v0, v1, v2: 三角形顶点（已投影到2D屏幕坐标）
    :param image_size: 图像尺寸
    :param sigma: 边缘模糊程度参数
    :return: 软光栅化后的图像
    """
    image = torch.zeros(image_size, image_size, device=device)
    
    # 遍历所有像素
    for i in range(image_size):
        for j in range(image_size):
            # 将像素坐标转换到[-1, 1]范围
            pixel = torch.tensor([(i / image_size - 0.5) * 2, (j / image_size - 0.5) * 2], device=device)
            
            # 计算距离
            dist = compute_distance_to_triangle(pixel, v0, v1, v2)
            
            # 使用sigmoid产生平滑过渡
            image[i, j] = sigmoid(-dist / sigma)
    
    return image

def create_sphere_mesh(num_lat=10, num_lon=10):
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

def project_to_2d(vertices, camera_distance=3.0):
    """
    将3D顶点投影到2D屏幕
    :param vertices: 3D顶点
    :param camera_distance: 相机距离
    :return: 2D投影坐标
    """
    # 简单的透视投影
    z = vertices[:, 2] + camera_distance
    x = vertices[:, 0] / z
    y = vertices[:, 1] / z
    return torch.stack([x, y], dim=1)

def laplacian_smoothing_loss(vertices, faces):
    """
    拉普拉斯平滑损失
    :param vertices: 顶点坐标
    :param faces: 三角形面
    :return: 拉普拉斯损失
    """
    loss = 0.0
    num_vertices = vertices.shape[0]
    
    # 构建邻接表
    adjacency = [[] for _ in range(num_vertices)]
    for face in faces:
        v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
        adjacency[v0].append(v1)
        adjacency[v0].append(v2)
        adjacency[v1].append(v0)
        adjacency[v1].append(v2)
        adjacency[v2].append(v0)
        adjacency[v2].append(v1)
    
    # 计算拉普拉斯损失
    for i in range(num_vertices):
        if len(adjacency[i]) > 0:
            # 将邻接顶点索引转换为tensor
            neighbor_indices = torch.tensor(adjacency[i], device=device, dtype=torch.long)
            neighbors = vertices[neighbor_indices]
            avg_neighbor = torch.mean(neighbors, dim=0)
            loss += torch.norm(vertices[i] - avg_neighbor)
    
    return loss / num_vertices

def edge_length_loss(vertices, faces):
    """
    边长一致性损失
    :param vertices: 顶点坐标
    :param faces: 三角形面
    :return: 边长损失
    """
    loss = 0.0
    edges = set()
    
    for face in faces:
        v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
        edges.add(tuple(sorted([v0, v1])))
        edges.add(tuple(sorted([v1, v2])))
        edges.add(tuple(sorted([v2, v0])))
    
    target_length = 0.1  # 目标边长
    
    for edge in edges:
        v0_idx, v1_idx = edge
        length = torch.norm(vertices[v0_idx] - vertices[v1_idx])
        loss += torch.abs(length - target_length)
    
    return loss / len(edges)

def create_target_silhouette():
    """
    创建一个简单的目标剪影（奶牛形状的简化版）
    :return: 目标剪影图像
    """
    image_size = 64
    image = torch.zeros(image_size, image_size, device=device)
    
    # 绘制一个简化的奶牛形状
    # 身体
    for i in range(20, 45):
        for j in range(15, 50):
            if (i - 32)**2 / 8**2 + (j - 32)**2 / 12**2 < 1:
                image[i, j] = 1.0
    
    # 头部
    for i in range(10, 25):
        for j in range(40, 55):
            if (i - 17)**2 + (j - 47)**2 < 25:
                image[i, j] = 1.0
    
    # 腿
    for i in range(38, 46):
        for j in range(18, 24):
            image[i, j] = 1.0
        for j in range(28, 34):
            image[i, j] = 1.0
        for j in range(38, 44):
            image[i, j] = 1.0
    
    return image

def optimize_mesh(source_vertices, faces, target_silhouette, num_iterations=50, lr=0.1):
    """
    优化网格使其拟合目标剪影
    :param source_vertices: 初始顶点
    :param faces: 三角形面
    :param target_silhouette: 目标剪影
    :param num_iterations: 迭代次数
    :param lr: 学习率
    :return: 优化后的顶点
    """
    # 创建可优化的顶点偏移量参数
    deform_verts = torch.zeros_like(source_vertices, device=device, requires_grad=True)
    
    # 设置优化器
    optimizer = torch.optim.Adam([deform_verts], lr=lr)
    
    # 正则化权重
    w_laplacian = 0.1
    w_edge = 0.1
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # 计算变形后的顶点
        vertices = source_vertices + deform_verts
        
        # 投影到2D
        projected = project_to_2d(vertices)
        
        # 渲染当前网格的剪影
        image_size = target_silhouette.shape[0]
        rendered = torch.zeros(image_size, image_size, device=device)
        
        for face in faces:
            v0 = projected[face[0]]
            v1 = projected[face[1]]
            v2 = projected[face[2]]
            tri_image = soft_rasterize_triangle(v0, v1, v2, image_size)
            rendered = torch.max(rendered, tri_image)
        
        # 计算剪影损失
        silhouette_loss = torch.nn.functional.mse_loss(rendered, target_silhouette)
        
        # 计算正则化损失
        laplacian_loss = laplacian_smoothing_loss(vertices, faces)
        edge_loss = edge_length_loss(vertices, faces)
        
        # 总损失
        total_loss = silhouette_loss + w_laplacian * laplacian_loss + w_edge * edge_loss
        
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {total_loss.item():.6f}")
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
    
    return (source_vertices + deform_verts).detach()

def plot_mesh(vertices, faces, title=""):
    """
    绘制3D网格
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 转换为numpy
    verts_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    
    # 绘制每个三角形
    for face in faces_np:
        x = verts_np[face, 0]
        y = verts_np[face, 1]
        z = verts_np[face, 2]
        # 绘制三角形的边
        ax.plot([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], color='blue')
        ax.plot([x[1], x[2]], [y[1], y[2]], [z[1], z[2]], color='blue')
        ax.plot([x[2], x[0]], [y[2], y[0]], [z[2], z[0]], color='blue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.savefig(f'output/{title}.png')
    plt.close()

def main():
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    print("创建初始球体网格...")
    vertices, faces = create_sphere_mesh(num_lat=8, num_lon=8)
    
    print("创建目标剪影...")
    target_silhouette = create_target_silhouette()
    
    # 保存目标剪影
    plt.figure(figsize=(6, 6))
    plt.imshow(target_silhouette.cpu().numpy(), cmap='gray')
    plt.title("Target Silhouette")
    plt.savefig('output/target_silhouette.png')
    plt.close()
    
    print("绘制初始网格...")
    plot_mesh(vertices, faces, "Initial Sphere")
    
    print("开始优化...")
    optimized_vertices = optimize_mesh(vertices, faces, target_silhouette)
    
    print("绘制优化后的网格...")
    plot_mesh(optimized_vertices, faces, "Optimized Mesh")
    
    print("完成！")

if __name__ == "__main__":
    main()