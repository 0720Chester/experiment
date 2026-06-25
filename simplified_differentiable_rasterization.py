import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def sigmoid_soft_rasterize(vertices, faces, image_size=64, sigma=1e-3):
    """
    简化版软光栅化实现
    vertices: (V, 2) 顶点坐标 [0, 1]
    faces: (F, 3) 面索引
    image_size: 输出图像大小
    sigma: 软化参数
    """
    image = torch.zeros(image_size, image_size)
    
    # 创建像素网格
    y, x = torch.meshgrid(torch.linspace(0, 1, image_size), 
                          torch.linspace(0, 1, image_size), indexing='ij')
    pixels = torch.stack([x.flatten(), y.flatten()], dim=-1)  # (N, 2)
    
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # 计算像素到三角形每条边的有符号距离
        d0 = distance_to_edge(pixels, v1, v2)
        d1 = distance_to_edge(pixels, v0, v2)
        d2 = distance_to_edge(pixels, v0, v1)
        
        # 像素在三角形内部的概率
        prob = torch.sigmoid(-d0 / sigma) * \
               torch.sigmoid(-d1 / sigma) * \
               torch.sigmoid(-d2 / sigma)
        
        image += prob.reshape(image_size, image_size)
    
    return torch.clamp(image, 0, 1)

def distance_to_edge(points, a, b):
    """计算点到线段ab的有符号距离（内部为正）"""
    ab = b - a
    ap = points - a
    t = torch.sum(ap * ab, dim=1) / torch.sum(ab * ab)
    closest = a + t.unsqueeze(1) * ab.unsqueeze(0)
    return torch.norm(points - closest, dim=1) * torch.sign(torch.cross(ab, ap))

def main():
    # 简单三角形
    vertices = torch.tensor([[0.2, 0.2], [0.8, 0.3], [0.5, 0.8]])
    faces = torch.tensor([[0, 1, 2]])
    
    sigmas = [1e-2, 1e-3, 1e-4]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for ax, sigma in zip(axes, sigmas):
        img = sigmoid_soft_rasterize(vertices, faces, sigma=sigma)
        ax.imshow(img.numpy(), cmap='gray')
        ax.set_title(f'σ = {sigma}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('soft_rasterization_demo.png')
    plt.show()

if __name__ == "__main__":
    main()
