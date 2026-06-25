import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras,
    RasterizationSettings, MeshRasterizer, SoftSilhouetteShader, BlendParams
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

def main():
    # 1. 加载目标模型
    print("加载目标模型...")
    obj_path = "cow.obj"
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"未找到 {obj_path}")
    
    verts, faces, _ = load_obj(obj_path)
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)
    verts = (verts - verts.mean(0)) / max(verts.abs().max(0)[0])
    cow_mesh = Meshes(verts=[verts], faces=[faces_idx])
    
    # 2. 设置渲染器
    print("设置渲染器...")
    num_views = 20
    R, T = look_at_view_transform(2.7, torch.zeros(num_views), 
                                   torch.linspace(-180, 180, num_views))
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    sigma = 1e-4
    blur_radius = np.log(1./sigma - 1.) * sigma
    raster_settings = RasterizationSettings(
        image_size=256, blur_radius=blur_radius, faces_per_pixel=50
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma, gamma=1e-4))
    
    # 3. 渲染目标剪影
    print("渲染目标剪影...")
    target_sil = shader(rasterizer(cow_mesh.extend(num_views)), 
                        cow_mesh.extend(num_views))[..., 3]
    
    # 4. 初始化球体并优化
    print("开始优化...")
    src_mesh = ico_sphere(4, device)
    deform_verts = torch.zeros_like(src_mesh.verts_packed(), requires_grad=True)
    optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
    
    output_dir = "output_meshes"
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(300):
        optimizer.zero_grad()
        new_mesh = src_mesh.offset_verts(deform_verts)
        pred_sil = shader(rasterizer(new_mesh.extend(num_views)), 
                         new_mesh.extend(num_views))[..., 3]
        
        loss_sil = ((pred_sil - target_sil) ** 2).mean()
        loss = (loss_sil + 
                1.0 * mesh_laplacian_smoothing(new_mesh) +
                0.1 * mesh_edge_loss(new_mesh) + 
                0.01 * mesh_normal_consistency(new_mesh))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}/300 | Loss: {loss.item():.4f} | Sil: {loss_sil.item():.4f}")
            save_obj(f"{output_dir}/mesh_{epoch:03d}.obj", 
                    new_mesh.verts_list()[0], new_mesh.faces_list()[0])
    
    # 5. 保存最终结果
    result_mesh = src_mesh.offset_verts(deform_verts)
    save_obj("final_result.obj", result_mesh.verts_list()[0], result_mesh.faces_list()[0])
    print("完成！结果保存至 final_result.obj")

if __name__ == "__main__":
    main()
