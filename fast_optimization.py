import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras,
    RasterizationSettings, MeshRasterizer, SoftSilhouetteShader, BlendParams
)
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # 加载目标
    verts, faces, _ = load_obj("cow.obj")
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)
    verts = (verts - verts.mean(0)) / max(verts.abs().max(0)[0])
    cow_mesh = Meshes(verts=[verts], faces=[faces_idx])
    
    # 快速渲染设置：更少视角、更小图像
    num_views = 8
    image_size = 128
    R, T = look_at_view_transform(2.7, torch.zeros(num_views), 
                                   torch.linspace(-180, 180, num_views))
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    sigma = 1e-3  # 更大的sigma加速收敛
    blur_radius = np.log(1./sigma - 1.) * sigma
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=blur_radius, faces_per_pixel=30
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma))
    
    target_sil = shader(rasterizer(cow_mesh.extend(num_views)), 
                        cow_mesh.extend(num_views))[..., 3]
    
    # Adam优化器，更大学习率
    src_mesh = ico_sphere(4, device)
    deform_verts = torch.zeros_like(src_mesh.verts_packed(), requires_grad=True)
    optimizer = torch.optim.Adam([deform_verts], lr=0.05)
    
    for i in range(200):
        optimizer.zero_grad()
        new_mesh = src_mesh.offset_verts(deform_verts)
        pred_sil = shader(rasterizer(new_mesh.extend(num_views)), 
                         new_mesh.extend(num_views))[..., 3]
        
        loss_sil = ((pred_sil - target_sil) ** 2).mean()
        loss = loss_sil + 0.5 * mesh_laplacian_smoothing(new_mesh)
        
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Fast Epoch {i:03d} | Loss: {loss.item():.4f}")
    
    result = src_mesh.offset_verts(deform_verts)
    save_obj("result_fast.obj", result.verts_list()[0], result.faces_list()[0])

if __name__ == "__main__":
    main()
