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

def load_target_mesh(obj_path):
    verts, faces, _ = load_obj(obj_path)
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)
    verts = (verts - verts.mean(0)) / max(verts.abs().max(0)[0])
    return Meshes(verts=[verts], faces=[faces_idx])

def setup_renderer(num_views=20, image_size=256, sigma=1e-4):
    dist = 2.7
    elev = torch.zeros(num_views)
    azim = torch.linspace(-180, 180, num_views)
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    blur_radius = np.log(1./sigma - 1.) * sigma
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=blur_radius, faces_per_pixel=50
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma, gamma=1e-4))
    
    return rasterizer, shader

def optimize(src_mesh, target_silhouette, rasterizer, shader, 
             epochs=300, lr=1.0, momentum=0.9):
    deform_verts = torch.zeros_like(src_mesh.verts_packed(), requires_grad=True)
    optimizer = torch.optim.SGD([deform_verts], lr=lr, momentum=momentum)
    
    num_views = target_silhouette.shape[0]
    
    for i in range(epochs):
        optimizer.zero_grad()
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        pred_silhouette = shader(rasterizer(new_src_mesh.extend(num_views)), 
                                 new_src_mesh.extend(num_views))[..., 3]
        
        loss_sil = ((pred_silhouette - target_silhouette) ** 2).mean()
        loss = (loss_sil + 
                1.0 * mesh_laplacian_smoothing(new_src_mesh) +
                0.1 * mesh_edge_loss(new_src_mesh) + 
                0.01 * mesh_normal_consistency(new_src_mesh))
        
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Epoch {i:03d} | Loss: {loss.item():.4f} | Sil: {loss_sil.item():.4f}")
    
    return src_mesh.offset_verts(deform_verts)

def main():
    cow_mesh = load_target_mesh("cow.obj")
    rasterizer, shader = setup_renderer()
    
    num_views = 20
    target_silhouette = shader(rasterizer(cow_mesh.extend(num_views)), 
                               cow_mesh.extend(num_views))[..., 3]
    
    src_mesh = ico_sphere(4, device)
    result_mesh = optimize(src_mesh, target_silhouette, rasterizer, shader)
    
    save_obj("result.obj", result_mesh.verts_list()[0], result_mesh.faces_list()[0])
    print("优化完成，结果已保存至 result.obj")

if __name__ == "__main__":
    main()
