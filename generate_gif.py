import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras,
    RasterizationSettings, MeshRasterizer, SoftSilhouetteShader, BlendParams
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # 加载目标
    verts, faces, _ = load_obj("cow.obj")
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)
    verts = (verts - verts.mean(0)) / max(verts.abs().max(0)[0])
    cow_mesh = Meshes(verts=[verts], faces=[faces_idx])
    
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
    shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma))
    
    target_sil = shader(rasterizer(cow_mesh.extend(num_views)), 
                        cow_mesh.extend(num_views))[..., 3]
    
    src_mesh = ico_sphere(4, device)
    deform_verts = torch.zeros_like(src_mesh.verts_packed(), requires_grad=True)
    optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
    
    frames = []
    epochs = 300
    save_interval = 5  # 每5步保存一帧
    
    for i in range(epochs):
        optimizer.zero_grad()
        new_mesh = src_mesh.offset_verts(deform_verts)
        pred_sil = shader(rasterizer(new_mesh.extend(num_views)), 
                         new_mesh.extend(num_views))[..., 3]
        
        loss_sil = ((pred_sil - target_sil) ** 2).mean()
        loss = (loss_sil + 1.0 * mesh_laplacian_smoothing(new_mesh) +
                0.1 * mesh_edge_loss(new_mesh) + 
                0.01 * mesh_normal_consistency(new_mesh))
        
        loss.backward()
        optimizer.step()
        
        if i % save_interval == 0:
            frames.append(pred_sil[0, ..., 3].detach().cpu().numpy())
            print(f"Frame {len(frames)} | Epoch {i:03d} | Loss: {loss.item():.4f}")
    
    # 生成GIF
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    def update(frame):
        ax.clear()
        ax.imshow(frames[frame], cmap='gray')
        ax.set_title(f'Optimization Step {frame * save_interval}')
        ax.axis('off')
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100)
    ani.save('optimization_process.gif', writer='pillow', fps=10)
    print("GIF已保存至 optimization_process.gif")

if __name__ == "__main__":
    main()
