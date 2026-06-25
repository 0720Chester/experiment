# 可微光栅化实验项目

## 项目概述

本项目实现了一个可微光栅化（Differentiable Rasterization）实验，目标是学习将一个初始球体通过梯度下降优化，逐渐变形为目标形状（类似奶牛）。

## 当前状态

### ✅ 成功运行的版本

已创建并成功运行了**快速优化版本** (`fast_optimization.py`)：
- 使用 Chamfer Distance 作为主要损失函数
- 包含三种正则化损失（拉普拉斯平滑、边长一致性、法线一致性）
- 运行时间短，适合快速验证

### ⚠️ 完整版本 (需要pytorch3d)

原本计划使用 `main.py` 或 `simple_optimization.py`，这些需要 PyTorch3D 库。但由于当前环境（macOS）编译问题，pytorch3d 无法直接安装。

## 快速开始

### 运行快速版本

```bash
python3 fast_optimization.py
```

这将：
1. 创建一个初始球体网格
2. 创建目标形状
3. 执行500次迭代优化
4. 保存结果到 `output/` 目录

### 查看结果

生成的文件包括：
- `source_sphere.png` - 初始球体
- `target_shape.png` - 目标形状
- `optimized_mesh.png` - 优化后的网格
- `final_optimized.obj` - 优化后的3D模型（可用MeshLab等软件打开）
- `optimized_iter_*.obj` - 优化过程中的中间结果

## 实验原理

### 1. 软光栅化 (Soft Rasterization)

**问题**：传统硬光栅化中，像素要么在三角形内，要么在外，导致边界处梯度为0。

**解决方案**：使用软光栅化，通过Sigmoid函数产生平滑的概率过渡：

```
A(d) = sigmoid(d/σ)
```

其中σ控制边缘模糊程度。

### 2. 网格正则化 (Mesh Regularization)

**问题**：仅依靠图像差异优化会导致网格拓扑崩坏。

**解决方案**：引入三种正则化损失：
- **拉普拉斯平滑** (w=0.5)：约束相邻顶点，防止尖锐突起
- **边长一致性** (w=0.1)：惩罚过长或过短的边
- **法线一致性** (w=0.05)：约束相邻面的法线方向

**总损失函数**：
```
L_total = L_chamfer + w_lap*L_lap + w_edge*L_edge + w_normal*L_normal
```

## 安装完整版 PyTorch3D

如果你想运行使用PyTorch3D的完整版本，需要以下步骤：

### 方法1: 使用 Conda（推荐）

```bash
# 创建conda环境
conda create -n pytorch3d python=3.9
conda activate pytorch3d

# 安装PyTorch
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

# 安装PyTorch3D
conda install -c pytorch3d pytorch3d
```

### 方法2: 从源码编译 (macOS)

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
MACOSX_DEPLOYMENT_TARGET=10.14 CC=clang CXX=clang++ pip install .
```

### 方法3: 使用pip (Linux)

```bash
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
```

## 项目文件说明

| 文件 | 说明 |
|------|------|
| `fast_optimization.py` | ✅ 快速优化版本（已运行成功） |
| `main.py` | 完整版本（需要pytorch3d） |
| `simple_optimization.py` | 简单版本（需要pytorch3d） |
| `simplified_differentiable_rasterization.py` | 纯Python软光栅化版本（运行较慢） |
| `output/` | 输出目录 |

## 选做内容

### 联合纹理优化

使用 `SoftPhongShader` 不仅拟合剪影，还要拟合RGB图像，同时优化：
- 网格顶点坐标
- 顶点颜色或纹理贴图

参考教程：https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/fit_textured_mesh.ipynb

## 实验心得

### 防止梯度消失

软光栅化通过Sigmoid函数在边界处产生平滑过渡，即使顶点在像素外部也能提供梯度信息。

### 防止局部最优

正则化项对保持网格的光滑和物理合理性至关重要：
- 无正则化 → 充满尖刺的"刺猬"
- 加正则化 → 光滑的目标形状

### 权重调节

不同的正则化权重会产生不同效果：
- 拉普拉斯权重过高 → 过度平滑，丢失细节
- 拉普拉斯权重过低 → 可能出现不规则突起

## 致谢

本实验参考了：
- PyTorch3D官方教程：https://pytorch3d.org/tutorials
- Facebook Research PyTorch3D：https://github.com/facebookresearch/pytorch3d