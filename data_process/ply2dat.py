import open3d as o3d
import numpy as np

# ======== 参数 ========
input_ply = "data/bunny_norm.ply"  # 输入点云文件路径
output_dat = "data/bunny_grid.dat"            # 输出 .dat 文件路径
grid_size = 128                          # voxel grid 的分辨率（128³）

# ======== 1. 读取点云 ========
print("📥 正在读取点云文件...")
pcd = o3d.io.read_point_cloud(input_ply)
points = np.asarray(pcd.points)

if points.size == 0:
    raise ValueError("❌ 点云为空，请检查输入文件！")

# ======== 2. 初始化体素 grid ========
print("🧱 初始化 voxel grid...")
voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)  # 默认全是 0（无点）

# ======== 3. 将点映射到 voxel grid ========
voxel_size = 1.0 / grid_size  # 每个 voxel 的边长
indices = np.floor(points / voxel_size).astype(int)  # 计算每个点所在体素的索引
indices = np.clip(indices, 0, grid_size - 1)         # 防止越界

# 在 voxel grid 中标记有点的位置为 0
voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

# ======== 4. 展平成一维数组 ========
print("📦 展平 voxel grid...")
# 按 (x, y, z) 顺序展平 grid（C-order：z 变化最快）
flattened = voxel_grid.flatten(order='C')  # 共有 grid_size³ 个元素

# ======== 5. 保存为纯文本 .dat 文件 ========
print("💾 保存为文本格式 .dat 文件...")
with open(output_dat, "w") as f:
    for value in flattened:
        f.write(f"{value}\n")

print(f"✅ 已成功生成 {output_dat}，共 {flattened.size} 行（{grid_size}³）")