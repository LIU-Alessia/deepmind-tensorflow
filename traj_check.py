import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ==========================================
# 【关键修复】开启 Eager Execution
# 这允许直接遍历 dataset，而不必建立 Session
# ==========================================
try:
    if tf.__version__.startswith('1.'):
        tf.enable_eager_execution()
    else:
        # 如果是 TF 2.x 但报这个错，可能是被意外禁用了，尝试确保它开启
        pass 
except Exception as e:
    print(f"Warning: Eager execution setup failed: {e}")

print(f"TensorFlow Version: {tf.__version__}")
print(f"Eager Execution Enabled: {tf.executing_eagerly()}")

# ==========================================
DATASET_NAME = 'square_room_20min_2.2m_100_segmented'
DATA_DIR = os.path.join('datasets/square_room', DATASET_NAME)
BOX_SIZE = 2.2
LIMIT = BOX_SIZE / 2.0

def parse_tfrecord(example_proto):
    """解析 TFRecord 中的单个样本"""
    feature_description = {
        'target_pos': tf.io.VarLenFeature(tf.float32), 
        'target_hd': tf.io.VarLenFeature(tf.float32),
        'ego_vel': tf.io.VarLenFeature(tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # 转换为 Dense Tensor 并转为 NumPy
    # 在 Eager 模式下，.numpy() 方法可用
    pos_flat = tf.sparse.to_dense(parsed['target_pos']).numpy()
    hd_flat = tf.sparse.to_dense(parsed['target_hd']).numpy()
    vel_flat = tf.sparse.to_dense(parsed['ego_vel']).numpy()
    
    # Reshape 回原始形状
    # pos: (Steps, 2)
    pos = pos_flat.reshape(-1, 2)
    # hd: (Steps, 1)
    hd = hd_flat.reshape(-1, 1)
    # vel: (Steps, 3) -> [v, sin, cos]
    vel = vel_flat.reshape(-1, 3)
    
    return pos, hd, vel

def visualize_trajectory():
    # 1. 查找文件
    # Windows 路径兼容处理
    search_path = os.path.join(DATA_DIR, "*.tfrecord")
    files = glob.glob(search_path)
    
    if not files:
        print(f"错误: 在 {DATA_DIR} 未找到 .tfrecord 文件。")
        print(f"当前工作目录: {os.getcwd()}")
        return
    
    file_path = files[0] # 取第一个文件进行可视化
    print(f"正在读取文件: {file_path}")
    
    # 2. 读取数据
    raw_dataset = tf.data.TFRecordDataset(file_path)
    
    # 3. 遍历数据 (现在开启了 Eager，这里不会报错了)
    for raw_record in raw_dataset.take(1): 
        pos, hd, vel = parse_tfrecord(raw_record)
        
        print(f"数据形状检查:")
        print(f"Position: {pos.shape}")
        print(f"Head Dir: {hd.shape}")
        print(f"Velocity: {vel.shape}")

        # --- 绘图 ---
        fig = plt.figure(figsize=(15, 10))
        
        # 子图 1: 完整轨迹 (20分钟)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(pos[:, 0], pos[:, 1], linewidth=0.5, alpha=0.7, label='Trajectory')
        # 画出 Start/End
        ax1.plot(pos[0, 0], pos[0, 1], 'go', markersize=8, label='Start')
        ax1.plot(pos[-1, 0], pos[-1, 1], 'rx', markersize=8, label='End')
        # 画出边界框
        ax1.plot([-LIMIT, LIMIT, LIMIT, -LIMIT, -LIMIT], 
                 [-LIMIT, -LIMIT, LIMIT, LIMIT, -LIMIT], 'k--', linewidth=2)
        ax1.set_title(f"Full Trajectory (20 min / {len(pos)} steps)")
        ax1.set_xlim(-LIMIT-0.2, LIMIT+0.2)
        ax1.set_ylim(-LIMIT-0.2, LIMIT+0.2)
        ax1.grid(True)
        ax1.legend()
        ax1.set_aspect('equal')

        # 子图 2: 局部放大 (前 1000 步)
        zoom_steps = 1000
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(pos[:zoom_steps, 0], pos[:zoom_steps, 1], linewidth=1, color='orange')
        ax2.plot(pos[0, 0], pos[0, 1], 'go', label='Start')
        # 画边界
        ax2.plot([-LIMIT, LIMIT, LIMIT, -LIMIT, -LIMIT], 
                 [-LIMIT, -LIMIT, LIMIT, LIMIT, -LIMIT], 'k--', linewidth=2)
        ax2.set_title(f"Zoom-in (First {zoom_steps} steps)")
        ax2.set_xlim(-LIMIT-0.2, LIMIT+0.2)
        ax2.set_ylim(-LIMIT-0.2, LIMIT+0.2)
        ax2.grid(True)
        ax2.set_aspect('equal')

        # 子图 3: 线速度分布
        v_data = vel[:, 0]
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.hist(v_data, bins=50, density=True, color='skyblue', alpha=0.7, label='Data')
        # 绘制理论 Rayleigh 分布 (b=0.13)
        x = np.linspace(0, np.max(v_data), 100)
        sigma = 0.13
        pdf = (x / sigma**2) * np.exp(-x**2 / (2 * sigma**2))
        ax3.plot(x, pdf, 'r-', linewidth=2, label='Theoretical Rayleigh (b=0.13)')
        ax3.set_title("Linear Velocity Distribution")
        ax3.set_xlabel("Speed (m/s)")
        ax3.legend()

        # 子图 4: 角速度分布
        # sin/cos -> angle -> diff -> velocity
        # 这里简略地使用 vel[:, 1] (sin) 和 vel[:, 2] (cos) 反推
        w_rad_per_step = np.arctan2(vel[:, 1], vel[:, 2])
        w_rad_per_sec = w_rad_per_step / 0.02
        
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.hist(w_rad_per_sec, bins=100, density=True, color='lightgreen', alpha=0.7, label='Data')
        ax4.set_title("Angular Velocity Distribution (rad/s)")
        ax4.set_xlabel("Angular Velocity")
        ax4.set_xlim(-15, 15)

        plt.tight_layout()
        plt.savefig('trajectory_visualization_1.png', dpi=300)

if __name__ == "__main__":
    visualize_trajectory()