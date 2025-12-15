import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ==========================================
# 开启 Eager Execution (TF 1.x 兼容)
# ==========================================
try:
    if tf.__version__.startswith('1.'):
        tf.enable_eager_execution()
except Exception:
    pass

# --- 配置 ---
# 请确保这里指向您最新的 Segmented 数据集文件夹
DATASET_NAME = 'square_room_20min_2.2m_100_segmented'
DATA_DIR = os.path.join('datasets/square_room', DATASET_NAME)
BOX_SIZE = 2.2
LIMIT = BOX_SIZE / 2.0
SEQ_LEN = 100  # 每个片段的长度

def parse_record(example_proto):
    """解析 TFRecord，提取 target_pos (100, 2)"""
    feature_description = {
        'target_pos': tf.io.VarLenFeature(tf.float32), 
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    # 还原形状
    pos = tf.sparse.to_dense(parsed['target_pos']).numpy().reshape(-1, 2)
    return pos

def visualize_dataset():
    # 1. 查找文件
    search_path = os.path.join(DATA_DIR, "*.tfrecord")
    files = glob.glob(search_path)
    
    if not files:
        print(f"错误: 在 {DATA_DIR} 未找到 .tfrecord 文件。")
        return
    
    # 我们只取第一个文件进行深度分析 (包含 600 个片段)
    file_path = files[0]
    print(f"正在分析文件: {file_path}")
    
    dataset = tf.data.TFRecordDataset(file_path)
    
    # --- 数据收集 ---
    all_positions = [] # 用于画全局覆盖图
    sample_trajs = []  # 用于画细节抽样图
    
    print("正在读取所有片段 (可能会花几秒钟)...")
    for i, raw_record in enumerate(dataset):
        pos = parse_record(raw_record)
        all_positions.append(pos)
        
        # 收集前 16 个片段用于展示细节
        if i < 16:
            sample_trajs.append(pos)
            
    # 将列表转换为大数组
    # Shape: (Num_Segments * 100, 2)
    all_positions_flat = np.concatenate(all_positions, axis=0)
    
    print(f"读取完成。总计步数: {len(all_positions_flat)}")
    print(f"总计片段数: {len(all_positions)}")

    # --- 开始绘图 ---
    fig = plt.figure(figsize=(16, 8))
    
    # === 图 1: 全局位置分布 (Coverage) ===
    ax1 = fig.add_subplot(1, 2, 1)
    
    # 画出每一个点 (使用极小的透明点)
    ax1.plot(all_positions_flat[:, 0], all_positions_flat[:, 1], '.', 
             color='blue', markersize=1, alpha=0.1)
    
    # 画边界
    ax1.plot([-LIMIT, LIMIT, LIMIT, -LIMIT, -LIMIT], 
             [-LIMIT, -LIMIT, LIMIT, LIMIT, -LIMIT], 'k-', linewidth=3)
    
    ax1.set_title(f"Global Coverage ({len(all_positions)} segments)\nShould cover the box uniformly", fontsize=14)
    ax1.set_xlim(-LIMIT-0.1, LIMIT+0.1)
    ax1.set_ylim(-LIMIT-0.1, LIMIT+0.1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # === 图 2: 片段抽样网格 (Sample Grid) ===
    # 在右侧创建一个 4x4 的子图网格
    # 使用 GridSpec 或简单的循环在右侧区域画图略显复杂，
    # 我们直接用一个新的 Figure 或者在右半边手动划分子图。
    # 这里为了简单，我们单独弹出一个窗口画细节，或者把右边分成 4x4
    
    # 让我们把右边区域分成 4x4 小图
    # 调整布局：左边占一半，右边占一半(分为16个格)
    
    # 重新定义一下布局
    plt.clf() # 清除刚才的设置
    
    # 使用 GridSpec
    gs = fig.add_gridspec(4, 8) # 4行8列
    
    # 左图：全局覆盖 (占据左半边 4行4列)
    ax_main = fig.add_subplot(gs[:, :4])
    ax_main.plot(all_positions_flat[:, 0], all_positions_flat[:, 1], '.', 
                 color='dodgerblue', markersize=2, alpha=0.05)
    ax_main.plot([-LIMIT, LIMIT, LIMIT, -LIMIT, -LIMIT], 
                 [-LIMIT, -LIMIT, LIMIT, LIMIT, -LIMIT], 'k-', linewidth=3)
    ax_main.set_title("Global Coverage (All Segments)", fontsize=16)
    ax_main.set_aspect('equal')
    ax_main.set_xlim(-LIMIT-0.1, LIMIT+0.1)
    ax_main.set_ylim(-LIMIT-0.1, LIMIT+0.1)

    # 右图：16 个样本 (占据右半边)
    for i in range(16):
        if i >= len(sample_trajs): break
        
        # 计算子图位置 (0-3行, 4-7列)
        row = i // 4
        col = 4 + (i % 4)
        
        ax_small = fig.add_subplot(gs[row, col])
        traj = sample_trajs[i]
        
        # 画轨迹
        ax_small.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=1.5, alpha=0.8)
        # 画起点终点
        ax_small.plot(traj[0, 0], traj[0, 1], 'g.', markersize=5) # Start
        ax_small.plot(traj[-1, 0], traj[-1, 1], 'k.', markersize=5) # End
        
        # 画边界
        ax_small.plot([-LIMIT, LIMIT, LIMIT, -LIMIT, -LIMIT], 
                      [-LIMIT, -LIMIT, LIMIT, LIMIT, -LIMIT], 'k--', linewidth=1, alpha=0.3)
        
        ax_small.set_xticks([])
        ax_small.set_yticks([])
        ax_small.set_xlim(-LIMIT, LIMIT)
        ax_small.set_ylim(-LIMIT, LIMIT)
        # ax_small.set_aspect('equal') 
        # 小图不需要严格 equal aspect，否则间距很难看，只看形状即可

    plt.suptitle(f"Dataset Verification: {DATASET_NAME}", fontsize=20)
    plt.tight_layout()
    plt.savefig('showall.jpg')

if __name__ == "__main__":
    visualize_dataset()