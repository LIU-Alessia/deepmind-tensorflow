import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ==========================================
# 开启 Eager Execution (TF 1.x 必需)
# ==========================================
try:
    if tf.__version__.startswith('1.'):
        tf.enable_eager_execution()
except Exception:
    pass

# --- 配置 ---
# 必须指向您刚刚生成的 Segmented 文件夹
DATASET_NAME = 'square_room_20min_2.2m_100_segmented'
DATA_DIR = os.path.join('datasets/square_room', DATASET_NAME)
BOX_SIZE = 2.2
LIMIT = BOX_SIZE / 2.0

def parse_segment(example_proto):
    """解析单个 100 步的片段"""
    feature_description = {
        'init_pos': tf.io.VarLenFeature(tf.float32),
        'target_pos': tf.io.VarLenFeature(tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    init_pos = tf.sparse.to_dense(parsed['init_pos']).numpy()
    target_pos = tf.sparse.to_dense(parsed['target_pos']).numpy().reshape(-1, 2)
    
    return init_pos, target_pos

def verify_and_plot():
    # 1. 查找文件
    search_path = os.path.join(DATA_DIR, "*.tfrecord")
    files = sorted(glob.glob(search_path)) # 排序以保证顺序读取
    
    if not files:
        print(f"错误: 未找到数据文件。请检查路径: {DATA_DIR}")
        return
    
    filename = files[0]
    print(f"正在检查文件: {filename}")
    
    dataset = tf.data.TFRecordDataset(filename)
    
    # 2. 读取前 N 个片段进行拼接
    NUM_SEGMENTS_TO_PLOT = 20 # 检查前 20 段 (2000步)
    
    stitched_path = []
    prev_end_pos = None
    segment_colors = []
    
    print("\n>>> 开始连贯性检查 (Continuity Check)...")
    
    for i, raw_record in enumerate(dataset.take(NUM_SEGMENTS_TO_PLOT)):
        init_pos, target_pos = parse_segment(raw_record)
        
        # --- 检查点 A: 初始位置与上一段终点是否重合 ---
        if prev_end_pos is not None:
            # 计算距离误差
            dist_error = np.linalg.norm(init_pos - prev_end_pos)
            if dist_error > 1e-4:
                print(f"[FAIL] 第 {i} 段断开了！")
                print(f"  上一段终点: {prev_end_pos}")
                print(f"  这一段起点: {init_pos}")
                print(f"  误差距离: {dist_error}")
            else:
                print(f"[OK] 第 {i} 段连接正常 (误差: {dist_error:.2e})")
        
        # --- 检查点 B: Init Pos 与 Target[0] 的物理合理性 ---
        # 第一步移动的距离
        step1_dist = np.linalg.norm(target_pos[0] - init_pos)
        if step1_dist < 1e-5 or step1_dist > 0.1:
             print(f"[WARN] 第 {i} 段起步异常: 移动距离 {step1_dist:.4f}m")

        # 收集数据用于绘图
        # 我们把每一段都加进去。为了画图连贯，把 init_pos 也加进去
        segment_path = np.vstack([init_pos, target_pos])
        stitched_path.append(segment_path)
        
        # 记录每段的终点用于下一次检查
        prev_end_pos = target_pos[-1]
    
    print("连贯性检查完成。\n")

    # 3. 绘图验证
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # 画边界
    ax.plot([-LIMIT, LIMIT, LIMIT, -LIMIT, -LIMIT], 
            [-LIMIT, -LIMIT, LIMIT, LIMIT, -LIMIT], 'k--', linewidth=3, label='Wall')
    
    # 画每一段 (使用不同颜色交替，以便看清切分点)
    colors = ['blue', 'red']
    
    for i, seg in enumerate(stitched_path):
        c = colors[i % 2] # 蓝红交替
        # 仅给第一个 segment 加标签以免图例混乱
        lbl = 'Segment (100 steps)' if i == 0 else None
        ax.plot(seg[:, 0], seg[:, 1], color=c, linewidth=1.5, alpha=0.8, label=lbl)
        
        # 在每段的起点画个小点，确保它们是连着的
        ax.plot(seg[0, 0], seg[0, 1], '.', color='black', markersize=3)

    # 标记总起点和总终点
    start_pt = stitched_path[0][0]
    end_pt = stitched_path[-1][-1]
    ax.plot(start_pt[0], start_pt[1], 'go', markersize=10, label='Start')
    ax.plot(end_pt[0], end_pt[1], 'rx', markersize=10, label='End')

    ax.set_title(f"Stitched Trajectory ({NUM_SEGMENTS_TO_PLOT} segments / {NUM_SEGMENTS_TO_PLOT*100} steps)\nBlue/Red alternating colors indicate separate segments")
    ax.set_xlim(-LIMIT-0.1, LIMIT+0.1)
    ax.set_ylim(-LIMIT-0.1, LIMIT+0.1)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_and_plot()