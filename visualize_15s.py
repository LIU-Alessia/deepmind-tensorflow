# -*- coding: utf-8 -*-
import os
# 【强制无头模式】防止 WSL2 绘图卡死
os.environ['MPLBACKEND'] = 'Agg'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 引入项目模块
import model
import utils
import ensembles

# 屏蔽警告
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

# 复用参数
FLAGS = tf.flags.FLAGS

if not FLAGS.is_parsed():
    tf.flags.DEFINE_string('task_dataset_info', 'square_room', 'Room info')
    tf.flags.DEFINE_string('task_root', './datasets/square_room/', 'Dataset path')
    tf.flags.DEFINE_float('task_env_size', 2.2, 'Environment size')
    tf.flags.DEFINE_list('task_n_pc', [256], 'Number of place cells')
    tf.flags.DEFINE_list('task_pc_scale', [0.01], 'PC scale')
    tf.flags.DEFINE_list('task_n_hdc', [12], 'Number of HD cells')
    tf.flags.DEFINE_list('task_hdc_concentration', [20.], 'HD concentration')
    tf.flags.DEFINE_integer('task_neurons_seed', 8341, 'Seeds')
    tf.flags.DEFINE_string('task_targets_type', 'softmax', 'Target type')
    tf.flags.DEFINE_string('task_lstm_init_type', 'softmax', 'Init type')
    tf.flags.DEFINE_bool('task_velocity_inputs', True, 'Velocity inputs')
    tf.flags.DEFINE_list('task_velocity_noise', [0.0, 0.0, 0.0], 'Velocity noise')
    
    tf.flags.DEFINE_integer('model_nh_lstm', 128, 'Hidden units LSTM')
    tf.flags.DEFINE_integer('model_nh_bottleneck', 256, 'Hidden units bottleneck')
    tf.flags.DEFINE_list('model_dropout_rates', [0.5], 'Dropout rates')
    tf.flags.DEFINE_float('model_weight_decay', 1e-5, 'Weight decay')
    tf.flags.DEFINE_bool('model_bottleneck_has_bias', False, 'Bias')
    tf.flags.DEFINE_float('model_init_weight_disp', 0.0, 'Init weight disp')
    
    tf.flags.DEFINE_string('saver_results_directory', './results/', 'Path to results')

# ------------------------------------------------------------------------
# 1. 严格遵循训练数据的生成逻辑
# ------------------------------------------------------------------------
def generate_paper_trajectory(steps):
    """
    生成符合论文描述且与训练数据分布一致的 15s 轨迹
    """
    print(f"正在生成 {steps} 步 (15秒) 轨迹...", flush=True)
    pos = np.zeros((steps, 2), dtype=np.float32)
    hd = np.zeros((steps, 1), dtype=np.float32)
    ego_vel = np.zeros((steps, 3), dtype=np.float32) # v, sin(w), cos(w)
    
    # 初始状态 (从中心开始，避免起步撞墙)
    x, y = 0.0, 0.0
    head_dir = np.random.uniform(-np.pi, np.pi)
    
    for t in range(steps):
        # 【关键】必须使用与 generate_tfrecord.py 完全一致的参数
        # 论文参数: Rayleigh scale ~0.13 m/s -> * 0.02s = 0.0026
        v = np.random.rayleigh(scale=0.13) * 0.02 
        # 论文参数: 转向噪声
        w = np.random.normal(0, 0.3) 
        
        # 更新状态
        head_dir += w
        x += v * np.cos(head_dir)
        y += v * np.sin(head_dir)
        
        # 简单的边界裁剪 (2.2m 环境 -> [-1.1, 1.1])
        x = np.clip(x, -1.1, 1.1)
        y = np.clip(y, -1.1, 1.1)
        
        pos[t] = [x, y]
        hd[t] = [head_dir]
        ego_vel[t] = [v, np.sin(w), np.cos(w)]
        
    # 增加 Batch 维度: [1, steps, dim]
    return (pos[np.newaxis, ...], 
            hd[np.newaxis, ...], 
            ego_vel[np.newaxis, ...])

def decode_position(logits, place_cell_ensemble):
    probs = tf.nn.softmax(logits, axis=-1)
    cell_centers = tf.constant(place_cell_ensemble.means, dtype=tf.float32)
    weighted_centers = tf.expand_dims(probs, -1) * tf.reshape(cell_centers, [1, 1, -1, 2])
    decoded_pos = tf.reduce_sum(weighted_centers, axis=2)
    return decoded_pos

def visualize_contrast():
    tf.reset_default_graph()
    
    # 15秒 @ 50Hz = 750步
    DURATION_STEPS = 750 

    # 1. 构建计算图 (使用 Placeholder 支持变长)
    # Shape: [batch=1, time=None, feat=3]
    ego_vel_ph = tf.placeholder(tf.float32, [1, None, 3], name="ego_vel_ph")
    init_pos_ph = tf.placeholder(tf.float32, [1, 2], name="init_pos_ph")
    init_hd_ph = tf.placeholder(tf.float32, [1, 1], name="init_hd_ph")

    # 2. Ensembles
    place_cell_ensembles = utils.get_place_cell_ensembles(
        env_size=FLAGS.task_env_size, neurons_seed=FLAGS.task_neurons_seed,
        targets_type=FLAGS.task_targets_type, lstm_init_type=FLAGS.task_lstm_init_type,
        n_pc=FLAGS.task_n_pc, pc_scale=FLAGS.task_pc_scale)

    head_direction_ensembles = utils.get_head_direction_ensembles(
        neurons_seed=FLAGS.task_neurons_seed, targets_type=FLAGS.task_targets_type,
        lstm_init_type=FLAGS.task_lstm_init_type, n_hdc=FLAGS.task_n_hdc,
        hdc_concentration=FLAGS.task_hdc_concentration)
    
    target_ensembles = place_cell_ensembles + head_direction_ensembles

    # 3. RNN 模型
    rnn_core = model.GridCellsRNNCell(
        target_ensembles=target_ensembles,
        nh_lstm=FLAGS.model_nh_lstm,
        nh_bottleneck=FLAGS.model_nh_bottleneck,
        dropoutrates_bottleneck=np.array(FLAGS.model_dropout_rates),
        bottleneck_weight_decay=FLAGS.model_weight_decay,
        bottleneck_has_bias=FLAGS.model_bottleneck_has_bias,
        init_weight_disp=FLAGS.model_init_weight_disp)
    rnn = model.GridCellsRNN(rnn_core, FLAGS.model_nh_lstm)

    # 4. 前向传播
    initial_conds = utils.encode_initial_conditions(
        init_pos_ph, init_hd_ph, place_cell_ensembles, head_direction_ensembles)
    
    outputs, _ = rnn(initial_conds, ego_vel_ph, training=False)
    
    # 解码 (取第0个ensemble即Place Cells)
    pc_logits = outputs[0][0] 
    decoded_pos_op = decode_position(pc_logits, place_cell_ensembles[0])

    # 5. 生成 15s 数据
    test_pos, test_hd, test_vel = generate_paper_trajectory(DURATION_STEPS)
    start_pos = test_pos[:, 0, :]
    start_hd = test_hd[:, 0, :]

    # 6. Session 运行 (CPU模式)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    print("加载模型进行路径积分推断...", flush=True)
    with tf.Session(config=config) as sess:
        ckpt_path = tf.train.latest_checkpoint(FLAGS.saver_results_directory)
        if ckpt_path is None:
            print(f"错误: 找不到 Checkpoint 在 {FLAGS.saver_results_directory}")
            return
        
        saver.restore(sess, ckpt_path)
        
        pred_pos_res = sess.run(decoded_pos_op, feed_dict={
            ego_vel_ph: test_vel,
            init_pos_ph: start_pos,
            init_hd_ph: start_hd
        })

    # 7. 绘图 (模仿论文 Figure 1b 风格)
    print("正在绘图...", flush=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 提取轨迹
    gt_x = test_pos[0, :, 0]
    gt_y = test_pos[0, :, 1]
    pred_x = pred_pos_res[0, :, 0]
    pred_y = pred_pos_res[0, :, 1]

    # 绘制边框 (2.2m x 2.2m)
    limit = 1.1
    rect = patches.Rectangle((-limit, -limit), 2.2, 2.2, 
                             linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # 绘制 Ground Truth (浅色，参考 Fig 1b "actual path")
    ax.plot(gt_x, gt_y, color='skyblue', linewidth=3, label='Actual path', alpha=0.8)
    
    # 绘制 Prediction (深色，参考 Fig 1b "decoded location")
    ax.plot(pred_x, pred_y, color='navy', linestyle='--', linewidth=2, label='Decoded', alpha=0.9)
    
    # 标记 Start / End
    ax.text(gt_x[0], gt_y[0], 'Start', fontsize=12, fontweight='bold', ha='right')
    ax.plot(gt_x[0], gt_y[0], 'ko', markersize=6)
    
    ax.text(gt_x[-1], gt_y[-1], 'End', fontsize=12, fontweight='bold', ha='left')
    ax.plot(gt_x[-1], gt_y[-1], 'ko', markersize=6)

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_aspect('equal')
    ax.axis('off') # 隐藏坐标轴刻度，像论文一样
    ax.legend(loc='upper right')
    ax.set_title("15s Path Integration (Paper Reproduction)")

    save_path = os.path.join(FLAGS.saver_results_directory, 'paper_repro_15s_contrast.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 论文复现图已保存至: {save_path}", flush=True)

if __name__ == '__main__':
    visualize_contrast()