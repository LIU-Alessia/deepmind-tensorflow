# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# 引入项目自带的模块
import dataset_reader
import model
import utils
import ensembles

# 复用 train.py 中的核心参数，必须与训练时完全一致
FLAGS = tf.flags.FLAGS

# 只有当 flags 未定义时才定义，防止重复定义报错
if not FLAGS.is_parsed():
    tf.flags.DEFINE_string('task_dataset_info', 'square_room', 'Room info')
    tf.flags.DEFINE_string('task_root', './datasets/square_room/', 'Dataset path')
    tf.flags.DEFINE_float('task_env_size', 2.2, 'Environment size')
    tf.flags.DEFINE_list('task_n_pc', [256], 'Number of place cells')
    tf.flags.DEFINE_list('task_pc_scale', [0.01], 'PC scale')
    tf.flags.DEFINE_list('task_n_hdc', [12], 'Number of HD cells')
    tf.flags.DEFINE_list('task_hdc_concentration', [20.], 'HD concentration')
    tf.flags.DEFINE_integer('task_neurons_seed', 8341, 'Seeds (必须与训练一致!)')
    tf.flags.DEFINE_string('task_targets_type', 'softmax', 'Target type')
    tf.flags.DEFINE_string('task_lstm_init_type', 'softmax', 'Init type')
    tf.flags.DEFINE_bool('task_velocity_inputs', True, 'Velocity inputs')
    tf.flags.DEFINE_list('task_velocity_noise', [0.0, 0.0, 0.0], 'Velocity noise')
    
    # Model config
    tf.flags.DEFINE_integer('model_nh_lstm', 128, 'Hidden units LSTM')
    tf.flags.DEFINE_integer('model_nh_bottleneck', 256, 'Hidden units bottleneck')
    tf.flags.DEFINE_list('model_dropout_rates', [0.5], 'Dropout rates')
    tf.flags.DEFINE_float('model_weight_decay', 1e-5, 'Weight decay')
    tf.flags.DEFINE_bool('model_bottleneck_has_bias', False, 'Bias')
    tf.flags.DEFINE_float('model_init_weight_disp', 0.0, 'Init weight disp')
    
    # Path config
    tf.flags.DEFINE_string('saver_results_directory', './results/', 'Path to results')

def decode_position(logits, place_cell_ensemble):
    """
    将模型输出的 Logits 解码为 (x, y) 坐标
    原理：计算所有 Place Cell 中心的加权平均，权重为 Softmax 概率
    """
    # 1. 计算概率分布 P(x)
    # logits shape: [batch, steps, n_cells]
    probs = tf.nn.softmax(logits, axis=-1)
    
    # 2. 获取每个 Place Cell 的中心坐标 (Means)
    # means shape: [n_cells, 2]
    # 注意：这里的 means 是通过随机种子生成的，只要种子一样，位置就一样
    cell_centers = tf.constant(place_cell_ensemble.means, dtype=tf.float32)
    
    # 3. 计算加权平均: sum(prob * center)
    # 扩展维度以便广播: [batch, steps, n_cells, 1] * [1, 1, n_cells, 2]
    weighted_centers = tf.expand_dims(probs, -1) * tf.reshape(cell_centers, [1, 1, -1, 2])
    
    # 在 cell 维度求和 -> [batch, steps, 2]
    decoded_pos = tf.reduce_sum(weighted_centers, axis=2)
    return decoded_pos

def visualize():
    tf.reset_default_graph()

    # 1. 准备数据读取器 (只读 1 个 batch 用于可视化)
    # batch_size=5, 画 5 条路径看看效果
    batch_size = 1
    data_reader = dataset_reader.DataReader(
        FLAGS.task_dataset_info, root=FLAGS.task_root, num_threads=1)
    # 读取数据 Operation
    batch_op = data_reader.read(batch_size=batch_size)
    init_pos, init_hd, ego_vel, target_pos, target_hd = batch_op

    # 2. 重新构建 Ensembles (为了获取细胞中心位置)
    place_cell_ensembles = utils.get_place_cell_ensembles(
        env_size=FLAGS.task_env_size,
        neurons_seed=FLAGS.task_neurons_seed,
        targets_type=FLAGS.task_targets_type,
        lstm_init_type=FLAGS.task_lstm_init_type,
        n_pc=FLAGS.task_n_pc,
        pc_scale=FLAGS.task_pc_scale)

    head_direction_ensembles = utils.get_head_direction_ensembles(
        neurons_seed=FLAGS.task_neurons_seed,
        targets_type=FLAGS.task_targets_type,
        lstm_init_type=FLAGS.task_lstm_init_type,
        n_hdc=FLAGS.task_n_hdc,
        hdc_concentration=FLAGS.task_hdc_concentration)
    
    target_ensembles = place_cell_ensembles + head_direction_ensembles

    # 3. 重新构建模型计算图
    rnn_core = model.GridCellsRNNCell(
        target_ensembles=target_ensembles,
        nh_lstm=FLAGS.model_nh_lstm,
        nh_bottleneck=FLAGS.model_nh_bottleneck,
        dropoutrates_bottleneck=np.array(FLAGS.model_dropout_rates),
        bottleneck_weight_decay=FLAGS.model_weight_decay,
        bottleneck_has_bias=FLAGS.model_bottleneck_has_bias,
        init_weight_disp=FLAGS.model_init_weight_disp)
    rnn = model.GridCellsRNN(rnn_core, FLAGS.model_nh_lstm)

    # 4. 构造输入并运行 RNN
    # 只需要 velocity，不需要 noise (可视化时通常去掉噪声)
    inputs = ego_vel 
    
    # 编码初始状态
    initial_conds = utils.encode_initial_conditions(
        init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
    
    # 运行模型
    # training=False 关闭 Dropout，保证预测确定性
    outputs, _ = rnn(initial_conds, inputs, training=False)
    ensembles_logits, bottleneck, lstm_output = outputs
    
    # 取出 Place Cell 的 logits (第一个 ensemble)
    pc_logits = ensembles_logits[0]
    
    # 5. 添加解码操作
    decoded_pos_op = decode_position(pc_logits, place_cell_ensembles[0])

    # 6. 恢复模型并运行
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session() as sess:
        # 启动队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        # 自动找到最新的 checkpoint
        ckpt_path = tf.train.latest_checkpoint(FLAGS.saver_results_directory)
        if ckpt_path is None:
            print("错误：在 {} 中找不到 checkpoint！".format(FLAGS.saver_results_directory))
            return
            
        print(f"正在加载模型: {ckpt_path}")
        saver.restore(sess, ckpt_path)
        
        # 运行一次获取数据和预测结果
        # 注意：这里会从 TFRecord 读取一个新的 Batch
        res_gt_pos, res_pred_pos = sess.run([target_pos, decoded_pos_op])
        
        coord.request_stop()
        coord.join(threads)

    # 7. 画图
    print("开始绘图...")
    plt.figure(figsize=(15, 5))
    
    for i in range(3): # 只画前 3 个样本
        ax = plt.subplot(1, 3, i+1)
        
        # 真实路径 (黑色实线)
        gt_x = res_gt_pos[i, :, 0]
        gt_y = res_gt_pos[i, :, 1]
        ax.plot(gt_x, gt_y, 'k-', linewidth=2, label='Ground Truth')
        ax.plot(gt_x[0], gt_y[0], 'ko', markersize=8) # 起点
        
        # 预测路径 (红色虚线)
        pred_x = res_pred_pos[i, :, 0]
        pred_y = res_pred_pos[i, :, 1]
        ax.plot(pred_x, pred_y, 'r--', linewidth=2, label='Model Prediction')
        ax.plot(pred_x[0], pred_y[0], 'ro', markersize=8) # 起点
        
        ax.set_title(f"Trajectory {i+1}")
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.grid(True)
        if i == 0:
            ax.legend()

    save_path = os.path.join(FLAGS.saver_results_directory, 'path_integration_vis.png')
    plt.savefig(save_path)
    print(f"路径积分可视化已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    visualize()