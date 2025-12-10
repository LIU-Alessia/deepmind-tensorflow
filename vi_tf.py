# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# 引入项目自带的数据读取模块
import dataset_reader

# === 配置部分 ===
FLAGS = tf.flags.FLAGS

# 防止 flag 重复定义的保护逻辑
if not FLAGS.is_parsed():
    # 数据集相关参数 (必须保留以供 dataset_reader 使用)
    tf.flags.DEFINE_string('task_dataset_info', 'square_room', '数据集名称')
    
    # 注意：这里需要填入 dataset 的根目录
    # 根据您提供的路径，目录应该是这个:
    tf.flags.DEFINE_string('task_root', r'.\datasets\square_room', '数据集根目录')
    
    # 这里的参数需要与生成数据时的一致，用于解析数据
    tf.flags.DEFINE_float('task_env_size', 2.2, '环境大小 (米)')
    
    # 虽然只看位置，但 dataset_reader 初始化需要这些参数，随便填默认值即可
    tf.flags.DEFINE_list('task_n_pc', [256], '占位参数')
    tf.flags.DEFINE_list('task_pc_scale', [0.01], '占位参数')
    tf.flags.DEFINE_list('task_n_hdc', [12], '占位参数')
    tf.flags.DEFINE_list('task_hdc_concentration', [20.], '占位参数')
    tf.flags.DEFINE_integer('task_neurons_seed', 8341, '占位参数')
    tf.flags.DEFINE_string('task_targets_type', 'softmax', '占位参数')
    tf.flags.DEFINE_string('task_lstm_init_type', 'softmax', '占位参数')
    tf.flags.DEFINE_bool('task_velocity_inputs', True, '占位参数')
    tf.flags.DEFINE_list('task_velocity_noise', [0.0, 0.0, 0.0], '占位参数')

def visualize_only_tfrecord():
    tf.reset_default_graph()

    print(f"正在读取目录: {FLAGS.task_root}")

    # 1. 准备数据读取器
    # 这里设置 batch_size 为 1，因为我们只是想逐个看路径
    batch_size = 1
    data_reader = dataset_reader.DataReader(
        FLAGS.task_dataset_info, root=FLAGS.task_root, num_threads=1)

    # 2. 获取读取操作 (Tensor)
    # read() 返回: init_pos, init_hd, ego_vel, target_pos, target_hd
    batch_op = data_reader.read(batch_size=batch_size)
    
    # 我们只关心 target_pos (真实的 (x,y) 轨迹)
    target_pos_op = batch_op[3] 

    # 3. 开启 Session 读取数据
    with tf.Session() as sess:
        # 启动队列协调器 (TF 1.x 读取数据必须步骤)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            # 设定我们要看几条路径
            num_trajectories_to_show = 4 
            
            plt.figure(figsize=(5 * num_trajectories_to_show, 5))

            for i in range(num_trajectories_to_show):
                print(f"正在读取第 {i+1} 条路径...")
                
                # 运行 Session 获取真实的 Numpy 数据
                # 每次 run 都会从 TFRecord 队列中弹出一批新数据
                traj_data = sess.run(target_pos_op)
                
                # traj_data shape: [batch_size, steps, 2] -> [1, 100, 2]
                # 取出第 0 个 batch
                path_x = traj_data[0, :, 0]
                path_y = traj_data[0, :, 1]

                # 4. 绘图
                ax = plt.subplot(1, num_trajectories_to_show, i+1)
                
                # 画路径
                ax.plot(path_x, path_y, 'b-', linewidth=2, label='Ground Truth')
                # 画起点 (绿色点)
                ax.plot(path_x[0], path_y[0], 'go', markersize=8, label='Start')
                # 画终点 (红色叉)
                ax.plot(path_x[-1], path_y[-1], 'rx', markersize=8, label='End')

                # 设置环境范围 (方形房间)
                limit = FLAGS.task_env_size / 2.0
                ax.set_xlim([-limit, limit])
                ax.set_ylim([-limit, limit])
                ax.set_aspect('equal') # 保证正方形比例
                ax.grid(True)
                ax.set_title(f"Sample {i+1}")
                if i == 0:
                    ax.legend()

            plt.tight_layout()
            plt.savefig('./tf_fig_new')

        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    visualize_only_tfrecord()