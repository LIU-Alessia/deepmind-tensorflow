import tensorflow as tf
import numpy as np
import os

# 配置参数
STEPS = 100
NUM_FILES = 100
SAMPLES_PER_FILE = 1000  # 总共 100 * 1000 = 10万条数据
OUTPUT_DIR = "datasets/square_room/square_room_100steps_2.2m_1000000"

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def generate_trajectory(steps):
    # 简单的随机游走生成逻辑
    # 注意：这里简化了，真实的论文使用的是更复杂的 Rat Motion Model
    pos = np.zeros((steps, 2), dtype=np.float32)
    hd = np.zeros((steps, 1), dtype=np.float32)
    ego_vel = np.zeros((steps, 3), dtype=np.float32) # v, sin(w), cos(w)
    
    x, y, head_dir = 0.0, 0.0, 0.0
    
    for t in range(steps):
        # 随机速度和角速度
        v = np.random.rayleigh(scale=0.13) * 0.02 # 线速度
        w = np.random.normal(0, 0.3)  # 角速度
        
        head_dir += w
        x += v * np.cos(head_dir)
        y += v * np.sin(head_dir)
        
        # 简单的边界处理 (2.2m x 2.2m -> -1.1 to 1.1)
        x = np.clip(x, -1.1, 1.1)
        y = np.clip(y, -1.1, 1.1)
        
        pos[t] = [x, y]
        hd[t] = [head_dir]
        ego_vel[t] = [v, np.sin(w), np.cos(w)]
        
    return pos, hd, ego_vel

def create_tfrecords():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"开始生成数据到 {OUTPUT_DIR} ...")
    
    for file_idx in range(NUM_FILES):
        filename = os.path.join(OUTPUT_DIR, "{:04d}-of-{:04d}.tfrecord".format(file_idx, NUM_FILES-1))
        writer = tf.python_io.TFRecordWriter(filename)
        
        for _ in range(SAMPLES_PER_FILE):
            target_pos, target_hd, ego_vel = generate_trajectory(STEPS)
            init_pos = target_pos[0]
            init_hd = target_hd[0]
            
            # 构建 Example
            example = tf.train.Example(features=tf.train.Features(feature={
                'init_pos': _float_feature(init_pos),
                'init_hd': _float_feature(init_hd),
                'ego_vel': _float_feature(ego_vel),
                'target_pos': _float_feature(target_pos),
                'target_hd': _float_feature(target_hd)
            }))
            
            writer.write(example.SerializeToString())
        
        writer.close()
        if file_idx % 10 == 0:
            print(f"已生成文件 {file_idx}/{NUM_FILES}")

if __name__ == "__main__":
    create_tfrecords()
    print("数据生成完毕！")