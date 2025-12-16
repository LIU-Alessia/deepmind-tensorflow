import tensorflow as tf
import numpy as np
import os

# --- 配置 ---
DATASET_NAME = 'square_room_1min_2.2m_100_segmented'
OUTPUT_DIR = os.path.join('datasets/square_room', DATASET_NAME)

NUM_FILES = 100           
TRAJS_PER_FILE = 1        
TOTAL_STEPS = 3000       
SEGMENT_LEN = 100         
NUM_SEGMENTS = TOTAL_STEPS // SEGMENT_LEN 
BOX_WIDTH = 2.2           
BOX_HEIGHT = 2.2

class TrajectoryGenerator(object):
    def __init__(self, dt=0.02, border_region=0.05):
        self.dt = dt
        self.border_region = border_region
        self.sigma_rot = 5.76 * 2  
        self.scale_fwd = 0.13      

    def compute_wall_bias(self, position, hd, box_width, box_height):
        x = position[:, 0]
        y = position[:, 1]
        dists = np.stack([box_width/2-x, box_height/2-y, x+box_width/2, y+box_height/2], axis=1)
        min_dist = np.min(dists, axis=1)
        min_idx = np.argmin(dists, axis=1)
        wall_normals = np.array([np.pi, -np.pi/2, 0, np.pi/2])
        theta = wall_normals[min_idx]
        hd = np.mod(hd, 2 * np.pi)
        a_diff = np.mod(hd - theta + np.pi, 2 * np.pi) - np.pi
        is_close = min_dist < self.border_region
        is_facing_out = np.abs(a_diff) > np.radians(80)
        bias = np.where(is_close & is_facing_out, -np.sign(a_diff) * 10.0, 0.0)
        return bias, is_close

    def generate_full_trajectory(self, box_width, box_height, batch_size, total_steps):
        """生成完整的 60000 步长轨迹"""
        samples = total_steps
        position = np.zeros([batch_size, samples + 1, 2], dtype=np.float32)
        head_dir = np.zeros([batch_size, samples + 1], dtype=np.float32)
        
        position[:, 0, 0] = np.random.uniform(-box_width/2 + 0.1, box_width/2 - 0.1, batch_size)
        position[:, 0, 1] = np.random.uniform(-box_height/2 + 0.1, box_height/2 - 0.1, batch_size)
        head_dir[:, 0] = np.random.uniform(-np.pi, np.pi, batch_size)
        
        random_turn = np.random.normal(0, self.sigma_rot, [batch_size, samples])
        random_vel = np.random.rayleigh(self.scale_fwd, [batch_size, samples])
        
        ego_vel_inputs = np.zeros([batch_size, samples, 3], dtype=np.float32)
        curr_pos = position[:, 0, :]
        curr_hd = head_dir[:, 0]
        
        for t in range(samples):
            v = random_vel[:, t]
            wall_bias, is_near = self.compute_wall_bias(curr_pos, curr_hd, box_width, box_height)
            
            turn_velocity = random_turn[:, t] + wall_bias
            v = np.where(is_near, v * 0.5, v)
            
            step_turn = turn_velocity * self.dt
            next_hd = np.mod(curr_hd + step_turn + np.pi, 2 * np.pi) - np.pi
            
            dx = v * self.dt * np.cos(next_hd)
            dy = v * self.dt * np.sin(next_hd)
            next_pos = curr_pos + np.stack([dx, dy], axis=1)
            
            lim_w = box_width/2 - 0.001
            lim_h = box_height/2 - 0.001
            next_pos[:, 0] = np.clip(next_pos[:, 0], -lim_w, lim_w)
            next_pos[:, 1] = np.clip(next_pos[:, 1], -lim_h, lim_h)
            
            position[:, t+1] = next_pos
            head_dir[:, t+1] = next_hd
            
            # 【修复点】使用 np.stack 确保形状为 (Batch, 3)
            ego_vel_inputs[:, t] = np.stack([v, np.sin(step_turn), np.cos(step_turn)], axis=1)
            
            curr_pos = next_pos
            curr_hd = next_hd
            
        return position, head_dir, ego_vel_inputs

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Generating segmented data to: {OUTPUT_DIR}")
    gen = TrajectoryGenerator(dt=0.02)
    
    for i in range(NUM_FILES):
        full_pos, full_hd, full_vel = gen.generate_full_trajectory(
            BOX_WIDTH, BOX_HEIGHT, TRAJS_PER_FILE, TOTAL_STEPS)
        
        filename = os.path.join(OUTPUT_DIR, '{:04d}-of-{:04d}.tfrecord'.format(i, NUM_FILES-1))
        
        with tf.io.TFRecordWriter(filename) as writer:
            for k in range(0, TOTAL_STEPS, SEGMENT_LEN):
                j = 0 
                
                seg_init_pos = full_pos[j, k, :]        
                seg_init_hd = full_hd[j, k]             
                
                seg_vel = full_vel[j, k : k+SEGMENT_LEN, :]
                seg_target_pos = full_pos[j, k+1 : k+1+SEGMENT_LEN, :]
                seg_target_hd = full_hd[j, k+1 : k+1+SEGMENT_LEN]
                
                if len(seg_vel) < SEGMENT_LEN:
                    continue

                feature = {
                    'init_pos': _float_feature(seg_init_pos),
                    'init_hd': _float_feature(np.array([seg_init_hd])),
                    'ego_vel': _float_feature(seg_vel),
                    'target_pos': _float_feature(seg_target_pos),
                    'target_hd': _float_feature(seg_target_hd),
                }
                writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{NUM_FILES} files")

if __name__ == "__main__":
    main()
