import tensorflow as tf
import numpy as np
import os

# --- Configuration ---
DATASET_NAME = 'square_room_20min_2.2m_100_final'
OUTPUT_DIR = os.path.join('datasets/square_room', DATASET_NAME)

NUM_FILES = 100           
TRAJS_PER_FILE = 1        
SEQUENCE_LENGTH = 60000   
BOX_WIDTH = 2.2           
BOX_HEIGHT = 2.2

class TrajectoryGenerator(object):
    def __init__(self, dt=0.02, border_region=0.05):
        self.dt = dt
        self.border_region = border_region
        
        # --- Physics Parameters ---
        # Rotational noise: High value to ensure chaotic, space-filling paths
        # ~660 degrees/sec variance (matches the "jagged" look of sample images)
        self.sigma_rot = 5.76 * 2  
        
        # Forward velocity: Rayleigh distribution
        self.scale_fwd = 0.13      # 13 cm/s

    def compute_wall_bias(self, position, hd, box_width, box_height):
        '''
        Calculates a rotational bias to gently steer the agent away from walls.
        Does NOT replace random motion, only biases it.
        '''
        x = position[:, 0]
        y = position[:, 1]
        
        # 1. Distances to walls [Right, Top, Left, Bottom]
        dists = np.stack([
            box_width/2 - x,   # Right
            box_height/2 - y,  # Top
            x + box_width/2,   # Left
            y + box_height/2   # Bottom
        ], axis=1)
        
        min_dist = np.min(dists, axis=1)
        min_idx = np.argmin(dists, axis=1)
        
        # 2. Wall Normals (Pointing INWARD)
        # Right->Left(pi), Top->Down(-pi/2), Left->Right(0), Bottom->Up(pi/2)
        wall_normals = np.array([np.pi, -np.pi/2, 0, np.pi/2])
        theta = wall_normals[min_idx]
        
        # 3. Calculate Angle Difference
        # How far is current heading from the safe inward normal?
        hd = np.mod(hd, 2 * np.pi)
        a_diff = hd - theta
        a_diff = np.mod(a_diff + np.pi, 2 * np.pi) - np.pi
        
        # 4. Determine Bias
        # Condition: Close to wall AND facing somewhat outward
        # If cos(a_diff) < 0, we are facing OUT (away from normal).
        # We want to steer towards angle 0 (the normal).
        
        # Bias strength depends on how close we are and how much we are facing the wall.
        # Ideally: Turn direction = -sign(a_diff)
        
        is_close = min_dist < self.border_region
        is_facing_out = np.abs(a_diff) > np.radians(80) # Facing wall-ish
        
        should_turn = is_close & is_facing_out
        
        # Apply a strong rotational bias (e.g., 10 rad/s) towards the normal
        # logic: if a_diff is + (left of normal), bias is - (turn right)
        bias = np.where(should_turn, -np.sign(a_diff) * 10.0, 0.0)
        
        return bias, is_close

    def generate_trajectory(self, box_width, box_height, batch_size, sequence_length):
        samples = sequence_length
        position = np.zeros([batch_size, samples + 1, 2], dtype=np.float32)
        head_dir = np.zeros([batch_size, samples + 1], dtype=np.float32)
        
        # --- 1. Truly Random Initialization ---
        # Uniformly sample the entire box area
        position[:, 0, 0] = np.random.uniform(-box_width/2 + 0.1, box_width/2 - 0.1, batch_size)
        position[:, 0, 1] = np.random.uniform(-box_height/2 + 0.1, box_height/2 - 0.1, batch_size)
        head_dir[:, 0] = np.random.uniform(-np.pi, np.pi, batch_size)
        
        # Pre-generate Base Noise (The "Random" part)
        # We generate this upfront to ensure it's a standard property of the simulation
        random_turn_noise = np.random.normal(0, self.sigma_rot, [batch_size, samples])
        random_vel_noise = np.random.rayleigh(self.scale_fwd, [batch_size, samples])
        
        ego_vel_inputs = np.zeros([batch_size, samples, 3], dtype=np.float32)

        curr_pos = position[:, 0, :]
        curr_hd = head_dir[:, 0]
        
        for t in range(samples):
            # Base velocity from Rayleigh distribution
            v = random_vel_noise[:, t]
            
            # --- 2. Calculate Wall Influence ---
            wall_bias, is_near_wall = self.compute_wall_bias(curr_pos, curr_hd, box_width, box_height)
            
            # --- 3. Superimpose Forces (The Key to Randomness) ---
            # Total Turn = Random Noise + Wall Bias
            # This ensures the rat jiggles even while turning away from the wall.
            turn_velocity = random_turn_noise[:, t] + wall_bias
            
            # Slow down near walls to allow the turn to take effect before hitting
            v = np.where(is_near_wall, v * 0.5, v)
            
            # Apply Time Step
            step_turn = turn_velocity * self.dt
            
            # --- 4. Update State ---
            next_hd = curr_hd + step_turn
            next_hd = np.mod(next_hd + np.pi, 2 * np.pi) - np.pi
            
            dist = v * self.dt
            dx = dist * np.cos(next_hd)
            dy = dist * np.sin(next_hd)
            
            next_pos = curr_pos + np.stack([dx, dy], axis=1)
            
            # Safety Clip (Final Line of Defense)
            # Only strictly necessary if the bias wasn't strong enough
            limit_w = box_width/2 - 0.001
            limit_h = box_height/2 - 0.001
            next_pos[:, 0] = np.clip(next_pos[:, 0], -limit_w, limit_w)
            next_pos[:, 1] = np.clip(next_pos[:, 1], -limit_h, limit_h)
            
            # --- Store ---
            position[:, t+1] = next_pos
            head_dir[:, t+1] = next_hd
            
            # Input to Network: [v, sin(w), cos(w)]
            # Note: w (angular velocity) = turn_velocity
            ego_vel_inputs[:, t, 0] = v
            ego_vel_inputs[:, t, 1] = np.sin(turn_velocity * self.dt) 
            ego_vel_inputs[:, t, 2] = np.cos(turn_velocity * self.dt)
            
            curr_pos = next_pos
            curr_hd = next_hd

        return {
            'init_pos': position[:, 0, :],      
            'init_hd': head_dir[:, 0:1],        
            'ego_vel': ego_vel_inputs,          
            'target_pos': position[:, 1:, :],   
            'target_hd': head_dir[:, 1:, None]  
        }

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Generating data to: {OUTPUT_DIR}")
    gen = TrajectoryGenerator(dt=0.02)
    
    for i in range(NUM_FILES):
        data = gen.generate_trajectory(BOX_WIDTH, BOX_HEIGHT, TRAJS_PER_FILE, SEQUENCE_LENGTH)
        
        filename = os.path.join(OUTPUT_DIR, '{:04d}-of-{:04d}.tfrecord'.format(i, NUM_FILES-1))
        with tf.io.TFRecordWriter(filename) as writer:
            for j in range(TRAJS_PER_FILE):
                feature = {
                    'init_pos': _float_feature(data['init_pos'][j]),
                    'init_hd': _float_feature(data['init_hd'][j]),
                    'ego_vel': _float_feature(data['ego_vel'][j]),
                    'target_pos': _float_feature(data['target_pos'][j]),
                    'target_hd': _float_feature(data['target_hd'][j]),
                }
                writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{NUM_FILES} files")

    print("Done.")

if __name__ == "__main__":
    main()