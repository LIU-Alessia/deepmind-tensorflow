from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
import numpy as np
import tensorflow as tf
import os

# Headless mode for plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dataset_reader
import model
import scores
import utils

# Task config
tf.flags.DEFINE_string('task_dataset_info', 'square_room', 'Experiment name')
tf.flags.DEFINE_string('task_root', './datasets/square_room/', 'Dataset path')
tf.flags.DEFINE_float('task_env_size', 2.2, 'Environment size (meters)')
tf.flags.DEFINE_list('task_n_pc', [256], 'Number of target place cells')
tf.flags.DEFINE_list('task_pc_scale', [0.01], 'Place cell scale')
tf.flags.DEFINE_list('task_n_hdc', [12], 'Number of target head direction cells')
tf.flags.DEFINE_list('task_hdc_concentration', [20.], 'HD concentration')
tf.flags.DEFINE_integer('task_neurons_seed', 8341, 'Seeds')
tf.flags.DEFINE_string('task_targets_type', 'softmax', 'Target type')
tf.flags.DEFINE_string('task_lstm_init_type', 'softmax', 'LSTM init type')
tf.flags.DEFINE_bool('task_velocity_inputs', True, 'Input velocity')
tf.flags.DEFINE_list('task_velocity_noise', [0.0, 0.0, 0.0], 'Velocity noise')

# Model config
tf.flags.DEFINE_integer('model_nh_lstm', 128, 'Hidden units in LSTM')
tf.flags.DEFINE_integer('model_nh_bottleneck', 256, 'Hidden units in bottleneck')
tf.flags.DEFINE_list('model_dropout_rates', [0.5], 'Dropout rates')
tf.flags.DEFINE_float('model_weight_decay', 1e-5, 'Weight decay')
tf.flags.DEFINE_bool('model_bottleneck_has_bias', False, 'Bias in bottleneck')
tf.flags.DEFINE_float('model_init_weight_disp', 0.0, 'Initial weight displacement')

# Training config
tf.flags.DEFINE_integer('training_epochs', 1000, 'Training epochs')
tf.flags.DEFINE_integer('training_steps_per_epoch', 1000, 'Steps per epoch')
tf.flags.DEFINE_integer('training_minibatch_size', 10, 'Minibatch size')
tf.flags.DEFINE_integer('training_evaluation_minibatch_size', 4000, 'Eval batch size')
tf.flags.DEFINE_string('training_clipping_function', 'utils.clip_all_gradients', 'Clip func')
tf.flags.DEFINE_float('training_clipping', 1e-5, 'Clip value')
tf.flags.DEFINE_string('training_optimizer_class', 'tf.train.RMSPropOptimizer', 'Optimizer')
tf.flags.DEFINE_string('training_optimizer_options', '{"learning_rate": 1e-5, "momentum": 0.9}', 'Opt options')

# Store
tf.flags.DEFINE_string('saver_results_directory', './results/', 'Results dir')
tf.flags.DEFINE_integer('saver_eval_time', 10, 'Eval frequency')

FLAGS = tf.flags.FLAGS

def train():
  tf.reset_default_graph()
  global_step = tf.train.get_or_create_global_step()

  # Data Reader
  data_reader = dataset_reader.DataReader(
      FLAGS.task_dataset_info, root=FLAGS.task_root, num_threads=4)
  train_traj = data_reader.read(batch_size=FLAGS.training_minibatch_size)

  # Targets
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

  # Model
  rnn_core = model.GridCellsRNNCell(
      target_ensembles=target_ensembles,
      nh_lstm=FLAGS.model_nh_lstm,
      nh_bottleneck=FLAGS.model_nh_bottleneck,
      dropoutrates_bottleneck=np.array(FLAGS.model_dropout_rates),
      bottleneck_weight_decay=FLAGS.model_weight_decay,
      bottleneck_has_bias=FLAGS.model_bottleneck_has_bias,
      init_weight_disp=FLAGS.model_init_weight_disp)
  rnn = model.GridCellsRNN(rnn_core, FLAGS.model_nh_lstm)

  # Inputs
  init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj
  input_tensors = []
  if FLAGS.task_velocity_inputs:
    vel_noise = tf.distributions.Normal(0.0, 1.0).sample(
        sample_shape=ego_vel.get_shape()) * FLAGS.task_velocity_noise
    input_tensors = [ego_vel + vel_noise] + input_tensors
  inputs = tf.concat(input_tensors, axis=2)

  initial_conds = utils.encode_initial_conditions(
      init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
  ensembles_targets = utils.encode_targets(
      target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)

  outputs, _ = rnn(initial_conds, inputs, training=True)
  ensembles_logits, bottleneck, lstm_output = outputs

  # Loss
  pc_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=ensembles_targets[0], logits=ensembles_logits[0], name='pc_loss')
  hd_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=ensembles_targets[1], logits=ensembles_logits[1], name='hd_loss')
  
  mean_pc_loss = tf.reduce_mean(pc_loss)
  mean_hd_loss = tf.reduce_mean(hd_loss)
  total_loss = pc_loss + hd_loss
  train_loss = tf.reduce_mean(total_loss, name='train_loss')

  # Optimizer
  optimizer_class = eval(FLAGS.training_optimizer_class)
  optimizer = optimizer_class(**eval(FLAGS.training_optimizer_options))
  grad = optimizer.compute_gradients(train_loss)
  clip_gradient = eval(FLAGS.training_clipping_function)
  clipped_grad = [clip_gradient(g, var, FLAGS.training_clipping) for g, var in grad]
  train_op = optimizer.apply_gradients(clipped_grad, global_step=global_step)

  # --- TensorBoard Summaries ---
  tf.summary.scalar('Loss/Total', train_loss)
  tf.summary.scalar('Loss/Place_Cell', mean_pc_loss)
  tf.summary.scalar('Loss/Head_Direction', mean_hd_loss)
  summary_op = tf.summary.merge_all()

  # Scorer
  starts = [0.2] * 10
  ends = np.linspace(0.4, 1.0, num=10)
  masks_parameters = zip(starts, ends.tolist())
  latest_epoch_scorer = scores.GridScorer(20, data_reader.get_coord_range(), masks_parameters)

  saver = tf.train.Saver(max_to_keep=2)
  saver_hook = tf.train.CheckpointSaverHook(
      checkpoint_dir=FLAGS.saver_results_directory,
      save_steps=FLAGS.training_steps_per_epoch,
      saver=saver)
  
  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.saver_results_directory,
      summary_op=summary_op
  )

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.train.SingularMonitoredSession(
      hooks=[saver_hook, summary_hook],
      checkpoint_dir=FLAGS.saver_results_directory,
      config=config) as sess:
    
    for epoch in range(FLAGS.training_epochs):
      loss_acc = []
      for _ in range(FLAGS.training_steps_per_epoch):
        res = sess.run({'train_op': train_op, 'total_loss': train_loss})
        loss_acc.append(res['total_loss'])

      tf.logging.info('Epoch %i, mean loss %.5f', epoch, np.mean(loss_acc))
      
      # Evaluation
      if epoch % FLAGS.saver_eval_time == 0:
        res = dict()
        # Evaluate multiple batches to cover enough space
        for _ in range(FLAGS.training_evaluation_minibatch_size // FLAGS.training_minibatch_size):
          mb_res = sess.run({
              'bottleneck': bottleneck,
              'lstm': lstm_output,
              'pos_xy': target_pos
          })
          res = utils.concat_dict(res, mb_res)
        
        filename = 'rates_and_sac_epoch_{}.pdf'.format(epoch)
        utils.get_scores_and_plot(
            latest_epoch_scorer, res['pos_xy'], res['bottleneck'],
            FLAGS.saver_results_directory, filename)

def main(unused_argv):
  tf.logging.set_verbosity(3)
  train()

if __name__ == '__main__':
  tf.app.run()