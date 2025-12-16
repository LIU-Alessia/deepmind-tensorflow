from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf
nest = tf.contrib.framework.nest

DatasetInfo = collections.namedtuple(
    'DatasetInfo', ['basepath', 'size', 'sequence_length', 'coord_range'])

_DATASETS = dict(
    square_room=DatasetInfo(
        # 【注意】对应新的文件夹名
        basepath='square_room_1min_2.2m_100_segmented', 
        size=100,             
        sequence_length=100, # 现在每个记录就是 100 步
        coord_range=((-1.1, 1.1), (-1.1, 1.1))),
)

def _get_dataset_files(dateset_info, root):
  basepath = dateset_info.basepath
  base = os.path.join(root, basepath)
  num_files = dateset_info.size
  template = '{:0%d}-of-{:0%d}.tfrecord' % (4, 4)
  return [
      os.path.join(base, template.format(i, num_files - 1))
      for i in range(num_files)
  ]

class DataReader(object):
  def __init__(self, dataset, root, num_threads=4, capacity=256, min_after_dequeue=128, seed=None):
    if dataset not in _DATASETS:
      raise ValueError('Unrecognized dataset {} requested.'.format(dataset))

    self._dataset_info = _DATASETS[dataset]
    self._train_len = 100 

    with tf.device('/cpu'):
      file_names = _get_dataset_files(self._dataset_info, root)
      filename_queue = tf.train.string_input_producer(file_names, seed=seed)
      reader = tf.TFRecordReader()

      read_ops = [
          self._make_read_op(reader, filename_queue) for _ in range(num_threads)
      ]
      dtypes = nest.map_structure(lambda x: x.dtype, read_ops[0])
      shapes = nest.map_structure(lambda x: x.shape[1:], read_ops[0])

      self._queue = tf.RandomShuffleQueue(
          capacity=capacity,
          min_after_dequeue=min_after_dequeue,
          dtypes=dtypes,
          shapes=shapes,
          seed=seed)

      enqueue_ops = [self._queue.enqueue_many(op) for op in read_ops]
      tf.train.add_queue_runner(tf.train.QueueRunner(self._queue, enqueue_ops))

  def read(self, batch_size):
    return self._queue.dequeue_many(batch_size)

  def get_coord_range(self):
    return self._dataset_info.coord_range

  def _make_read_op(self, reader, filename_queue):
    _, raw_data = reader.read_up_to(filename_queue, num_records=64)
    
    # 直接解析 100 步的数据
    seq_len = self._dataset_info.sequence_length # 100
    
    feature_map = {
        'init_pos': tf.FixedLenFeature(shape=[2], dtype=tf.float32),
        'init_hd': tf.FixedLenFeature(shape=[1], dtype=tf.float32),
        'ego_vel': tf.FixedLenFeature(shape=[seq_len, 3], dtype=tf.float32),
        'target_pos': tf.FixedLenFeature(shape=[seq_len, 2], dtype=tf.float32),
        'target_hd': tf.FixedLenFeature(shape=[seq_len, 1], dtype=tf.float32),
    }
    example = tf.parse_example(raw_data, feature_map)

    batch = [
        example['init_pos'], 
        example['init_hd'], 
        example['ego_vel'], 
        example['target_pos'], 
        example['target_hd']
    ]
    return batch
