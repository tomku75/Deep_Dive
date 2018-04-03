
'''
$ python download_and_convert_data.py \
    --dataset_name=mnist \
    --data_output=data
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import imageio
import random
import sys
import os

from datasets import dataset_utils

_IMAGE_SIZE = 28
_NUM_CHANNELS = 1

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# Small letters
small_letters = ['ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช',
                 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด',
                 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ',
                 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส',
                 'ห', 'ฬ', 'อ', 'ฮ']
# Select characters
_CLASS_NAMES = numbers + small_letters

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_images', None, 'train directory')

tf.app.flags.DEFINE_string('test_images',  None, 'test directory')

tf.app.flags.DEFINE_string('data_output',  None, 'The directory where the output TFRecords and temporary files are saved.')

def get_labels_and_files(folder):
  # Make a list of lists of files for each label
  filelists = []
  dir_path = []
  for char in _CLASS_NAMES:
      dir_path.append(char)
  #print(dir_path)
  for label in range(0, len(_CLASS_NAMES)):
    filelist = []
    filelists.append(filelist);
    dirname = os.path.join(folder, dir_path[label])
    #print(dirname)
    for file in os.listdir(dirname):
      if (file.endswith('.png')):
        fullname = os.path.join(dirname, file)
        #print(fullname)
        if (os.path.getsize(fullname) > 0):
          filelist.append(fullname)
        else:
          print('file ' + fullname + ' is empty')
    # sort each list of files so they start off in the same order
    # regardless of how the order the OS returns them in
    filelist.sort()
    #print(filelists[0])

  # Take the specified number of items for each label and
  # build them into an array of (label, filename) pairs
  # Since we seeded the RNG, we should get the same sample each run
  labelsAndFiles = []
  for label in range(0, len(_CLASS_NAMES)):
    filelist = random.sample(filelists[label], len(filelists[label]))
    #print(filelist)
    for filename in filelist:
      #print(filename)
      labelsAndFiles.append((label, filename))
  #print(labelsAndFiles)
  return labelsAndFiles

def make_arrays(labelsAndFiles):
  images = []
  labels = []
  #print(labelsAndFiles)
  for i in range(0, len(labelsAndFiles)):

    # display progress, since this can take a while
    if (i % 100 == 0):
      sys.stdout.write("\r%d%% complete" % ((i * 100)/len(labelsAndFiles)))
      sys.stdout.flush()

    filename = labelsAndFiles[i][1]
    try:
      image = imageio.imread(filename)
      images.append(image)
      labels.append(labelsAndFiles[i][0])
    except:
      # If this happens we won't have the requested number
      print("\nCan't read image file " + filename)

  images = np.reshape(images, (len(images), 28, 28, 1))
  imagedata = np.zeros((len(images), 28, 28, 1), dtype=np.uint8)
  labeldata = np.zeros(len(images), dtype=np.uint8)
  for i in range(0, len(labelsAndFiles)):
    imagedata[i] = images[i]
    labeldata[i] = labels[i]
  print("\n")
  return imagedata, labeldata

def _get_output_filename(dataset_dir, split_name):
  """Creates the output filename.

  Args:
    dataset_dir: The directory where the temporary files are stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/mnist_%s.tfrecord' % (dataset_dir, split_name)

def _add_to_tfrecord(images, labels, num_images, tfrecord_writer):
  """Loads data from the binary MNIST files and writes files to a TFRecord.

  Args:
    data_filename: The filename of the MNIST images.
    labels_filename: The filename of the MNIST labels.
    num_images: The number of images in the dataset.
    tfrecord_writer: The TFRecord writer to use for writing.
  """
  #images = _extract_images(data_filename, num_images)
  #labels = _extract_labels(labels_filename, num_images)

  shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
  with tf.Graph().as_default():
    image = tf.placeholder(dtype=tf.uint8, shape=shape)
    print(image)
    encoded_png = tf.image.encode_png(image)
    print(encoded_png)

    with tf.Session('') as sess:
      for j in range(num_images):
        sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
        sys.stdout.flush()

        png_string = sess.run(encoded_png, feed_dict={image: images[j]})

        example = dataset_utils.image_to_tfexample(
            png_string, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[j])
        tfrecord_writer.write(example.SerializeToString())

def run(dataset_dir, train_images, test_images):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  training_filename = _get_output_filename(dataset_dir, 'train')
  testing_filename = _get_output_filename(dataset_dir, 'test')

  if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  labelsAndFiles_train = get_labels_and_files(train_images)
  random.shuffle(labelsAndFiles_train)
  imagedata_train, labeldata_train = make_arrays(labelsAndFiles_train)
  print(np.reshape(imagedata_train[0], (28, 28)), labeldata_train[0])

  # First, process the training data:
  with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
    _add_to_tfrecord(imagedata_train, labeldata_train, len(labeldata_train), tfrecord_writer)

  labelsAndFiles_test = get_labels_and_files(test_images)
  random.shuffle(labelsAndFiles_test)
  imagedata_test, labeldata_test = make_arrays(labelsAndFiles_test)

  # Next, process the testing data:
  with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
    _add_to_tfrecord(imagedata_test, labeldata_test, len(labeldata_test), tfrecord_writer)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
  dataset_utils.write_label_file(labels_to_class_names, FLAGS.data_output)

  print('\nFinished !')

def main(_):
  if not FLAGS.train_images:
    raise ValueError('You must supply images train with --train_image')
  if not FLAGS.test_images:
    raise ValueError('You must supply images train with --test_images')
  if not FLAGS.data_output:
    raise ValueError('You must supply the dataset directory with --data_output')

  run(FLAGS.data_output,
      FLAGS.train_images,
      FLAGS.test_images)

if __name__ == '__main__':
  tf.app.run()
