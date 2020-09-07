from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import img_to_array, load_img


from typing import Optional, Dict, Iterable, Any, List
import functools
import os, math
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Some hyper-parameters:
batch_size  = 8          # Images per batch (reduce/increase according to the machine's capability)
num_epochs  = 300           # Max number of training epochs
random_seed = 42            # Seed for some random operations, for reproducibility

# print("TF version:", tf.__version__)
# print("Hub version:", hub.__version__)
# print(tf.config.list_physical_devices('GPU'))


""" Download and prepare tfds dataset """


class TFDS():
  """Download and process datasets from tensorflow dataset"""
  AVAILABLE_DATASETS = tfds.list_builders()

  def __init__(self, name:str, seed=1234, data_dir:Optional[str]=None) -> None:

    (self.train_ds, self.val_ds, self.test_ds), self.metadata = tfds.load(
      name=name,
      split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
      with_info=True,
      as_supervised=True,
      data_dir=data_dir)

    self.total_imgs = self.metadata.splits['train'].num_examples
    self.num_trainImgs = int(self.total_imgs * 0.8)
    self.num_valImgs = int(self.total_imgs * 0.1)
    self.num_testImgs = int(self.total_imgs * 0.1)
    self.img_shape = next(iter(self.train_ds))[0].shape
    self.get_label_name = self.metadata.features['label'].int2str
    self.seed=seed
    self.Preprocess = Preprocess(seed=self.seed)

  def get_dataset(self, num_epochs:int=300, batch_size:int=32,
                  input_shape:Iterable[int]=(32, 32, 3), seed=None):
    """[Generate train, val, test sets from tfds]

      Args:
          batch_size (int, optional): [number of images for each batch]. Defaults to 32.
          num_epochs (int, required): [number of epochs to run on experiment]. Defaults to 300.
          input_shape (tuple, optional): [shape of each image (h, w, c)]. Defaults to (32, 32, 3).
          seed ([type], optional): []. Defaults to None.

      Returns:
          [tuple]: [train, val, test sets]
    """ 
    train_prepare_data_fn = functools.partial(self.Preprocess.preprocess, input_shape=input_shape)
    test_prepare_data_fn = functools.partial(self.Preprocess.preprocess, augment=False, input_shape=input_shape)
    train_ds = (self.train_ds
                     .repeat(num_epochs)
                     .shuffle(10000, seed=seed)
                     .map(train_prepare_data_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                     .batch(batch_size)
                     .prefetch(tf.data.experimental.AUTOTUNE)
    )
    val_ds =  (self.val_ds
                     .repeat(num_epochs)
                     .map(test_prepare_data_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                     .batch(batch_size)
                     .prefetch(tf.data.experimental.AUTOTUNE)
    )
    test_ds = self.test_ds.map(test_prepare_data_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return (train_ds, val_ds, test_ds)

class Preprocess():
    """Preprocess images from tfds"""

    def __init__(self, seed):
        self.seed = seed
    
    @staticmethod
    def visualize(original, augmented):
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.title('Original image')
        plt.imshow(original)

        plt.subplot(1,2,2)
        plt.title('Augmented image')
        plt.imshow(augmented)

    def preprocess(self, image:List[float], label:Any, input_shape:Iterable[int], augment:bool=True):
      """[Perform augmentation on input image]

        Args:
            image (List[float]): [input image]
            label (Any): [label of the image]
            input_shape (Iterable[int]): [expected shape of the output image ]
            augment (bool, optional): [whether to augment or not]. Defaults to True.

        Returns:
            [type]: [tuple of image and label]
      """        
      image = tf.image.convert_image_dtype(image, tf.float32)

      if augment:
        # Randomly applied horizontal flip:
        image = tf.image.random_flip_left_right(image, seed=self.seed)

        # Random B/S changes:
        image = tf.image.random_brightness(image, max_delta=0.1, seed=self.seed)
        if input_shape[2] == 3:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=self.seed)
        image = tf.clip_by_value(image, 0.0, 1.0) # keeping pixel values in check

        # Random resize and random crop back to expected size:
        
        random_scale_factor = tf.random.uniform([1], minval=1., maxval=1.4, dtype=tf.float32, seed=self.seed)
        scaled_height = tf.cast(tf.cast(input_shape[0], tf.float32) * random_scale_factor, 
                                tf.int32)
        scaled_width = tf.cast(tf.cast(input_shape[1], tf.float32) * random_scale_factor, 
                                tf.int32)
        scaled_shape = tf.squeeze(tf.stack([scaled_height, scaled_width]))
        image = tf.image.resize(image, scaled_shape)
        image = tf.image.random_crop(image, input_shape, seed=self.seed)
      else:
        image = tf.image.resize(image, input_shape[:2])
      
      return (image, label)

class Evaluation():
  """ Eval utility functions"""
  def __init__(self):
    pass

  def load_image(self, image_path, size):
      """
      Load an image as a Numpy array.
      :param image_path:  Path of the image
      :param size:        Target size
      :return             Image array, normalized between 0 and 1
      """
      image = img_to_array(load_img(image_path, target_size=size)) / 255.
      return image


  def process_predictions(self, class_probabilities, class_readable_labels, k=5):
      """
      Process a batch of predictions from our estimator.
      :param class_probabilities:     Prediction results returned by the Keras classifier for a batch of data
      :param class_readable_labels:   List of readable-class labels, for display
      :param k:                       Number of top predictions to consider
      :return                         Readable labels and probabilities for the predicted classes
      """
      topk_labels, topk_probabilities = [], []
      for i in range(len(class_probabilities)):
          # Getting the top-k predictions:
          topk_classes = sorted(np.argpartition(class_probabilities[i], -k)[-k:])

          # Getting the corresponding labels and probabilities:
          topk_labels.append([class_readable_labels[predicted] for predicted in topk_classes])
          topk_probabilities.append(class_probabilities[i][topk_classes])

      return topk_labels, topk_probabilities


  def display_predictions(self, images, topk_labels, topk_probabilities, true_labels):
      """
      Plot a batch of predictions.
      :param images:                  Batch of input images
      :param topk_labels:             String labels of predicted classes
      :param topk_probabilities:      Probabilities for each class
      """
      num_images = len(images)
      num_images_sqrt = np.sqrt(num_images)
      plot_cols = plot_rows = int(np.ceil(num_images_sqrt))

      figure = plt.figure(figsize=(13, 10))
      grid_spec = gridspec.GridSpec(plot_cols, plot_rows)

      for i in range(num_images):
          img, pred_labels, pred_proba, true_label = images[i], topk_labels[i], topk_probabilities[i], true_labels[i]
          # Shortening the labels to better fit in the plot:
          pred_labels = [label.split(',')[0][:20] for label in pred_labels]

          grid_spec_i = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=grid_spec[i],
                                                        hspace=0.1)

          # Drawing the input image:
          ax_img = figure.add_subplot(grid_spec_i[:2])
          ax_img.axis('off')
          ax_img.imshow(img)
          ax_img.set_title(f'True Label: {true_label}')
          ax_img.autoscale(tight=True)

          # Plotting a bar chart for the predictions:
          ax_pred = figure.add_subplot(grid_spec_i[2])
          ax_pred.spines['top'].set_visible(False)
          ax_pred.spines['right'].set_visible(False)
          ax_pred.spines['bottom'].set_visible(False)
          ax_pred.spines['left'].set_visible(False)
          y_pos = np.arange(len(pred_labels))
          ax_pred.barh(y_pos, pred_proba, align='center')
          ax_pred.set_yticks(y_pos)
          ax_pred.set_yticklabels(pred_labels)
          ax_pred.invert_yaxis()

      plt.tight_layout()
      plt.show()

if __name__ == "__main__":
    pass