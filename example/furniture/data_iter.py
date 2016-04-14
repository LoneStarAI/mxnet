import sys
sys.path.insert(0, "../../python")
#import mxnet as mx
import mxnet as mx

dataiter = mx.io.ImageRecordIter(
  # Utility Parameter
  # Optional
  # Name of the data, should match the name of the data input of the network
  # data_name='data',
  # Utility Parameter
  # Optional
  # Name of the label, should match the name of the label parameter of the network.
  # Usually, if the loss layer is named 'foo', then the label input has the name
  # 'foo_label', unless overwritten
  # label_name='softmax_label',
  # Dataset Parameter
  # Impulsary
  # indicating the data file, please check the data is already there
  path_imgrec="furniture.rec",
  # Dataset Parameter
  # Impulsary
  # indicating the image size after preprocessing
  data_shape=(3,224,224),
  # Batch Parameter
  # Impulsary
  # tells how many images in a batch
  batch_size=100,
  # Augmentation Parameter
  # Optional
  # when offers mean_img, each image will substract the mean value at each pixel
  #mean_img="data/cifar/cifar10_mean.bin",
  # Augmentation Parameter
  # Optional
  # randomly crop a patch of the data_shape from the original image
  rand_crop=True,
  # Augmentation Parameter
  # Optional
  # randomly mirror the image horizontally
  rand_mirror=True,
  # Augmentation Parameter
  # Optional
  # randomly shuffle the data
  shuffle=True,
  # Backend Parameter
  # Optional
  # Preprocessing thread number
  preprocess_threads=4,
  # Backend Parameter
  # Optional
  # Prefetch buffer size
  prefetch_buffer=1)
