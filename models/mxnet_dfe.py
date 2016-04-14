#get_ipython().magic(u'matplotlib inline')
#import mxnet as mx
from graphlab import mxnet as mx
import logging
import numpy as np
import os
import pdb
#import ipdb
from skimage import io, transform
import graphlab as gl

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
dfe = "inception_21k"
#dfe = "Inception"

# setting up model specs
if dfe=="Inception":
  model_dir = "Inception"
  prefix = "Inception/Inception_BN"
  num_round = 39  # num of epocs for training the model
  mean_img = mx.nd.load("Inception/mean_224.nd")["mean_img"]

if dfe=="inception_21k":
  model_dir = "./inception_21k"
  prefix = "./inception_21k/Inception"
  num_round = 9
  mean_img = 117 * np.ones((3, 224, 224))

def load_model(model_dir, prefix, num_round=39, batchsize=1):
  model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=batchsize)
  synset_file = os.path.join(model_dir, 'synset.txt')
  synset = [l.strip() for l in open(synset_file).readlines()]

  return model, synset

model, synset = load_model(model_dir, prefix, num_round=num_round)

def PreprocessImage(path, show_img=False):
  # crop center, subtract mean, and then extract features 
  img = io.imread(path)
  print("Original Image Shape: ", img.shape)
  # we crop image from center
  short_egde = min(img.shape[:2])
  yy = int((img.shape[0] - short_egde) / 2)
  xx = int((img.shape[1] - short_egde) / 2)
  crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
  # resize to 224, 224
  resized_img = transform.resize(crop_img, (224, 224))
  if show_img:
      io.imshow(resized_img)
  # convert to numpy.ndarray
  sample = np.asarray(resized_img) * 256
  # swap axes to make image from (224, 224, 4) to (3, 224, 224)
  sample = np.swapaxes(sample, 0, 2)
  sample = np.swapaxes(sample, 1, 2)
  # sub mean 
  if isinstance(mean_img, type(sample)):
    normed_img = sample - mean_img
  else:
    normed_img = sample - mean_img.asnumpy()
  normed_img.resize(1, 3, 224, 224)
  return normed_img

# Get preprocessed batch (single image batch)
def mxnet_transform(path, model, synset):
  if isinstance(path, str):
    batch = PreprocessImage(path, False)
  elif isinstance(path, gl.data_structures.sframe.SFrame):
    path = path.head(6000)
    path['resized_image'] = gl.image_analysis.resize(path['image'], 224, 224, 3)
    batch = mx.io.SFrameIter(sframe=path, data_field=['resized_image'], batch_size=100)
  # batch = map(lambda x: PreprocessImage(x), path)
  # Get prediction probability of 1000 classes from model
  prob = model.predict(batch)[0]
  # Argsort, get prediction index from largest prob to lowest
  pred = np.argsort(prob)[::-1]
  # Get top1 label
  top1 = synset[pred[0]]
  print("Top1: ", top1)
  # Get top5 label
  top5 = [synset[pred[i]] for i in range(5)]
  print("Top5: ", top5)
  internals = model.symbol.get_internals()
  # get feature layer symbol out of internals
  fea_symbol = internals["global_pool_output"]
  # Make a new model by using an internal symbol. We can reuse all parameters from model we trained before
  # In this case, we must set ```allow_extra_params``` to True, Because we don't need params from FullyConnected symbol
  pdb.set_trace()
  feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=1,arg_params=model.arg_params, aux_params=model.aux_params,allow_extra_params=True)
  # predict feature
  feature = feature_extractor.predict(batch)
  feature = feature.reshape(path.__len__(), -1)
  path["feature"] = feature
  pdb.set_trace()
  #print(global_pooling_feature.shape)
  return feature, top1, top5

if __name__=="__main__":
  path='./living_room.jpg'
  path = gl.load_sframe("../../py-faster-rcnn/IKEA/cata_db_image.gl")
  feature, top1, top5 = mxnet_transform(path, model, synset)