from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd
from src import configuration
from src import encoder_manager
import json
import tensorflow as tf
# Set paths to the model.
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("Glove_path", './data/cpws', "Path to Glove dictionary")
tf.flags.DEFINE_string("model_config", '../model_configs/train.json', "Model configuration json")
tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")
tf.flags.DEFINE_string("output_dir", './data/cpws', "Output directory.")
# VOCAB_FILE = "./data/output/vocab.txt"
# EMBEDDING_MATRIX_FILE = "./data/embeddings.npy"
# CHECKPOINT_PATH = "./model/train/model.ckpt-7154"
# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.
MR_DATA_DIR = os.path.join(FLAGS.output_dir, "test.txt")
# MR_DATA_DIR1 = "./data/output1/tokenized.txt"
# Set up the encoder. Here we are using a single unidirectional model.
# To use a bidirectional model as well, call load_model() again with
# configuration.model_config(bidirectional_encoder=True) and paths to the
# bidirectional model's files. The encoder will use the concatenation of
# all loaded models.
with open(FLAGS.model_config) as json_config_file:
    model_config = json.load(json_config_file)

model_config = configuration.model_config(model_config, mode="encode")
encoder = encoder_manager.EncoderManager()
encoder.load_model(model_config)

# Load the movie review dataset.
data = []
with open(MR_DATA_DIR, 'r') as f:#rt-polarity.neg
  data.extend([line.strip() for line in f])
# with open(MR_DATA_DIR1, 'r') as f:#rt-polarity.neg
#   data.extend([line.strip() for line in f])
# with open(os.path.join(MR_DATA_DIR, 'rt-polarity.pos'), 'rb') as f:
#   data.extend([line.decode('latin-1').strip() for line in f])

# Generate Skip-Thought Vectors for each sentence in the dataset.

encodings = encoder.encode(data)

# Define a helper function to generate nearest neighbors.
LABEL_DIR = os.path.join(FLAGS.output_dir, "testlabel.txt")
RESULT_DIR = os.path.join(FLAGS.output_dir, "testresult.txt")
fl = open(LABEL_DIR,'r',encoding='utf-8')
label = [line.strip() for line in fl.readlines()]
f=open(RESULT_DIR,'w',encoding='utf-8')
def get_nn(ind, num=10):
  encoding = encodings[ind]
  scores = sd.cdist([encoding], encodings, "cosine")[0]
  sorted_ids = np.argsort(scores)
  print("Sentence:", data[ind], label[ind])
  f.write(data[ind]+' '+label[ind]+'\n')
  print("\nNearest neighbors:")
  acc=0
  for i in range(1, num + 1):
    if label[sorted_ids[i]]==label[ind]:
      acc+=1
    print(" %d. %s (%.3f) %s" %
          (i, data[sorted_ids[i]], scores[sorted_ids[i]], label[sorted_ids[i]]))

    f.write(str(i)+data[sorted_ids[i]]+str(scores[sorted_ids[i]])+label[sorted_ids[i]])

    f.write('\n')
    print('acc:',float(acc/num))
  return float(acc/num)
  # Compute nearest neighbors of the first sentence in the dataset.
# get_nn(0)
# get_nn(1)
f_acc=0.0
for i in range(len(data)):
# for i in range(100):
    acc=get_nn(i)
    f_acc+=acc
print("f_acc:",f_acc/len(data))
f.write(str(f_acc/len(data)))
