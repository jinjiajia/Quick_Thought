# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts a set of text files to TFRecord format with Example protos.

Each Example proto in the output contains the following fields:

  decode_pre: list of int64 ids corresponding to the "previous" sentence.
  encode: list of int64 ids corresponding to the "current" sentence.
  decode_post: list of int64 ids corresponding to the "post" sentence.

In addition, the following files are generated:

  vocab.txt: List of "<word> <id>" pairs, where <id> is the integer
             encoding of <word> in the Example protos.
  word_counts.txt: List of "<word> <count>" pairs, where <count> is the number
                   of occurrences of <word> in the input files.

The vocabulary of word ids is constructed from the top --num_words by word
count. All other words get the <unk> word id.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os,re,codecs
import jieba
import pandas as pd
import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_files",'train_data.xlsx',
                       "sentence not tokenized.")

tf.flags.DEFINE_string("vocab_file", "",
                       "(Optional) existing vocab file. Otherwise, a new vocab "
                       "file is created and written to the output directory. "
                       "The file format is a list of newline-separated words, "
                       "where the word id is the corresponding 0-based index "
                       "in the file.")

tf.flags.DEFINE_string("output_dir", 'output1', "Output directory.")

tf.flags.DEFINE_integer("train_output_shards", 100,
                        "Number of output shards for the training set.")

tf.flags.DEFINE_integer("validation_output_shards", 1,
                        "Number of output shards for the validation set.")

tf.flags.DEFINE_integer("num_validation_sentences", 10000,
                        "Number of output shards for the validation set.")

tf.flags.DEFINE_integer("num_words", 20000,
                        "Number of words to include in the output.")

tf.flags.DEFINE_integer("max_sentences", 0,
                        "If > 0, the maximum number of sentences to output.")

tf.flags.DEFINE_integer("max_sentence_length", 100,
                        "If > 0, exclude sentences whose encode, decode_pre OR"
                        "decode_post sentence exceeds this length.")

tf.flags.DEFINE_boolean("case_sensitive", False,
                        "Use case sensitive vocabulary")

tf.logging.set_verbosity(tf.logging.INFO)

UNK = "<unk>"
UNK_ID = 0

def _build_tokenfile(input_files):
    """creat fenci files"""
    jieba.load_userdict("jiebauserdict100w.txt")
    tf.logging.info("Creating sentence tokenized file")
    data_xls = pd.read_excel(input_files[0], usecols='B,E')  # skiprow=[0]去除第一行.usecols的第一列是B列，序号列不计入
    newdata = data_xls.values  # to numpy
    alltext = newdata[:, 0]
    alllab = np.array(newdata[:, 1], np.int16)

    sentence = [' '.join(jieba.lcut(''.join(text.split()))) for text in alltext]
    print(sentence)

    tokenized_file = os.path.join(FLAGS.output_dir, "tokenized.txt")
    with tf.gfile.FastGFile(tokenized_file, "w") as f:
        f.write("\n".join(sentence))
    tf.logging.info("Wrote sentence to %s", sentence)

def _build_vocabulary(input_file):
  """Loads or builds the model vocabulary.

  Args:
    input_files: List of pre-tokenized input .txt files.

  Returns:
    vocab: A dictionary of word to id.
  """
  if FLAGS.vocab_file:
    tf.logging.info("Loading existing vocab file.")
    vocab = collections.OrderedDict()
    with tf.gfile.GFile(FLAGS.vocab_file, mode="r") as f:
      for i, line in enumerate(f):
        word = line.decode("utf-8").strip()
        if word in vocab:
          print('Duplicate word:', word)
        #assert word not in vocab, "Attempting to add word twice: %s" % word
        vocab[word] = i
    tf.logging.info("Read vocab of size %d from %s",
                    len(vocab), FLAGS.vocab_file)
    return vocab

  tf.logging.info("Creating vocabulary.")
  num = 0
  wordcount = collections.Counter()
  for sentence in tf.gfile.FastGFile(input_file):
      wordcount.update(sentence.split())

      num += 1
      if num % 1000000 == 0:
        tf.logging.info("Processed %d sentences", num)

  tf.logging.info("Processed %d sentences total", num)

  words = list(wordcount.keys())
  freqs = list(wordcount.values())
  sorted_indices = np.argsort(freqs)[::-1]

  vocab = collections.OrderedDict()
  vocab[UNK] = UNK_ID
  for w_id, w_index in enumerate(sorted_indices[0:FLAGS.num_words - 2]):
    vocab[words[w_index]] = w_id + 1  # 0: <unk>

  tf.logging.info("Created vocab with %d words", len(vocab))

  vocab_file = os.path.join(FLAGS.output_dir, "vocab.txt")
  with tf.gfile.FastGFile(vocab_file, "w") as f:
    f.write("\n".join(vocab.keys()))
  tf.logging.info("Wrote vocab file to %s", vocab_file)

  word_counts_file = os.path.join(FLAGS.output_dir, "word_counts.txt")
  with tf.gfile.FastGFile(word_counts_file, "w") as f:
    for i in sorted_indices:
      f.write("%s %d\n" % (words[i], freqs[i]))
  tf.logging.info("Wrote word counts file to %s", word_counts_file)

  return vocab


def _int64_feature(value):
  """Helper for creating an Int64 Feature."""
  return tf.train.Feature(int64_list=tf.train.Int64List(
      value=[int(v) for v in value]))


def _sentence_to_ids(sentence, vocab):
  """Helper for converting a sentence (list of words) to a list of ids."""
  if FLAGS.case_sensitive:
    ids = [vocab.get(w, UNK_ID) for w in sentence]
  else:
    ids = [vocab.get(w.lower(), UNK_ID) for w in sentence]
  return ids


#def _create_serialized_example(predecessor, current, successor, vocab):
def _create_serialized_example(current, vocab):
  """Helper for creating a serialized Example proto."""
  #example = tf.train.Example(features=tf.train.Features(feature={
  #    "decode_pre": _int64_feature(_sentence_to_ids(predecessor, vocab)),
  #    "encode": _int64_feature(_sentence_to_ids(current, vocab)),
  #    "decode_post": _int64_feature(_sentence_to_ids(successor, vocab)),
  #}))
  example = tf.train.Example(features=tf.train.Features(feature={
      "features": _int64_feature(_sentence_to_ids(current, vocab)),
  }))
  #example = tf.train.Example(features=tf.train.Features(feature=
  #    _int64_feature(_sentence_to_ids(current, vocab)),
  #))

  return example.SerializeToString()

def _process_input_file(filename, vocab, stats):
  """Processes the sentences in an input file.

  Args:
    filename: Path to a pre-tokenized input .txt file.
    vocab: A dictionary of word to id.

  Returns:
    processed: A list of serialized Example protos
  """
  
  tf.logging.info("Processing input file: %s", filename)
  processed = []

  for sentence_str in tf.gfile.FastGFile(filename):
    print("sentence_str",sentence_str)
    sentence_tokens = sentence_str.split()

    sentence_tokens = sentence_tokens[:FLAGS.max_sentence_length]

    serialized = _create_serialized_example(sentence_tokens, vocab)
    print("serialized",serialized)
    processed.append(serialized)
    stats.update(["sentence_count"])
  print("processed",processed)
  print("stats",stats)

  tf.logging.info("Completed processing file %s", filename)
  return processed

def _write_shard(filename, dataset, indices):
  """Writes a TFRecord shard."""
  with tf.python_io.TFRecordWriter(filename) as writer:
    for j in indices:
      writer.write(dataset[j])


def _write_dataset(name, dataset, indices, num_shards):
  """Writes a sharded TFRecord dataset.

  Args:
    name: Name of the dataset (e.g. "train").
    dataset: List of serialized Example protos.
    indices: List of indices of 'dataset' to be written.
    num_shards: The number of output shards.
  """
  tf.logging.info("Writing dataset %s", name)
  borders = np.int32(np.linspace(0, len(indices), num_shards + 1))
  print(name, dataset, indices, num_shards)
  print("borders",borders)
  for i in range(num_shards):
    filename = os.path.join(FLAGS.output_dir, "%s-%.5d-of-%.5d" % (name, i,
                                                                   num_shards))
    shard_indices = indices[borders[i]:borders[i + 1]]
    print("shard_indices",shard_indices)
    _write_shard(filename, dataset, shard_indices)
    tf.logging.info("Wrote dataset indices [%d, %d) to output shard %s",
                    borders[i], borders[i + 1], filename)
  tf.logging.info("Finished writing %d sentences in dataset %s.",
                  len(indices), name)

def export_word2vec_vectors(vocab, word2vec_dir, trimmed_filename):
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.save(trimmed_filename, embeddings)
    print('saving npy word2vec')

def main(unused_argv):
  if not FLAGS.input_files:
    raise ValueError("--input_files is required.")
  if not FLAGS.output_dir:
    raise ValueError("--output_dir is required.")

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = []
  for pattern in FLAGS.input_files.split(","):
    match = tf.gfile.Glob(pattern)
    if not match:
      raise ValueError("Found no files matching %s" % pattern)
    input_files.extend(match)
  tf.logging.info("Found %d input files.", len(input_files))

  token_file = os.path.join(FLAGS.output_dir, "tokenized.txt")
  if not os.path.exists(token_file):
    _build_tokenfile(input_files)
  vocab = _build_vocabulary(token_file)
  print(vocab)

  tf.logging.info("Generating dataset.")
  stats = collections.Counter()
  dataset = []
  dataset.extend(_process_input_file(token_file, vocab, stats))
  print(dataset)
  # for filename in input_files:
  #   dataset.extend(_process_input_file(filename, vocab, stats))
  #   if FLAGS.max_sentences and stats["sentence_count"] >= FLAGS.max_sentences:
  #     break

  tf.logging.info("Generated dataset with %d sentences.", len(dataset))
  #for k, v in stats.items():
  #  tf.logging.info("%s: %d", k, v)

  #tf.logging.info("Shuffling dataset.")
  #np.random.seed(123)
  #shuffled_indices = np.random.permutation(len(dataset))
  #val_indices = shuffled_indices[:FLAGS.num_validation_sentences]
  #train_indices = shuffled_indices[FLAGS.num_validation_sentences:]
  indices = range(len(dataset))
  val_indices = indices[:FLAGS.num_validation_sentences]
  train_indices = indices[FLAGS.num_validation_sentences:]

  _write_dataset("train", dataset, train_indices, FLAGS.train_output_shards)
  _write_dataset("validation", dataset, val_indices,
                 FLAGS.validation_output_shards)


  vector_word_npy=os.path.join(FLAGS.output_dir, 'embeddings.npy')
  word2vec_dir='100W-word2vec.txt'
  vocab_file = os.path.join(FLAGS.output_dir, "vocab.txt")

  if not os.path.exists(vector_word_npy):
      words = open(vocab_file)  # io.textiowrapper
      words = list(words)  # list with \n in every word
      cwords = []
      for word in words:
          cwords.append(re.sub(r'\n', '', word))  # delete \n
      word_to_id = dict(zip(cwords, range(len(cwords))))
      export_word2vec_vectors(word_to_id, word2vec_dir, vector_word_npy)

if __name__ == "__main__":
  tf.app.run()
