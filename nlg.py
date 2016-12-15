from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import nlg_data_untils as data_utils
import seq2seq_model

import subprocess
import stat


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("send_vocabulary_size", 10000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("gen_vocabulary_size", 10000, "French vocabulary size.")
tf.app.flags.DEFINE_integer("feat_vocabulary_size", 10000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# _buckets = [(5, 10,10), (10, 15,15), (20,25, 25),(40,50,50)]
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

# def read_data(send_path, gen_path,feat_path, max_size=None):
#   """Read data from source and target files and put into buckets.
#
#   Args:
#     source_path: path to the files with token-ids for the source language.
#     target_path: path to the file with token-ids for the target language;
#       it must be aligned with the source file: n-th line contains the desired
#       output for n-th line from the source_path.
#     max_size: maximum number of lines to read, all other will be ignored;
#       if 0 or None, data files will be read completely (no limit).
#
#   Returns:
#     data_set: a list of length len(_buckets); data_set[n] contains a list of
#       (source, target) pairs read from the provided data files that fit
#       into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
#       len(target) < _buckets[n][1]; source and target are lists of token-ids.
#   """
#   data_set = [[] for _ in _buckets]
#   print('data_file:',send_path,gen_path,feat_path)
#   with tf.gfile.GFile(send_path, mode="r") as send_file:
#     with tf.gfile.GFile(gen_path, mode="r") as gen_file:
#         with tf.gfile.GFile(feat_path, mode="r") as feat_file:
#           send, gen,feat = send_file.readline(), gen_file.readline(),feat_file.readline()
#           counter = 0
#           while send and gen and feat and (not max_size or counter < max_size):
#             counter += 1
#             # if counter % 100000 == 0:
#             #   print("  reading data line %d" % counter)
#             #   sys.stdout.flush()
#             send_ids = [int(x) for x in send.split()]
#             gen_ids = [int(x) for x in gen.split()]
#             feat_ids = [int(x) for x in feat.split()]
#             gen_ids.append(data_utils.EOS_ID)
#             feat_ids.append(data_utils.EOS_ID)
#             for bucket_id, (send_size, gen_size,feat_size) in enumerate(_buckets):
#               if len(send_ids) < send_size and len(gen_ids) < gen_size and len(feat_ids) < feat_size:
#                 data_set[bucket_id].append([send_ids, [gen_ids,feat_ids]])
#                 break
#             send, gen, feat = send_file.readline(), gen_file.readline(), feat_file.readline()
#   return data_set
def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

def get_bleu(filename,targ):
    #data/data/valid.en < data/data/valid.es > result.txt
    #BLEU = 0.95, 4.4 / 0.9 / 0.5 / 0.4(BP=1.000, ratio=1.177, hyp_len=35759339, ref_len=30376256)
    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/multi-bleu.perl'
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                            _conlleval,filename , ' < ',targ],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()))
    out = None


    for line in stdout.split(','):
        if 'BLEU =' in line:
            return  line.replace('BLEU =','')
    if out == None:
        return '0'

def save_result(l,filename,mode):
    out = ''
    for i in l:
        if type(i) is list:
            out += ' '.join(i) + '\n'
        else:
            out += i + '\n'
    with open(filename, mode) as f:
    # print(out)
        f.writelines(out)  # remove the ending \n on last line

def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.send_vocabulary_size,
      FLAGS.gen_vocabulary_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  """Train a en->fr translation model using WMT data."""
  # Prepare WMT data.
  print("Preparing nlg data in %s" % FLAGS.data_dir)
  send_train, gen_train,feat_train, send_dev, gen_dev,feat_dev, _, _,_ = data_utils.prepare_nlg_data(
      FLAGS.data_dir, FLAGS.send_vocabulary_size, FLAGS.gen_vocabulary_size,FLAGS.feat_vocabulary_size)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(send_dev, feat_dev)
    train_set = read_data(send_train, feat_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    # Load vocabularies.
    send_vocab_path = os.path.join(FLAGS.data_dir, "send_vocab_path%d.txt" % FLAGS.send_vocabulary_size)
    gen_vocab_path = os.path.join(FLAGS.data_dir, "en_vocab_path%d.txt" % FLAGS.gen_vocabulary_size)
    feat_vocab_path = os.path.join(FLAGS.data_dir, "feat_vocab_path%d.txt" % FLAGS.feat_vocabulary_size)

    # en_vocab_path = os.path.join(FLAGS.data_dir,
    #                              "vocab%d.en" % FLAGS.en_vocab_size)
    # fr_vocab_path = os.path.join(FLAGS.data_dir,
    #                              "vocab%d.es" % FLAGS.fr_vocab_size)
    send_vocab, rev_send_vocab = data_utils.initialize_vocabulary(send_vocab_path)
    _, rev_gen_vocab = data_utils.initialize_vocabulary(gen_vocab_path)
    _, rev_feat_vocab = data_utils.initialize_vocabulary(feat_vocab_path)

    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "nlg.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        sentence_res,sentence_res_input = [],[]
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          # encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          #     {bucket_id: [(token_ids, [])]}, bucket_id)
          _, eval_loss, output_logits= model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          # print(output_logits)
          tmp_logits_arr,outputs,tmp_input = [],None,[]

          # for input in encoder_inputs:
          #     tmp_input = [int(np.argmax(logit)) for logit in input]
          #     sentence_res_input.append(' '.join([tf.compat.as_str(rev_en_vocab[input]) for input in tmp_input]))

          for logits in output_logits:
            outputs = [int(np.argmax(logit)) for logit in logits]
            tmp_logits_arr.append(outputs)
          # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            sentence_res.append(" ".join([tf.compat.as_str(rev_feat_vocab[output]) for output in outputs]))
        print('sentence_res len',len(sentence_res),'sentence_res_input len',len(sentence_res_input))
        save_result(sentence_res_input, FLAGS.data_dir + 'nlg_valid_comp_result.txt', 'w')
        save_result(sentence_res,FLAGS.data_dir+'nlg_valid_result.txt','w')
        # bleu = get_bleu(FLAGS.data_dir+'nlg_valid_result.txt',FLAGS.data_dir+'data/valid.es')
        # print("  eval: bleu " + bleu)
        sys.stdout.flush()

def test():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # # Load vocabularies.
    # en_vocab_path = os.path.join(FLAGS.data_dir,
    #                              "vocab%d.en" % FLAGS.en_vocab_size)
    # fr_vocab_path = os.path.join(FLAGS.data_dir,
    #                              "vocab%d.es" % FLAGS.fr_vocab_size)
    # en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    # _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    send_vocab_path = os.path.join(FLAGS.data_dir, "send_vocab_path%d.txt" % FLAGS.send_vocabulary_size)
    gen_vocab_path = os.path.join(FLAGS.data_dir, "en_vocab_path%d.txt" % FLAGS.gen_vocabulary_size)
    feat_vocab_path = os.path.join(FLAGS.data_dir, "feat_vocab_path%d.txt" % FLAGS.feat_vocabulary_size)

    send_vocab, rev_send_vocab = data_utils.initialize_vocabulary(send_vocab_path)
    _, rev_gen_vocab = data_utils.initialize_vocabulary(gen_vocab_path)
    _, rev_feat_vocab = data_utils.initialize_vocabulary(feat_vocab_path)

    # Decode from standard input
    sentence_es_arr = []
    with open(FLAGS.data_dir+'NLG_data/test.txt','r') as infile:
        for sentence in infile:
          # Get token-ids for the input sentence.
          token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), send_vocab)
          # Which bucket does it belong to?
          bucket_id = len(_buckets) - 1
          for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
              bucket_id = i
              break

          # Get a 1-element batch to feed the sentence to the model.
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)
          # Get output logits for the sentence.
          _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)
          # This is a greedy decoder - outputs are just argmaxes of output_logits.
          outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

          # If there is an EOS symbol in outputs, cut them at that point.
          if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
          # Print out French sentence corresponding to outputs.
          # print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
          sentence_es_arr.append(" ".join([tf.compat.as_str(rev_feat_vocab[output]) for output in outputs]))
    save_result(sentence_es_arr,FLAGS.data_dir+'valid_result_bleu.txt','w')
    bleu = get_bleu(FLAGS.data_dir + 'valid_result_bleu.txt', FLAGS.data_dir + 'data/valid.es')
    print("  eval: bleu " + bleu)

def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    send_vocab_path = os.path.join(FLAGS.data_dir, "send_vocab_path%d.txt" % FLAGS.send_vocabulary_size)
    gen_vocab_path = os.path.join(FLAGS.data_dir, "en_vocab_path%d.txt" % FLAGS.gen_vocabulary_size)
    feat_vocab_path = os.path.join(FLAGS.data_dir, "feat_vocab_path%d.txt" % FLAGS.feat_vocabulary_size)

    send_vocab, rev_send_vocab = data_utils.initialize_vocabulary(send_vocab_path)
    _, rev_gen_vocab = data_utils.initialize_vocabulary(gen_vocab_path)
    _, rev_feat_vocab = data_utils.initialize_vocabulary(feat_vocab_path)

    # Decode from standard input
    sentence_es_arr = []
    with open(FLAGS.data_dir+'NLG_data/test.txt','r') as infile:
        for sentence in infile:
          # Get token-ids for the input sentence.
          token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), send_vocab)
          # Which bucket does it belong to?
          bucket_id = len(_buckets) - 1
          for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
              bucket_id = i
              break

          # Get a 1-element batch to feed the sentence to the model.
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)
          # Get output logits for the sentence.
          _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)
          # This is a greedy decoder - outputs are just argmaxes of output_logits.
          outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

          # If there is an EOS symbol in outputs, cut them at that point.
          if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
          # Print out French sentence corresponding to outputs.
          # print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
          sentence_es_arr.append(" ".join([tf.compat.as_str(rev_feat_vocab[output]) for output in outputs]))
    save_result(sentence_es_arr,FLAGS.data_dir+'nlg_test_result.txt','w')


def main(_):
  if FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()