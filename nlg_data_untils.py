from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import os.path
import json

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def load_json_train_data(data_dir,filename):
    file = os.path.join(data_dir,filename)
    with open(file) as data_file:
        data = json.load(data_file)
    sent_path = os.path.join(data_dir,filename+'_sent.txt')
    gen_path = os.path.join(data_dir,filename+'_gen.txt')
    feat_path = os.path.join(data_dir,filename+'_feat.txt')
    with open(sent_path,'w') as f_s:
        with open(gen_path,'w') as f_g:
            with open(feat_path, 'w') as f_f:
                for d in data:
                    f_s.write(d[0]+'\n')
                    f_g.write(d[1] + '\n')
                    f_f.write(d[2] + '\n')
    return sent_path,gen_path,feat_path

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                        tokenizer=None, normalize_digits=True):


    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")

def sentence_to_token_ids(sentence, vocabulary,
        tokenizer=None, normalize_digits=True):

    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def data_to_token_ids(data_path, target_path, vocabulary_path,
                    tokenizer=None, normalize_digits=True):

    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                                      tokenizer, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_nlg_data(data_dir, send_vocabulary_size, gen_vocabulary_size,feat_vocabulary_size, tokenizer=None):
    train_path = os.path.join(data_dir,'NLG_data/')
    # dev_path = os.path.join(data_dir,'NLG_data/')

    send_path,gen_path,feat_path = load_json_train_data(train_path,'train.json')
    # Create vocabularies of the appropriate sizes.
    send_vocab_path = os.path.join(data_dir, "send_vocab_path%d.txt" % send_vocabulary_size)
    gen_vocab_path = os.path.join(data_dir, "en_vocab_path%d.txt" % gen_vocabulary_size)
    feat_vocab_path = os.path.join(data_dir, "feat_vocab_path%d.txt" % feat_vocabulary_size)
    create_vocabulary(send_vocab_path, send_path, send_vocabulary_size, tokenizer)
    create_vocabulary(gen_vocab_path, gen_path, gen_vocabulary_size, tokenizer)
    create_vocabulary(feat_vocab_path, feat_path, feat_vocabulary_size, tokenizer)

    # Create token ids for the training data.
    send_train_ids_path = train_path + ("train.ids%d.send" % send_vocabulary_size)
    gen_train_ids_path = train_path + ("train.ids%d.gen" % gen_vocabulary_size)
    feat_train_ids_path = train_path + ("train.ids%d.feat" % feat_vocabulary_size)
    data_to_token_ids(send_path, send_train_ids_path, send_vocab_path, tokenizer)
    data_to_token_ids(gen_path , gen_train_ids_path, gen_vocab_path, tokenizer)
    data_to_token_ids(feat_path, feat_train_ids_path, feat_vocab_path, tokenizer)

    # Create token ids for the development data.
    send_path, gen_path, feat_path = load_json_train_data(train_path, 'valid.json')
    send_dev_ids_path = train_path + ("dev.ids%d.send" % send_vocabulary_size)
    gen_dev_ids_path = train_path + ("dev.ids%d.gen" % gen_vocabulary_size)
    feat_dev_ids_path = train_path + ("dev.ids%d.feat" % feat_vocabulary_size)
    data_to_token_ids(send_path, send_dev_ids_path, send_vocab_path, tokenizer)
    data_to_token_ids(gen_path, gen_dev_ids_path, gen_vocab_path, tokenizer)
    data_to_token_ids(feat_path, feat_dev_ids_path, feat_vocab_path, tokenizer)


    return (send_train_ids_path, gen_train_ids_path,feat_train_ids_path,
            send_dev_ids_path, gen_dev_ids_path,feat_dev_ids_path,
            send_vocab_path, gen_vocab_path,feat_vocab_path)

