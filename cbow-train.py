import random
import os
import numpy as np
import tensorflow as tf

from config import logger
from TextUtils import TextProcessing
from LanguageModels.CBOW import CBOW
from zipfile import ZipFile

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_LOCATION = "data/"
DATASET_ZIP_FILE = "text8.zip"
DATASET_FILE = DATASET_ZIP_FILE.split(".")[0]

logger.info("Checking dataset location.")
if not (
        os.path.exists(os.path.join(CURRENT_DIR, DATASET_LOCATION)) and
        os.path.exists(os.path.join(CURRENT_DIR, DATASET_LOCATION, DATASET_ZIP_FILE))
):
    raise FileNotFoundError(f"Dataset '{DATASET_ZIP_FILE}' doesn't exists.")


logger.info("Extracting dataset compressed file.")
if os.path.exists(os.path.join(CURRENT_DIR, DATASET_LOCATION, DATASET_ZIP_FILE)) and \
        not os.path.exists(os.path.join(CURRENT_DIR, DATASET_LOCATION, DATASET_FILE)):

    with ZipFile(os.path.join(CURRENT_DIR, DATASET_LOCATION, DATASET_ZIP_FILE)) as _data_zip_file:
        _data_zip_file.extractall(
            os.path.join(CURRENT_DIR, DATASET_LOCATION)
        )

with open(os.path.join(CURRENT_DIR, DATASET_LOCATION, DATASET_FILE), "r") as _dataset_text_file:
    dataset_text = _dataset_text_file.read()

dataset_text = TextProcessing.replace_punctuation(dataset_text)
dataset_tokens = TextProcessing.remove_low_occurrence_words(TextProcessing.tokenize(dataset_text))

dictionary, rev_dictionary, word_index = TextProcessing.create_word_map(dataset_tokens)
cbow = CBOW(word_index)

# batch, label = cbow.samples(100, 5)
batch_length = 128
embedding_size = 128
skip_window = 1
num_skips = 2
num_steps = 100001

tensor_flow_graph = tf.Graph()
with tensor_flow_graph.as_default():
    validation_cnt = 16
    validation_dict = 100
    validation_words = np.array(
        random.sample(range(validation_dict), validation_cnt // 2)
    )
    validation_words = np.append(
        validation_words, random.sample(range(1000, 1000 + validation_dict), validation_cnt // 2)
    )

    train_dataset = tf.placeholder(tf.int32, shape=[batch_length, 2 * skip_window])
    train_labels = tf.placeholder(tf.int32, shape=[batch_length, 1])
    validation_data = tf.constant(validation_words, dtype=tf.int32)

    vocabulary_size = len(rev_dictionary)

    word_embed = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
    )

    context_embeddings = []
    for i in range(2 * skip_window):
        context_embeddings.append(
            tf.nn.embedding_lookup(word_embed, train_dataset[:, i])
        )

    embedding = tf.reduce_mean(
        tf.stack(axis=0, values=context_embeddings), 0, keepdims=False
    )

    weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size))
    )
    bias = tf.Variable(
        tf.zeros([vocabulary_size])
    )

    loss = tf.nn.sampled_softmax_loss(
        weights=weights,
        biases=bias,
        inputs=embedding,
        labels=train_labels,
        num_sampled=64,
        num_classes=vocabulary_size
    )
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(cost)

    normalization_embed = word_embed / tf.sqrt(
        tf.reduce_sum(
            tf.square(word_embed), 1, keepdims=True
        )
    )

    validation_embed = tf.nn.embedding_lookup(normalization_embed, validation_data)
    word_similarity = tf.matmul(validation_embed, tf.transpose(normalization_embed))

with tf.Session(graph=tensor_flow_graph) as _session:
    _session.run(tf.global_variables_initializer())

    avg_loss = 0.0

    for step in range(num_steps):
        batch_words, batch_label = cbow.samples(batch_length, skip_window)

        _, loss_value = _session.run(
            [optimizer, loss],
            feed_dict={
                train_dataset: batch_words, train_labels: batch_label
            }
        )
        avg_loss += loss_value

        if step % 2000 == 0:

            if step > 0:
                avg_loss = avg_loss / 2000

            print(f"Avg. loss at step {step} is {np.mean(avg_loss)}")
            avg_loss = 0.0

        if step % 10000 == 0:
            sim = word_similarity.eval()

            for i in range(validation_cnt):
                valid_word = rev_dictionary[validation_words[i]]
                top_k = 10
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                print(f"Nearest {valid_word}: {[rev_dictionary[nearest[k]] for k in range(top_k)]}")

final_embeddings = normalization_embed.eval()
