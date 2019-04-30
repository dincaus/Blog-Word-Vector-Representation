import random
import os
import numpy as np
import tensorflow as tf

from config import logger
from TextUtils import TextProcessing
from LanguageModels.SkipGram import SkipGram
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
skip_gram = SkipGram(word_index)

vocab_size = len(rev_dictionary)

# Model implementation
tensor_flow_graph = tf.Graph()
with tensor_flow_graph.as_default():
    input_placeholder = tf.placeholder(tf.int32, [None], name="input_")
    label_placeholder = tf.placeholder(tf.int32, [None, None], name="label")

    word_embedding = tf.Variable(
        tf.random_uniform((vocab_size, 300), -1, 1)
    )
    embedding = tf.nn.embedding_lookup(word_embedding, input_placeholder)

    weights = tf.Variable(
        tf.truncated_normal(
            (vocab_size, 300), stddev=0.1
        )
    )

    bias = tf.Variable(
        tf.zeros(vocab_size)
    )

    loss = tf.nn.sampled_softmax_loss(
        weights=weights,
        biases=bias,
        labels=label_placeholder,
        inputs=embedding,
        num_sampled=100,
        num_classes=vocab_size
    )

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    validation_count = 20
    validation_dict = 100
    validation_words = np.array(
        random.sample(
            range(validation_dict), validation_count // 2
        )
    )
    validation_words = np.append(
        validation_words,
        random.sample(
            range(1000, 1000 + validation_dict), validation_count // 2
        )
    )
    validation_data = tf.constant(validation_words, dtype=tf.int32)

    normalized_embed = word_embedding / (tf.sqrt(tf.reduce_sum(tf.square(word_embedding), 1, keepdims=True)))
    validation_embed = tf.nn.embedding_lookup(
        normalized_embed, validation_data
    )
    word_similarity = tf.matmul(
        validation_embed, tf.transpose(normalized_embed)
    )


epochs = 2
batch_length = 1000
word_window = 10

with tf.Session(graph=tensor_flow_graph) as _session:
    iteration = 1
    loss = 0
    _session.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = skip_gram.samples(batch_length, word_window)

        for x, y in batches:
            train_loss, _ = _session.run(
                [cost, optimizer],
                feed_dict={
                    input_placeholder: x,
                    label_placeholder: np.array(y)[:, None]
                }
            )

            loss += train_loss

            if iteration % 50 == 0:
                print(f"Epoch {e}/{epochs}, Iteration {iteration} [Avg. Training loss {loss/50}]")
                loss = 0

            if iteration % 1000 == 0:
                similarity_ = word_similarity.eval()
                for i in range(validation_count):
                    validated_words = rev_dictionary[validation_words[i]]
                    top_k = 10
                    nearest = (-similarity_[i, :]).argsort()[1:top_k+1]

                    print(
                        f"Nearest to {validated_words}: {[rev_dictionary[nearest[k]] for k in range(top_k)]}"
                    )

            iteration += 1
