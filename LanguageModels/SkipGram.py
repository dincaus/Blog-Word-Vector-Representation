import numpy as np
import random
from toolz import frequencies


class SkipGram(object):

    THRESHOLD = 0.00005

    def __init__(self, word_idx, threshold=None):

        self._threshold = threshold or SkipGram.THRESHOLD
        self._word_idx = word_idx

        self.__init()

    def __init(self):
        word_counts = frequencies(self._word_idx)
        total_count = len(word_counts)

        freqs = {
            word: count / total_count
            for word, count in word_counts.items()
        }

        p_drop = {
            word: 1 - np.sqrt(self._threshold / freqs[word])
            for word in word_counts
        }

        self._train_words = list(
            filter(
                lambda word:  p_drop[word] < random.random(),
                self._word_idx
            )
        )

    def _target_set(self, batch, batch_idx, window):
        random_num = np.random.randint(1, window + 1)
        word_start = batch_idx - random_num if (batch_idx - random_num) > 0 else 0
        word_stop = batch_idx + random_num

        window_target = set(
            batch[word_start:batch_idx] + batch[batch_idx + 1:word_stop + 1]
        )

        return list(
            window_target
        )

    def samples(self, batch_length, window):
        batch_cnt = len(self._train_words) // batch_length
        train_words = self._train_words[:batch_cnt * batch_length]

        for word_idx in range(0, len(train_words), batch_length):
            input_words, label_words = [], []
            word_batch = train_words[word_idx:word_idx + batch_length]

            for idx in range(len(word_batch)):
                batch_input = word_batch[idx]
                batch_label = self._target_set(word_batch, idx, window)
                label_words.extend(batch_label)
                input_words.extend([batch_input] * len(batch_label))

                yield input_words, label_words
