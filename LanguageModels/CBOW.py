import numpy as np

from toolz import frequencies
from collections import deque


class CBOW(object):

    def __init__(self, word_idx):
        self._word_idx = word_idx

    def samples(self, batch_length, window):
        relevant_words = 2 * window + 1

        batch = np.ndarray(shape=(batch_length, relevant_words - 1), dtype=np.int32)
        label = np.ndarray(shape=(batch_length, 1), dtype=np.int32)

        buffer = deque(maxlen=relevant_words)
        data_index = 0

        for _ in range(relevant_words):
            buffer.append(self._word_idx[data_index])
            data_index = (data_index + 1) % len(self._word_idx)

        for i in range(batch_length):
            target = window

            col_idx = 0
            for j in range(relevant_words):

                if j == relevant_words // 2:
                    continue

                batch[i, col_idx] = buffer[j]
                col_idx += 1

            label[i, 0] = buffer[target]

            buffer.append(self._word_idx[data_index])
            data_index = (data_index + 1) % len(self._word_idx)

        assert batch.shape[0] == batch_length and batch.shape[1] == relevant_words - 1

        return batch, label
