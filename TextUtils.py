from toolz import frequencies, itemfilter


class TextProcessing(object):

    @staticmethod
    def replace_punctuation(dataset_text):

        if dataset_text is None:
            raise TypeError("Dataset text has to be provided. None value is not allowed.")

        dataset_text = dataset_text.lower()
        dataset_text = dataset_text.replace(".", "<period>").replace(",", "<comma>").\
            replace('"', "<quotation>").replace(";", "<semicolon>").\
            replace("!", "<excal>").replace("?", "<question>").\
            replace("(", "<paren_left>").replace(")", "<paren_right>").\
            replace("--", "<hyphen>").replace(":", "<colon>")

        return dataset_text

    @staticmethod
    def tokenize(dataset_text):

        if dataset_text is None:
            raise TypeError("Dataset text has to be provided. None value is not allowed.")

        return dataset_text.split()

    @staticmethod
    def remove_low_occurrence_words(tokens, k=5):

        if tokens is None:
            raise TypeError("Tokens has to be provided. None value is not allowed.")

        return list(
            itemfilter(
                lambda item: item[1] >= k,
                frequencies(tokens)
            ).keys()
        )

    @staticmethod
    def create_word_map(tokens):

        freqs = frequencies(tokens)
        sorted_vocabulary = sorted(freqs, key=freqs.get, reverse=True)

        index_to_word_mapping = {
            idx: word
            for idx, word in enumerate(sorted_vocabulary)
        }

        word_to_index_mapping = {
            word: idx
            for idx, word in index_to_word_mapping.items()
        }

        word_idx = list(
            map(
                lambda t: word_to_index_mapping[t],
                tokens
            )
        )

        return word_to_index_mapping, index_to_word_mapping, word_idx
