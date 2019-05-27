import numpy as np
from abc import ABC, abstractmethod
from typing import List


class TextParser(ABC):
    def __init__(self, type_func):
        self.type_func = type_func

    @abstractmethod
    def __call__(self, x: List[List[str]]):
        pass

    def convert_type(self, x):
        return self.type_func(x)


class BasicParser(TextParser):
    def __call__(self, x):
        return self.convert_type(x)


class BagOfWords(TextParser):
    def __init__(self, vocabulary, type_func):
        """

        :param vocabulary: List of strings representing the vocabulary.
        :param type_func: Function which converts the output to the desired type, e.g. np.array.
        """
        self.vocab = vocabulary
        self.vocab_size = len(self.vocab)
        TextParser.__init__(self, type_func)

    def __call__(self, x):
        bags_of_words = np.zeros((len(x), self.vocab_size + 1))  # +1 for out of vocabulary tokens.
        for idx, token_list in enumerate(x):
            for token in token_list:
                try:
                    token_idx = self.vocab.index(token)
                except:
                    token_idx = -1
                bags_of_words[idx, token_idx] += 1

        return self.convert_type(bags_of_words)


class Word2Vec(TextParser):
    def __init__(self, type_func, word2vec_model, return_func):
        """

        :param type_func: Function which converts the output to the desired type, e.g. np.array.
        :param word2vec_model: Gensim model for word embeddings.
        :param return_func: Operation to perform on the list of embeddings when a list is not the required output. For
                            instance, a desired output may be the sum of embeddings.
        """
        self.word2vec_model = word2vec_model
        self.return_func = return_func
        self.vector_size = self.word2vec_model.vector_size
        self.zero_vec = np.zeros(self.vector_size)
        TextParser.__init__(self, type_func)

    def __call__(self, x):
        embeddings_list = [[] for _ in range(len(x))]
        for idx, token_list in enumerate(x):
            for token in token_list:
                if token not in self.word2vec_model:
                    embeddings_list[idx].append(self.zero_vec)
                else:
                    embeddings_list[idx].append(self.word2vec_model[token])

        ret_list = []
        for sub_list in embeddings_list:
            ret_list.append(self.return_func(sub_list))

        return self.convert_type(np.array(ret_list))


def basic_func(x: List[float]) -> np.ndarray:
    return np.array(x)


def sum_func(x: List[float]) -> float:
    return sum(x)


def tokenizer(x: str) -> List[str]:
    """
    Parses a string into a list of lower-case tokens, removing punctuation and stopwords.

    :param x: String
    :return: List of strings.
    """
    from nltk.tokenize import word_tokenize
    import re
    tokens = word_tokenize(x)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    import string
    # table = str.maketrans('', '', string.punctuation)
    # stripped = [w.translate(table) for w in tokens]
    stripped = []
    for token in tokens:
        for sub_token in re.findall(r"\w+|[^\w\s]", token, re.UNICODE):
            stripped.append(sub_token)
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    # from nltk.corpus import stopwords
    # stop_words = set(stopwords.words('english'))
    # words = [w for w in words if not w in stop_words]

    return words


def _test_word2vec():
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    parser = Word2Vec(lambda x: x, model, lambda x: x)
    inp = [tokenizer('Clearing You are in a clearing, with a forest surrounding you on all sides.'
                     'A path leads south. On the ground is a pile of leaves.'),
           tokenizer('You have no items on you.')]
    output = parser(inp)
    print(output)


if __name__ == '__main__':
    _test_word2vec()
