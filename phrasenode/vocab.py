"""Vocab and lookup-based embeddings.

gtd naming scheme:
- Vocab: list of words, with word2index(w) and index2word(i) methods
- Embeddings: Vocab + matrix of the same length
- TokenEmbedder: a torch Module that takes an Embeddings
"""
import logging
import os

import numpy as np

from codecs import open

from gtd.ml.vocab import SimpleVocab, SimpleEmbeddings

from phrasenode.constants import UNK, EOS
from phrasenode import data

from hashlib import md5
from pymagnitude import Magnitude


class VocabWithUnk(SimpleVocab):
    """Vocab where unknown words are mapped to UNK

    IMPORTANT NOTE: VocabWithUnk is blind to casing! All words are converted to lower-case.
    """

    def __init__(self, tokens):
        """
        Args:
            tokens (list[basestring]): Must begin with UNK and EOS
        """
        tokens = [t.lower() for t in tokens]
        if tokens[0] != UNK:
            raise ValueError('UNK must be the first element of the tokens list')
        if tokens[1] != EOS:
            raise ValueError('EOS must be the second element of the tokens list')
        super(VocabWithUnk, self).__init__(tokens)

    def word2index(self, w):
        """Map a word to an integer.

        If the word is not known to the vocab, return the index for UNK.
        """
        sup = super(VocabWithUnk, self)
        try:
            return sup.word2index(w.lower())
        except KeyError:
            return sup.word2index(UNK)


##############################################################################
# Hashing Trick: courtesy of Keras
# https://github.com/keras-team/keras/blob/master/keras/preprocessing/text.py
##############################################################################

def hashing_trick(word, n,
                  hash_function=None):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        word: Input word (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    return hash_function(word) % (n - 1) + 1


class VocabWithHashTrick(SimpleVocab):
    """Vocab where unknown words are mapped via the hashing trick

    IMPORTANT NOTE: VocabWithHashTrick is blind to casing! All words are converted to lower-case.
    """

    def __init__(self, tokens, output_dim):
        """
        Args:
            tokens (list[basestring]): Must begin with UNK and EOS
        """
        self.output_dim = output_dim
        tokens = [t.lower() for t in tokens]
        if tokens[0] != UNK:
            raise ValueError('UNK must be the first element of the tokens list')
        if tokens[1] != EOS:
            raise ValueError('EOS must be the second element of the tokens list')
        super(VocabWithHashTrick, self).__init__(tokens)

    def word2index(self, w):
        """Map a word to an integer via hashing"""
        return hashing_trick(w.lower(), self.output_dim)


################################################
# Load pre-trained word embedding from a file.

def read_magnitude_vectors(magnitude_filepath, vocab_filepath, vocab_size, dim, special_tokens=[UNK]):
    """Read word vectors from *.magnitude

    Args:
        magnitude_filepath (str): magnituide file path
        vocab_filepath (str): vocabulary file path
        vocab_size (int): Maximum vocab size (including special tokens)
        dim (int): Dimension of the word vectors to load
        special_tokens (list[str])

    Return:
        words (list[str]): list of length vocab_size
        embeddings (np.array): (vocab_size, dim)
    """
    logging.info('Loading word vectors from %s', magnitude_filepath)
    words = [x for x in special_tokens]
    word_set = set()
    with open(vocab_filepath, 'r', 'utf8') as fin:
        for line in fin:
            word = line.strip().split("\t")[0]
            if word in word_set:
                logging.warning("token must be unique. non-unique token='{}'".format(word))
            elif len(word) > 0:
                word_set.add(word)
                words.append(word)
                if len(words) == vocab_size:
                    break
    magnitude = Magnitude(magnitude_filepath, case_insensitive=True, normalized=True)
    vectors = magnitude.query(words[len(special_tokens):])
    # special vectors for UNK
    special_vectors = np.random.normal(size=(len(special_tokens), dim))
    special_vectors /= np.linalg.norm(special_vectors, ord=2, axis=1, keepdims=True)
    # Concatenate
    vectors = np.vstack([special_vectors, vectors]).astype('float32')
    assert vectors.shape[0] == len(words)
    assert vectors.shape[1] == dim
    logging.info('Loaded %d word vectors; shape = %s', len(words), str(vectors.shape))
    return words, vectors


class MagnitudeEmbeddings(SimpleEmbeddings):

    def __init__(self, magnitude_filename, vocab_filename, vocab_size, dim):
        """Read word vectors from [data.workspace.word_embeddings]/*.magnitude

        Args:
            magnitude_filename (str): magnituide file name
            vocab_filename (str): vocabulary file name
            vocab_size (int)
            dim (int)
        """
        dirname = data.workspace.word_embeddings
        magnitude_filepath = os.path.join(dirname, magnitude_filename.format(dim))
        vocab_filepath = os.path.join(dirname, vocab_filename.format(dim))
        words, vectors = read_magnitude_vectors(magnitude_filepath, vocab_filepath,
                                                vocab_size, dim, special_tokens=[UNK, EOS])
        vocab = VocabWithHashTrick(words, vocab_size)
        # vocab = VocabWithUnk(words)
        super(MagnitudeEmbeddings, self).__init__(vectors, vocab)


################################################
# Random: Read vocab from file but randomly initialize embeddings.

class RandomEmbeddings(SimpleEmbeddings):

    def __init__(self, words, embed_dim):
        """
        Args:
            words (list[unicode]): List of tokens
            embed_dim (int): Dimension of the embedded vectors
        """
        vocab = VocabWithUnk(words)
        embed_matrix = np.random.uniform(
                -np.sqrt(3. / embed_dim), np.sqrt(3. / embed_dim),
                size=(len(vocab), embed_dim)).astype(np.float32)
        super(RandomEmbeddings, self).__init__(embed_matrix, vocab)


def read_frequency_vocab(filename, min_freq):
    """Read a list of words from the frequency file.

    Each line in the file must be <frequency> <TAB> <word>

    Args:
        filename: filename inside the vocab directory
        min_freq: minimum frequency of the words to include
    Returns:
        list[str], including UNK and EOS at the beginning
    """
    filename = os.path.join(data.workspace.vocab, filename)
    words = [UNK, EOS]
    with open(filename, 'r', 'utf8') as fin:
        for line in fin:
            freq, word = line.rstrip('\n').split('\t')
            if word.strip() and int(freq) >= min_freq:
                words.append(word.strip())
    logging.info('Read %d words from %s', len(words), filename)
    return words
