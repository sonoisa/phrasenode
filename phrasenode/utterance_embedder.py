"""Utterance embedder"""
import re
import torch
import torch.nn as nn

from gtd.ml.torch.attention import Attention
from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.source_encoder import BidirectionalSourceEncoder

from phrasenode.constants import EOS


################################################
# Tokenization

TOKENIZER = re.compile(r'[^\W_]+|[^\w\s-]', re.UNICODE | re.MULTILINE | re.DOTALL)


def word_tokenize(text):
    """Tokenize without keeping the mapping to the original string.

    Args:
        text (str or unicode)
    Return:
        list[unicode]
    """
    return TOKENIZER.findall(text)


TOKENIZER2 = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w]+", re.UNICODE | re.MULTILINE | re.DOTALL)


# courtesy https://stackoverflow.com/questions/6202549/word-tokenization-using-python-regular-expressions
def word_tokenize2(text):
    """Tokenize without keeping the mapping to the original string.
    Removes punctuation, keeps dashes, and splits on capital letters correctly.
    Returns tokenized words in lower case.
    E.g.
    Jeff's dog is un-American SomeTimes! BUT NOTAlways
    ['jeff's', 'dog', 'is', 'un', 'american', 'some', 'times', 'but', 'not', 'always']


    Args:
        text (str or unicode)
    Return:
        list[unicode]
    """
    return [s.lower() for s in TOKENIZER2.findall(text)]


################################################
# Utterance Embedder

class AverageUtteranceEmbedder(nn.Module):
    """Takes a string, embeds the tokens using the token_embedder,
    and return the average of the results.
    """

    def __init__(self, token_embedder, max_words):
        """Initialize

        Args:
            token_embedder (TokenEmbedder): used to embed each token
            max_words (int): maximum number of words to embed
        """
        super(AverageUtteranceEmbedder, self).__init__()
        self._token_embedder = token_embedder
        self._embed_dim = token_embedder.embed_dim
        self._max_words = max_words

    def forward(self, utterances):
        """Embeds an utterances.

        Args:
            utterances (list[list[str]]): list[str] is a list of tokens
            forming a sentence. list[list[str]] is batch of sentences.

        Returns:
            Tensor: batch x word_embed_dim (average of word vectors)
        """
        # Cut to max_words + look up indices
        utterances = [utterance[:self._max_words] + [EOS] for utterance in utterances]
        token_indices = SequenceBatch.from_sequences(
                utterances, self._token_embedder.vocab)
        # batch x seq_len x token_embed_dim
        token_embeds = self._token_embedder.embed_seq_batch(token_indices)
        # batch x token_embed_dim
        averaged = SequenceBatch.reduce_mean(token_embeds)
        return averaged

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def max_words(self):
        return self._max_words

    @property
    def token_embedder(self):
        return self._token_embedder

    def tokenize(self, text):
        return word_tokenize2(text)


class LSTMUtteranceEmbedder(nn.Module):
    """Takes a string, embeds the tokens using the token_embedder, and passes
    the embeddings through a biLSTM padded / masked up to sequence_length.
    Returns the concatenation of the two front and end hidden states.
    """

    def __init__(self, token_embedder, lstm_dim, max_words):
        """Initialize

        Args:
            token_embedder (TokenEmbedder): used to embed each token
            lstm_dim (int): output dim of the lstm
            max_words (int): maximum number of words to embed
        """
        super(LSTMUtteranceEmbedder, self).__init__()
        self._token_embedder = token_embedder
        self._bilstm = BidirectionalSourceEncoder(
               token_embedder.embed_dim, lstm_dim, nn.LSTMCell)
        self._embed_dim = lstm_dim
        self._max_words = max_words

    def forward(self, utterances):
        """Embeds a batch of utterances.

        Args:
            utterances (list[list[unicode]]): list[unicode] is a list of tokens
            forming a sentence. list[list[unicode]] is batch of sentences.

        Returns:
            Variable[FloatTensor]: batch x lstm_dim
                (concatenated first and last hidden states)
        """
        # Cut to max_words + look up indices
        utterances = [utterance[:self._max_words] + [EOS] for utterance in utterances]
        token_indices = SequenceBatch.from_sequences(
                utterances, self._token_embedder.vocab)
        # batch x seq_len x token_embed_dim
        token_embeds = self._token_embedder.embed_seq_batch(token_indices)
        bi_hidden_states = self._bilstm(token_embeds.split())
        final_states = torch.cat(bi_hidden_states.final_states, 1)
        return torch.stack(final_states, 0)

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def max_words(self):
        return self._max_words

    @property
    def token_embedder(self):
        return self._token_embedder

    def tokenize(self, text):
        return word_tokenize2(text)


class AttentionUtteranceEmbedder(nn.Module):
    """Takes a string, embeds the tokens using the token_embedder, and passes
    the embeddings through a biLSTM padded / masked up to sequence_length.
    Returns the concatenation of the two front and end hidden states.
    """

    def __init__(self, token_embedder, lstm_dim, max_words):
        """Initialize

        Args:
            token_embedder (TokenEmbedder): used to embed each token
            lstm_dim (int): output dim of the lstm
            max_words (int): maximum number of words to embed
        """
        super(AttentionUtteranceEmbedder, self).__init__()
        self._token_embedder = token_embedder
        self._bilstm = BidirectionalSourceEncoder(
               token_embedder.embed_dim, lstm_dim, nn.LSTMCell)
        self._embed_dim = lstm_dim
        self._max_words = max_words

        self._attention = Attention(token_embedder.embed_dim, lstm_dim, lstm_dim)

    def forward(self, utterances):
        """Embeds a batch of utterances.

        Args:
            utterances (list[list[unicode]]): list[unicode] is a list of tokens
            forming a sentence. list[list[unicode]] is batch of sentences.

        Returns:
            Variable[FloatTensor]: batch x lstm_dim
                (concatenated first and last hidden states)
        """
        # Cut to max_words + look up indices
        utterances = [utterance[:self._max_words] + [EOS] for utterance in utterances]
        token_indices = SequenceBatch.from_sequences(
                utterances, self._token_embedder.vocab)
        # batch x seq_len x token_embed_dim
        token_embeds = self._token_embedder.embed_seq_batch(token_indices)
        # print('token_embeds', token_embeds)
        bi_hidden_states = self._bilstm(token_embeds.split())
        final_states = torch.cat(bi_hidden_states.final_states, 1)

        hidden_states = SequenceBatch.cat(bi_hidden_states.combined_states)
        return self._attention(hidden_states, final_states).context

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def max_words(self):
        return self._max_words

    @property
    def token_embedder(self):
        return self._token_embedder

    def tokenize(self, text):
        return word_tokenize2(text)
