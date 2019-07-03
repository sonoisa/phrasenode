"""Stupid embedder. Only sum the texts."""
import torch.nn as nn

from gtd.ml.torch.token_embedder import TokenEmbedder

from phrasenode.utterance_embedder import AverageUtteranceEmbedder, LSTMUtteranceEmbedder
from phrasenode.vocab import MagnitudeEmbeddings


class StupidEmbedder(nn.Module):

    def __init__(self, dim, utterance_embedder, dropout):
        """
        Args:
            dim (int): Target embedding dimension
            utterance_embedder (UtteranceEmbedder)
            dropout (float): Dropout rate
        """
        super(StupidEmbedder, self).__init__()
        self._dim = dim
        # Text embedder
        self._utterance_embedder = utterance_embedder
        self._max_words = utterance_embedder.max_words
        # Combine
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(utterance_embedder.embed_dim, dim)

    @property
    def embed_dim(self):
        return self._dim

    @property
    def token_embedder(self):
        return self._utterance_embedder.token_embedder

    @property
    def utterance_embedder(self):
        return self._utterance_embedder

    def forward(self, nodes):
        """Embeds a batch of Nodes.

        Args:
            nodes (list[Node])
        Returns:
            embeddings (Tensor): num_nodes x embed_dim
        """
        texts = []
        utterance_embedder = self._utterance_embedder
        for node in nodes:
            text = ' '.join(node.all_texts(max_words=self._max_words))
            # texts.append([x.lower() for x in word_tokenize(text)])
            texts.append(utterance_embedder.tokenize(text))
        text_embeddings = self._utterance_embedder(texts)
        return text_embeddings


def get_stupid_embedder(config):
    """Create a new StupidEmbedder based on the config

    Args:
        config (Config): the root config
    Returns:
        StupidEmbedder
    """
    cm = config.model
    cmu = cm.utterance_embedder
    cmt = cm.node_embedder.token_embedder

    # Token embedder
    magnitude_filename = cmt.magnitude_filename
    vocab_filename = cmt.vocab_filename
    word_embeddings = MagnitudeEmbeddings(magnitude_filename, vocab_filename, cmt.vocab_size, cmt.word_embed_dim)
    token_embedder = TokenEmbedder(word_embeddings, trainable=cmu.trainable)

    # Utterance embedder
    if cmu.type == 'average':
        utterance_embedder = AverageUtteranceEmbedder(token_embedder, cmu.max_words)
    elif cmu.type == 'lstm':
        utterance_embedder = LSTMUtteranceEmbedder(token_embedder, cmu.lstm_dim, cmu.max_words)
    else:
        raise ValueError('Unknown UtteranceEmbedder type {}'.format(cmu.type))
    # Embedder
    embedder = StupidEmbedder(cm.dim, utterance_embedder, cm.dropout)
    return embedder
