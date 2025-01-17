"""Richer embeddings for nodes"""
import torch
import torch.nn as nn

from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.torch.utils import send_to_device as V

from phrasenode.constants import UNK, EOS, TAGS
from phrasenode.utterance_embedder import AverageUtteranceEmbedder, LSTMUtteranceEmbedder, AttentionUtteranceEmbedder
from phrasenode.vocab import MagnitudeEmbeddings, RandomEmbeddings, read_frequency_vocab


def semantic_attrs(attrs):
    whitelist = ['aria', 'tooltip', 'placeholder', 'label', 'title', 'name']
    attrs = [value for key, value in attrs.items() if any(k in key.lower() for k in whitelist)]
    return ' '.join(attrs)


################################################
# Base Embedder

class AllanBaseEmbedder(nn.Module):

    def __init__(self, dim, utterance_embedder, attribute_embedder, recursive_texts,
                 attr_embed_dim, max_attr_tokens, min_id_freq, min_class_freq, dropout,
                 ablate_text=False, ablate_attrs=False):
        """
        Args:
            dim (int): Target embedding dimension
            utterance_embedder (UtteranceEmbedder)
            attribute_embedder (UtteranceEmbedder)
            recursive_texts (bool): For node text, whether to recursively combine the
                texts of the descendants
            attr_embed_dim (int): Size of each attribute embedding
            max_attr_tokens (int): Limit the number of attribute tokens to embed
            min_id_freq (int): Minimum token frequency of tokens in id vocab
            min_class_freq (int): Minimum token frequency of tokens in class vocab
            dropout (float): Dropout rate
        """
        super(AllanBaseEmbedder, self).__init__()
        self._dim = dim

        # Text embedder
        self._utterance_embedder = utterance_embedder
        self._max_words = utterance_embedder.max_words
        self._recursive_texts = recursive_texts
        self.ablate_text = ablate_text
        self.ablate_attrs = ablate_attrs

        # Attribute embedders
        self._attr_embed_dim = attr_embed_dim

        tags = [UNK, EOS] + TAGS
        self._tag_embedder = TokenEmbedder(RandomEmbeddings(tags, attr_embed_dim))

        ids = read_frequency_vocab('frequent-ids', min_id_freq)
        self._id_embedder = AverageUtteranceEmbedder(TokenEmbedder(RandomEmbeddings(ids, attr_embed_dim)),
                                                     max_attr_tokens, lang="en")
        # self._id_embedder = attr_embedder

        classes = read_frequency_vocab('frequent-classes', min_class_freq)
        self._classes_embedder = AverageUtteranceEmbedder(TokenEmbedder(RandomEmbeddings(classes, attr_embed_dim)),
                                                          max_attr_tokens, lang="en")
        # self._classes_embedder = attr_embedder
        coords_dim = 3

        self._other_embedder = attribute_embedder

        # Combine
        input_dim = (2 * self._utterance_embedder.embed_dim + 3 * attr_embed_dim + coords_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(input_dim, dim)

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
            if not self.ablate_text:
                if self._recursive_texts:
                    text = ' '.join(node.all_texts(max_words=self._max_words))
                else:
                    text = node.text or ''
                texts.append(utterance_embedder.tokenize(text))
            else:
                texts.append([])
        text_embeddings = utterance_embedder(texts)

        # num_nodes x attr_embed_dim
        tags = [node.tag for node in nodes]
        tag_embeddings = self._tag_embedder.embed_tokens(tags)

        # num_nodes x attr_embed_dim
        id_embedder = self._id_embedder
        if not self.ablate_attrs:
            ids = [id_embedder.tokenize(node.id_) for node in nodes]
        else:
            ids = [[] for node in nodes]
        id_embeddings = id_embedder(ids)

        # num_nodes x attr_embed_dim
        classes_embedder = self._classes_embedder
        if not self.ablate_attrs:
            classes = [classes_embedder.tokenize(' '.join(node.classes)) for node in nodes]
        else:
            classes = [[] for node in nodes]
        class_embeddings = self._classes_embedder(classes)

        other_embedder = self._other_embedder
        if not self.ablate_attrs:
            other = [other_embedder.tokenize(semantic_attrs(node.attributes)) for node in nodes]
        else:
            other = [[] for node in nodes]
        other_embeddings = self._other_embedder(other)

        # num_nodes x 3
        coords = V(torch.tensor([[node.x_ratio, node.y_ratio, float(node.visible)] for node in nodes], dtype=torch.float32))

        # num_nodes x dom_embed_dim
        dom_embeddings = torch.cat((text_embeddings, tag_embeddings, id_embeddings, class_embeddings, other_embeddings, coords), dim=1)
        # dom_embeddings = text_embeddings
        return self.fc(dom_embeddings)
        # return F.relu(self.fc(self.dropout(dom_embeddings)))
        # return F.sigmoid(self.fc(dom_embeddings))


# Node properties:
# 'add_child', 'all_texts', 'ancestor_path', 'attributes', 'bottom',
# 'children', 'classes', 'depth', 'height', 'hidden', 'id_', 'is_leaf', 'left',
# 'left_offset', 'neighbors', 'next_sibling', 'old_ref', 'parent',
# 'prev_sibling', 'raw_info', 'ref', 'right', 'style', 'style_overrides', 'tag',
# 'text', 'top', 'top_level', 'top_offset', 'value', 'visible', 'visualize',
# 'web_page', 'width', 'x_ratio', 'xid', 'y_ratio'


################################################
# Final model

def make_embedder(token_embedder, config, lang):
    # Attribute embedder
    if config.type == 'average':
        return AverageUtteranceEmbedder(token_embedder, config.max_words, lang)
    elif config.type == 'lstm':
        return LSTMUtteranceEmbedder(token_embedder, config.lstm_dim, config.max_words, lang)
    elif config.type == 'attention_lstm':
        return AttentionUtteranceEmbedder(token_embedder, config.lstm_dim, config.max_words, lang)
    else:
        raise ValueError('Unknown AttributeEmbedder type {}'.format(config.type))


def get_allan_embedder(config):
    """Create a new AllanEmbedder based on the config

    Args:
        config (Config): the root config
    Returns:
        AllanEmbedder
    """
    cm = config.model
    cmu = cm.utterance_embedder
    cmt = cm.node_embedder.token_embedder
    # cma = cm.node_embedder.attr_embedder
    cmb = cm.node_embedder.base_embedder

    # Token embedder
    magnitude_filename = cmt.magnitude_filename
    vocab_filename = cmt.vocab_filename
    word_embeddings = MagnitudeEmbeddings(magnitude_filename, vocab_filename, cmt.vocab_size, cmt.word_embed_dim)
    token_embedder = TokenEmbedder(word_embeddings, trainable=cmt.trainable)

    lang = cmt.lang
    # Utterance embedder
    utterance_embedder = make_embedder(token_embedder, cmu, lang)

    # Attribute embedder
    # attr_embedder = make_embedder(attr_token_embedder, cma)
    # AverageUtteranceEmbedder(TokenEmbedder(RandomEmbeddings(ids, attr_embed_dim)), max_attr_tokens)
    attr_embedder = utterance_embedder

    # Base node embedder
    base_embedder = AllanBaseEmbedder(cm.dim,
                                      utterance_embedder, attr_embedder, cmb.recursive_texts,
                                      cmt.word_embed_dim, cmb.max_attr_tokens,
                                      cmb.min_id_freq, cmb.min_class_freq, cm.dropout,
                                      ablate_text=cm.ablate_text, ablate_attrs=cm.ablate_attrs)
    return base_embedder
