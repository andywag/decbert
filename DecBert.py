import configparser

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder
import torch.nn as nn
import torch
from typing import Optional, Tuple
import torch.nn.functional as F


class BertDecBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.position_embeddings = nn.Embedding(config.max_position_dimension, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = BertEncoder(config)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None):

        position_ids = torch.arange(self.config.max_position_embeddings).to('cuda:0')
        positions = self.position_embeddings(position_ids)
        embedding_out = positions + input_ids
        layer_out = self.layer_norm(embedding_out)
        drop_out = self.dropout(layer_out)

        encoder_output = self.encoder(drop_out)
        return encoder_output[0]


class BertDecModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.layer = nn.ModuleList([BertDecBlock(config) for _ in range(config.encoder_depth)])
        self.loss = nn.MSELoss()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None
    ) :
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        embeddings = self.word_embeddings(input_ids)
        batch_size = int(embeddings.shape[0])
        sequence_length = int(embeddings.shape[1])
        group_div = int(sequence_length/self.config.encoder_block_size)

        real_embeddings = embeddings.reshape((batch_size*group_div, self.config.encoder_block_size, self.config.hidden_size))
        real_attention_mask = attention_mask.reshape((batch_size*group_div, self.config.encoder_block_size))
        #print("A", real_embeddings.shape, real_attention_mask.shape)

        for x in range(self.config.encoder_depth):
            decode_result = self.layer[x](real_embeddings, attention_mask)
            if x < self.config.encoder_depth - 1:
                old_shape = decode_result.shape
                real_embeddings = decode_result[:, ::2, :].reshape((int(old_shape[0]/2),old_shape[1],old_shape[2]))
                real_attention_mask = real_attention_mask[:, ::2].reshape((int(old_shape[0]/2),old_shape[1]))
            else:
                real_embeddings = decode_result

            #print("D", real_embeddings.shape, real_attention_mask.shape, real_attention_mask)
        real_output = self.mean_pooling(real_embeddings,real_attention_mask)
        norm_output = F.normalize(real_output, 2, 1)

        if target is not None:
            loss = self.loss(norm_output, target)
        else:
            loss = None

        return norm_output, loss
