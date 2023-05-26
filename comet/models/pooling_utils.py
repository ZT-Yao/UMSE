# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch


def average_pooling(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """Average pooling function.
    :param tokens: Word ids [batch_size x seq_length]
    :param embeddings: Word embeddings [batch_size x seq_length x hidden_size]
    :param mask: Padding mask [batch_size x seq_length]
    :param padding_index: Padding value.
    """
    wordemb = mask_fill(0.0, tokens, embeddings, padding_index)
    sentemb = torch.sum(wordemb, 1)
    sum_mask = mask.unsqueeze(-1).expand(embeddings.size()).float().sum(1)
    return sentemb / sum_mask


def average_each_pooling(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    padding_index: int,
    token_type: torch.Tensor,
    num_of_inputs: int
) -> torch.Tensor:
    """Average pooling function.
    :param tokens: Word ids [batch_size x seq_length]
    :param embeddings: Word embeddings [batch_size x seq_length x hidden_size]
    :param mask: Padding mask [batch_size x seq_length]
    :param padding_index: Padding value.
    :param token_type: Token types for multiple input segments.
    :param num_of_inputs: Number of input segments.
    """
    all_sent_embs = list()
    for i in range(0, num_of_inputs):
        sub_mask = token_type.eq(i)
        # print('sub_mask for each step:', sub_mask.long())
        wordemb = embeddings.masked_fill(mask=sub_mask.bitwise_not().unsqueeze(dim=-1), value=0.0)
        sentlen = sub_mask.unsqueeze(dim=-1).float().sum(dim=1)
        # print('embedding:', wordemb, wordemb.size())
        # print('sentlen:', sentlen, sentlen.size())
        # print('sentlen for each step:', sentlen)
        sentemb = wordemb.sum(dim=1) / sentlen
        all_sent_embs.append(sentemb)
        # input()

    # print('all_sent_embs:', list(x.size() for x in all_sent_embs))
    return torch.cat(all_sent_embs, dim=1)


def class_each_pooling(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    padding_index: int,
    cls_token_ids: torch.Tensor,
    num_of_inputs: int,
    buffered_position_ids: torch.Tensor
) -> torch.Tensor:
    """Class pooling function.
    :param tokens: Word ids [batch_size x seq_length]
    :param embeddings: Word embeddings [batch_size x seq_length x hidden_size]
    :param mask: Padding mask [batch_size x seq_length]
    :param padding_index: Padding value.
    :param cls_token_ids: Indexes for cls tokens for each input segment, [batch_size x num_of_inputs]
    :param num_of_inputs: Number of input segments.
    :param buffered_position_ids: Buffered position indexes from pretrained language model.
    """
    # print('function class_each_pooling')
    # print('tokens:', tokens)
    # print('embeddings:', embeddings, embeddings.size())
    # print('cls_token_ids:', cls_token_ids, cls_token_ids.size())
    # print('num_of_inputs:', num_of_inputs)
    # input()
    all_sent_embs = [embeddings[:, 0, :]]
    batch_size, max_seq_len = tokens.size()
    embedding_size = embeddings.size(-1)
    # print('batch_size:', batch_size)
    # print('max_seq_len:', max_seq_len)
    # print('embedding_size:', embedding_size)

    # print('extra shift for cls_ids:', buffered_position_ids[:, :batch_size].view(-1, 1) * max_seq_len)    
    true_cls_ids = (cls_token_ids + buffered_position_ids[:, :batch_size].view(-1, 1) * max_seq_len).view(-1)
    # print('true_cls_ids:', true_cls_ids, true_cls_ids.size())
    cls_rep = embeddings.view(batch_size * max_seq_len, embedding_size)[true_cls_ids].view(batch_size, num_of_inputs * embedding_size)
    # print('cls_rep:', cls_rep, cls_rep.size())
    # input()
    return cls_rep.contiguous()


def max_pooling(
    tokens: torch.Tensor, embeddings: torch.Tensor, padding_index: int
) -> torch.Tensor:
    """Max pooling function.
    :param tokens: Word ids [batch_size x seq_length]
    :param embeddings: Word embeddings [batch_size x seq_length x hidden_size]
    :param padding_index: Padding value.
    """
    return mask_fill(float("-inf"), tokens, embeddings, padding_index).max(dim=1)[0]


def mask_fill(
    fill_value: float,
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)
