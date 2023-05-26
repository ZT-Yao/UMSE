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

r"""
RegressionMetric
========================
    Regression Metric that learns to predict a quality assessment by looking
    at source, translation and reference.
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from comet.models.base import CometModel
from comet.modules import FeedForward
from torchmetrics import MetricCollection, PearsonCorrcoef, SpearmanCorrcoef
from transformers import AdamW

from torch.nn.utils.rnn import pad_sequence
from torch.nn import Parameter
from math import ceil

import re
import jsonlines
from string import punctuation
from nltk.corpus import stopwords

import json
import random

torch.use_deterministic_algorithms(False)
class RegressionMetric(CometModel):
    """RegressionMetric:

    :param nr_frozen_epochs: Number of epochs (% of epoch) that the encoder is frozen.
    :param keep_embeddings_frozen: Keeps the encoder frozen during training.
    :param optimizer: Optimizer used during training.
    :param encoder_learning_rate: Learning rate used to fine-tune the encoder model.
    :param learning_rate: Learning rate used to fine-tune the top layers.
    :param layerwise_decay: Learning rate % decay from top-to-bottom encoder layers.
    :param encoder_model: Encoder model to be used.
    :param pretrained_model: Pretrained model from Hugging Face.
    :param pool: Pooling strategy to derive a sentence embedding ['cls', 'max', 'avg', 'avg_each', 'cls_each'].
    :param layer: Encoder layer to be used ('mix' for pooling info from all layers.)
    :param dropout: Dropout used in the top-layers.
    :param batch_size: Batch size used during training.
    :param train_data: Path to a csv file containing the training data.
    :param validation_data: Path to a csv file containing the validation data.
    :param hidden_sizes: Hidden sizes for the Feed Forward regression.
    :param activations: Feed Forward activation function.
    :param load_weights_from_checkpoint: Path to a checkpoint file.
    :param input_segments: Which segments for input, any combination among 'src', 'hyp' and 'ref'.
    :param pooling_rep: Which representations for regression, any combination among 'hyp', 'ref', 'src_hyp_prod', 'src_hyp_l1', 'ref_hyp_prod' and 'ref_hyp_l1'.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = False,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 1e-05,
        learning_rate: float = 3e-05,
        embedding_learning_rate: Union[None, float] = None,
        layerwise_decay: float = 0.95,
        encoder_model: str = "RoBERTa",
        pretrained_model: str = "roberta-large",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        dropout: float = 0.1,
        batch_size: int = 4,
        training_data_path: Dict[str, str] = {},
        validation_data_path: Optional[str] = None, 
        hidden_sizes: List[int] = [2304, 768],
        activations: str = "Tanh",
        final_activation:  Optional[str] = None,
        load_weights_from_checkpoint: Optional[str] = None,
        input_segments: List[str] = ['hyp', 'src', 'ref'],
        pooling_rep: List[str] = ['hyp', 'ref', 'src_hyp_prod', 'src_hyp_l1', 'ref_hyp_prod', 'ref_hyp_l1'],
        combine_inputs: bool = False,
        multiple_segment_embedding: bool = False,
        attention_excluded_regions: List[str] = [],
        attention_excluded_regions_dict: Dict[str, str] = {},
        attention_excluded_regions_sampling: Dict[str, float] = {},
        cls_from_all_to_cls: bool = False,
        cls_from_cls_to_all: bool = False,
        reset_position_for_each_segment: bool = False,
        bos_for_segments: List[str] = ['<s>', '</s>', '</s>'],
        eos_for_segments: List[str] = ['</s>', '</s>', '</s>'],
  
    ) -> None:
        super().__init__(
            nr_frozen_epochs,
            keep_embeddings_frozen,
            optimizer,
            encoder_learning_rate,
            learning_rate,
            layerwise_decay,
            encoder_model,
            pretrained_model,
            pool,
            layer,
            dropout,
            batch_size,
            training_data_path,
            validation_data_path,
            load_weights_from_checkpoint,
            "regression_metric",
        )
        self.save_hyperparameters()
        if any(x not in ['hyp', 'src', 'ref'] for x in input_segments):
            raise ValueError('Invalid setting for input segments excluding \'hyp\', \'src\' and \'ref\'.')

        self.input_segments = list()
        for x in ['hyp', 'src', 'ref']:
            if x in input_segments:
                self.input_segments.append(x)
        self.input_segments_dict = dict((v, k) for k, v in enumerate(self.input_segments))

        if 'hyp' in self.input_segments and len(self.input_segments) == 1:
            raise ValueError('Invalid setting for metric because only input is candidate.')
        self.pooling_rep = pooling_rep
        self.combine_inputs = combine_inputs
        self.multiple_segment_embedding = multiple_segment_embedding

        special_tokens = self.encoder.prepare_sample(['<pad>'])['input_ids'].view(-1).cpu().tolist()
        self.pad_idx = special_tokens[1]
        self.bos_idx = special_tokens[0]
        self.eos_idx = special_tokens[2]

        self.special_tokens = {'<s>': self.bos_idx, '<pad>': self.pad_idx, '</s>': self.eos_idx}
        print('Indexes for special tokens:')
        print('PAD (<pad>):', self.pad_idx)
        print('BOS (<s>):', self.bos_idx)
        print('EOS (</s>):', self.eos_idx)

        if self.combine_inputs:
            print('The input will be formatted as the combination of', self.input_segments, '.')
        else:
            print('The input will be seperated as', self.input_segments, '.')
            print('Features for regression: %s' % ' '.join(self.pooling_rep))
        
        if self.multiple_segment_embedding:
            if self.encoder.model.embeddings.token_type_embeddings.weight.size(0) == 1:
                print('Segment embeddings will be repeated for each input among \'[%s]\' for initialization.' % ', '.join(self.input_segments))
                self.encoder.model.embeddings.token_type_embeddings.weight = Parameter(self.encoder.model.embeddings.token_type_embeddings.weight.repeat(len(self.input_segments), 1).contiguous())
            elif self.encoder.model.embeddings.token_type_embeddings.weight.size(0) == len(self.input_segments):
                print('Segment embeddings loaded from checkpoint has the same size as that of training model.')
            else:
                raise ValueError('Segment embeddings loaded from checkpoint can\'t convert to acquired embeddings.')
            print('Segment embeddings size:', self.encoder.model.embeddings.token_type_embeddings.weight.size())


        if self.hparams.pool in ['avg_each', 'cls_each'] and not self.combine_inputs:
            raise ValueError('%s pooling only works for setting combine_inputs to True. Please use %s instead.' % (self.hparams.pool, self.hparams.pool[:3]))
            
        self.cls_from_all_to_cls = cls_from_all_to_cls
        self.cls_from_cls_to_all = cls_from_cls_to_all
        self.reset_position_for_each_segment = reset_position_for_each_segment
        if self.reset_position_for_each_segment:
            print('For each segment, position embedding will be restarted from 0.')

        if len(bos_for_segments) < len(self.input_segments):
            raise ValueError('Number of bos for segments doesn\'t match the numebr of input segments.')
        
        if len(bos_for_segments) > len(self.input_segments):
            print('Number of bos for segments doesn\'t match the number of input segments. Cutting the former to meet the consistency.')
            bos_for_segments = bos_for_segments[:len(self.input_segments)]
        
        if any(x not in self.special_tokens.keys() for x in bos_for_segments):
            raise ValueError('BOS tokens are advised to be chosen from [%s].' % (', '.join(self.special_tokens.keys())))
        self.bos_for_segments = {k: self.special_tokens[v] for k, v in zip(self.input_segments, bos_for_segments)}
        print('BOS symbols are {%s} for all segments.' % (', '.join(str(x1) + ': ' + str(x2) for (x1, x2) in self.bos_for_segments.items())))

        if len(eos_for_segments) < len(self.input_segments):
            raise ValueError('Number of eos for segments doesn\'t match the numebr of input segments.')
        
        if len(eos_for_segments) > len(self.input_segments):
            print('Number of bos for segments doesn\'t match the number of input segments. Cutting the former to meet the consistency.')
            eos_for_segments = eos_for_segments[:len(self.input_segments)]
        
        if any(x not in self.special_tokens.keys() for x in eos_for_segments):
            raise ValueError('EOS tokens are advised to be chosen from [%s].' % (', '.join(self.special_tokens.keys())))
        self.eos_for_segments = {k: self.special_tokens[v] for k, v in zip(self.input_segments, eos_for_segments)}
        print('EOS symbols are {%s} for all segments.' % (', '.join(str(x1) + ': ' + str(x2) for (x1, x2) in self.eos_for_segments.items())))

        in_dim_scale = 1
        if self.combine_inputs and self.hparams.pool in ['avg_each', 'cls_each']:
            in_dim_scale = len(self.input_segments)
        elif not self.combine_inputs:
            in_dim_scale = len(self.pooling_rep)
            
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units * in_dim_scale,
            out_dim=2,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
        )
        

    def init_metrics(self):
        metrics = MetricCollection(
            {"spearman": SpearmanCorrcoef(), "pearson": PearsonCorrcoef()}
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Sets the optimizers to be used during training."""
        prefix_encoder_parameters = [
            {"params": self.encoder.prefix_encoder.parameters(), "lr": self.hparams.learning_rate}
        ]

        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        if self.hparams.embedding_learning_rate is not None:
            layer_parameters[0]['lr'] = self.hparams.embedding_learning_rate

        top_layers_parameters = [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate}
        ]
        if self.layerwise_attention:
            layerwise_attn_params = [
                {
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]
            params = layer_parameters + top_layers_parameters + layerwise_attn_params + prefix_encoder_parameters
        else:
            params = layer_parameters + top_layers_parameters + prefix_encoder_parameters

        optimizer = AdamW(
            params,
            lr=self.hparams.learning_rate,
            correct_bias=True,
        )
        return [optimizer], []

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = {k: [dic[k] for dic in sample] for k in sample[0]}
        
        
        src_inputs = self.encoder.prepare_sample(sample["document"])
        mt_inputs = self.encoder.prepare_sample(sample["summary"])
        ref_inputs = self.encoder.prepare_sample(sample["reference"])


        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
        inputs = {**src_inputs, **mt_inputs, **ref_inputs}
        
        targets = {"score": torch.tensor(sample["label"], dtype=torch.long)}
        
        
        return inputs, targets


        
        

    def prepare_valid_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = {k: [dic[k] for dic in sample] for k in sample[0]}

        src_inputs = self.encoder.prepare_sample(sample["document"])
        mt_inputs = self.encoder.prepare_sample(sample["summary"])
        ref_inputs = self.encoder.prepare_sample(sample["reference"])
        
    
        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
        inputs = {**src_inputs, **mt_inputs, **ref_inputs}
        
        
        score={}
        dimension=['coherence','consistency','fluency','relevance']
        for dim in dimension:
            score[dim]=sample[dim]
        return inputs,score
        

    def forward(
        self,
        input_segments: str=None,
        src_input_ids: torch.tensor=None,
        src_attention_mask: torch.tensor=None,
        mt_input_ids: torch.tensor=None,
        mt_attention_mask: torch.tensor=None,
        ref_input_ids: torch.tensor=None,
        ref_attention_mask: torch.tensor=None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        if self.combine_inputs:
            all_input_ids = list()
            inputs_group = input_segments.split('-')

            if 'hyp' in inputs_group:
                mt_input_ids[:, 0] = self.bos_for_segments['hyp']
                mt_input_ids_segments = list(x.masked_select(y.ne(0)) for x, y in zip(mt_input_ids.unbind(dim=0), mt_attention_mask.unbind(dim=0)))
                for x in mt_input_ids_segments:
                    x[-1] = self.eos_for_segments['hyp']
                all_input_ids.append(mt_input_ids_segments)

            if 'src' in inputs_group:
                src_input_ids[:, 0] = self.bos_for_segments['src']
                src_input_ids_segments = list(x.masked_select(y.ne(0)) for x, y in zip(src_input_ids.unbind(dim=0), src_attention_mask.unbind(dim=0)))
                for x in src_input_ids_segments:
                    x[-1] = self.eos_for_segments['src']
                all_input_ids.append(src_input_ids_segments)

            if 'ref' in inputs_group:
                ref_input_ids[:, 0] = self.bos_for_segments['ref']
                ref_input_ids_segments = list(x.masked_select(y.ne(0)) for x, y in zip(ref_input_ids.unbind(dim=0), ref_attention_mask.unbind(dim=0)))
                for x in ref_input_ids_segments:
                    x[-1] = self.eos_for_segments['ref']
                all_input_ids.append(ref_input_ids_segments)
            

            if len(inputs_group) == 3:
                all_input_concat_padded, all_input_seq_lens = cut_long_sequences3(all_input_ids, 380, self.pad_idx)
            else:
                all_input_concat_padded, all_input_seq_lens = cut_long_sequences2(all_input_ids, 380, self.pad_idx)

            
            cls_ids = torch.cat((all_input_seq_lens.new_zeros(size=(all_input_seq_lens.size(0), 1)), all_input_seq_lens.cumsum(dim=-1)), dim=-1).contiguous() 

            token_type_ids, token_type_masks = compute_token_type_ids(
                token_ids=all_input_concat_padded,
                cls_ids_with_sum_lens=cls_ids,
                buffered_position_ids=self.encoder.model.embeddings.position_ids,
                num_of_inputs=len(inputs_group),
                padding_value=len(self.input_segments),
                replaced_seg_ids=list(self.input_segments_dict[x] for x in inputs_group)
            )
            
            all_mask_concat_padded = all_input_concat_padded.ne(self.pad_idx).long()

            if self.reset_position_for_each_segment:
                position_ids = compute_position_ids_for_each_segment(all_input_seq_lens, len(self.input_segments), token_type_ids, self.encoder.model.embeddings.position_ids)
            else:
                position_ids = None

            embedded_sequences = self.get_sentence_embedding(
                all_input_concat_padded,
                all_mask_concat_padded,
                all_mask_concat_padded,
                token_type_ids if self.multiple_segment_embedding else None,
                token_type_masks,
                cls_ids[:, :-1],
                position_ids,
                len(input_segments),
                input_segments
            )

        else:
            if 'hyp' in self.input_segments:
                mt_sentemb = self.get_sentence_embedding(mt_input_ids, mt_attention_mask)
            else:
                mt_sentemb = None

            if 'src' in self.input_segments:
                src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
            else:
                src_sentemb = None            
            
            if 'ref' in self.input_segments:
                ref_sentemb = self.get_sentence_embedding(ref_input_ids, ref_attention_mask)
            else:
                ref_sentemb = None

            reps_for_regression = []
            if 'hyp' in self.pooling_rep:
                reps_for_regression.append(mt_sentemb)
            if 'src' in self.pooling_rep:
                reps_for_regression.append(src_sentemb)
            if 'ref' in self.pooling_rep:
                reps_for_regression.append(ref_sentemb)

            if 'ref_hyp_prod' in self.pooling_rep:
                prod_ref = mt_sentemb * ref_sentemb
                reps_for_regression.append(prod_ref)
            
            if 'ref_hyp_l1' in self.pooling_rep:
                diff_ref = torch.abs(mt_sentemb - ref_sentemb)
                reps_for_regression.append(diff_ref)
            
            if 'src_hyp_prod' in self.pooling_rep:
                prod_src = mt_sentemb * src_sentemb
                reps_for_regression.append(prod_src)
            
            if 'src_hyp_l1' in self.pooling_rep:
                diff_src = torch.abs(mt_sentemb - src_sentemb)
                reps_for_regression.append(diff_src)
                
            embedded_sequences = torch.cat(reps_for_regression, dim=1)
        score=self.estimator(embedded_sequences)
        return {"score": score}

    def process_text(self,raw_text):

        processed_text = re.sub(r"[%s]+" % punctuation, " ", raw_text)
        words_list = processed_text.lower().split()
        words_list = [word for word in words_list if word not in stopwords.words('english')]
        return words_list



    def read_csv(self, training_data_path: dict,  validation_data_path: str):

        human_annotation = {}
        with open(validation_data_path, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                pair_id = item['id'] + "-" + item['model_id']
                human_annotation[pair_id] = {
                    'document': item['text'],
                    'reference': item['references'][0],
                    'summary': item['decoded'],
                }
                avg_expert = {'coherence': 0, 'consistency': 0, 'fluency': 0, 'relevance': 0}
                for expert in item['expert_annotations']:
                    for k, v in expert.items():
                        avg_expert[k] += v / len(item['expert_annotations'])
                for k, v in avg_expert.items():
                    human_annotation[pair_id].update({k: v})

        training_data = {}
        for k in training_data_path.keys():
            with open(training_data_path[k], 'r', encoding='utf-8') as f:
                training_data[k] = json.loads(f.read())
                random.shuffle(training_data[k])

        return {
            'training_data': training_data,
            'validation_data': list(human_annotation.values())
        }


def cut_long_sequences2(all_input_concat: List[List[torch.Tensor]], maximum_length: int = 512, pad_idx: int = 1):
    all_input_concat = list(zip(*all_input_concat))
    collected_tuples = list()
    collected_lens = list()
    for tensor_tuple in all_input_concat:
        all_lens = tuple(len(x) for x in tensor_tuple)

        if sum(all_lens) > maximum_length:
            lengths = dict(enumerate(all_lens))
            lengths_sorted_idxes = list(x[0] for x in sorted(lengths.items(), key=lambda d: d[1], reverse=True))

            offset = ceil((sum(lengths.values()) - maximum_length) / 2)

            if min(all_lens) > (maximum_length // 2) and min(all_lens) > offset:
                lengths = dict((k, v - offset) for k, v in lengths.items())
            else:
                lengths[lengths_sorted_idxes[0]] = maximum_length - lengths[lengths_sorted_idxes[1]]

            new_lens = list(lengths[k] for k in range(0, len(tensor_tuple)))
            new_tensor_tuple = tuple(x[:y] for x, y in zip(tensor_tuple, new_lens))
            for x, y in zip(new_tensor_tuple, tensor_tuple):
                x[-1] = y[-1]
            collected_tuples.append(new_tensor_tuple)
            collected_lens.append(new_lens)
        else:
            collected_tuples.append(tensor_tuple)
            collected_lens.append(all_lens)

    concat_tensor = list(torch.cat(x, dim=0) for x in collected_tuples)
    all_input_concat_padded = pad_sequence(concat_tensor, batch_first=True, padding_value=pad_idx)
    collected_lens = torch.Tensor(collected_lens).long().to(all_input_concat_padded.device)

    return all_input_concat_padded, collected_lens


def cut_long_sequences3(all_input_concat: List[List[torch.Tensor]], maximum_length: int = 512, pad_idx: int = 1):
    all_input_concat = list(zip(*all_input_concat))
    collected_tuples = list()
    collected_lens = list()
    for tensor_tuple in all_input_concat:
        all_lens = tuple(len(x) for x in tensor_tuple)

        if sum(all_lens) > maximum_length:
            lengths = dict(enumerate(all_lens))
            lengths_sorted_idxes = list(x[0] for x in sorted(lengths.items(), key=lambda d: d[1], reverse=True))

            offset = ceil((sum(lengths.values()) - maximum_length) / 3)

            if min(all_lens) > (maximum_length // 3) and min(all_lens) > offset:
                lengths = dict((k, v - offset) for k, v in lengths.items())
            else:
                while sum(lengths.values()) > maximum_length:
                    if lengths[lengths_sorted_idxes[0]] > lengths[lengths_sorted_idxes[1]]:
                        offset = maximum_length - lengths[lengths_sorted_idxes[1]] - lengths[lengths_sorted_idxes[2]]
                        if offset > lengths[lengths_sorted_idxes[1]]:
                            lengths[lengths_sorted_idxes[0]] = offset
                        else:
                            lengths[lengths_sorted_idxes[0]] = lengths[lengths_sorted_idxes[1]]
                    elif lengths[lengths_sorted_idxes[0]] == lengths[lengths_sorted_idxes[1]] > lengths[lengths_sorted_idxes[2]]:
                        offset = (maximum_length - lengths[lengths_sorted_idxes[2]]) // 2
                        if offset > lengths[lengths_sorted_idxes[2]]:
                            lengths[lengths_sorted_idxes[0]] = lengths[lengths_sorted_idxes[1]] = offset
                        else:
                            lengths[lengths_sorted_idxes[0]] = lengths[lengths_sorted_idxes[1]] = lengths[lengths_sorted_idxes[2]]
                    else:
                        lengths[lengths_sorted_idxes[0]] = lengths[lengths_sorted_idxes[1]] = lengths[lengths_sorted_idxes[2]] = maximum_length // 3

            new_lens = list(lengths[k] for k in range(0, len(lengths)))
            new_tensor_tuple = tuple(x[:y] for x, y in zip(tensor_tuple, new_lens))
            
            for x, y in zip(new_tensor_tuple, tensor_tuple):
                x[-1] = y[-1]
            collected_tuples.append(new_tensor_tuple)
            collected_lens.append(new_lens)
        else:
            collected_tuples.append(tensor_tuple)
            collected_lens.append(all_lens)

    concat_tensor = list(torch.cat(x, dim=0) for x in collected_tuples)
    all_input_concat_padded = pad_sequence(concat_tensor, batch_first=True, padding_value=pad_idx)
    collected_lens = torch.Tensor(collected_lens).long().to(all_input_concat_padded.device)

    return all_input_concat_padded, collected_lens


def compute_token_type_ids(token_ids: torch.Tensor, cls_ids_with_sum_lens: torch.Tensor, buffered_position_ids: torch.Tensor, num_of_inputs: int, padding_value: int, replaced_seg_ids: List[int]):
    max_seq_lens = token_ids.size(1)
    type_ids_meta = buffered_position_ids[:, :max_seq_lens].expand_as(token_ids)
    type_ids = type_ids_meta.clone().detach().fill_(padding_value)

    for i in range(0, num_of_inputs):
        sub_mask = type_ids_meta.ge(cls_ids_with_sum_lens[:, i: i + 1]) & type_ids_meta.lt(cls_ids_with_sum_lens[:, i + 1: i + 2])
        type_ids.masked_fill_(mask=sub_mask, value=replaced_seg_ids[i])
    

    mask_for_pad = type_ids.eq(padding_value)
    type_ids_for_out = type_ids.masked_fill(mask=mask_for_pad, value=replaced_seg_ids[-1])

    
    return type_ids_for_out, type_ids


def compute_attention_masks_for_regions(
        seq_lens: torch.Tensor,
        excluded_regions: List[tuple],
        buffered_position_ids: torch.Tensor,
        num_of_inputs: int,
        token_type_ids: torch.Tensor,
        cls_token_ids: torch.Tensor,
        cls_from_cls_to_all: bool,
        cls_from_all_to_cls: bool
        ):

    cumsum_seq_lens = seq_lens.cumsum(dim=-1)
    max_seq_lens = int(cumsum_seq_lens[:, -1].max())
    batch_size = seq_lens.size(0)
    type_ids_meta = buffered_position_ids[:,:max_seq_lens].view(1, 1, -1).repeat(batch_size, 1, 1) 
    collected_masks = list()
    pivots = cls_token_ids.unsqueeze(dim=-1)  

    for i in range(0, num_of_inputs):
        temp_collected_mask = type_ids_meta.lt(pivots[:, -1:]).long()
        sub_excluded_regions = list(filter(lambda x: x[0] == i, excluded_regions))
        for sub_excluded_region in sub_excluded_regions:
            index = sub_excluded_region[-1]
            temp_mask = (type_ids_meta.ge(pivots[:, index:index + 1]) & type_ids_meta.lt(pivots[:, index + 1:index + 2])).long()
            temp_collected_mask = temp_collected_mask - temp_mask
        collected_masks.append(temp_collected_mask)

    collected_masks = torch.cat(collected_masks, dim=1)  
    new_idxes = token_type_ids + buffered_position_ids[:, :batch_size].view(-1, 1) * num_of_inputs
    output_masks = collected_masks.view(batch_size * num_of_inputs, -1).index_select(dim=0, index=new_idxes.view(-1)).view(batch_size, max_seq_lens, -1)
    if cls_from_cls_to_all:
        output_masks[:, :, 0] = 1
    if cls_from_all_to_cls:
        temp_mask = type_ids_meta.lt(pivots[:, -1:]).view(batch_size, -1).long()
        output_masks[:, 0, :] = temp_mask

    return output_masks


def compute_position_ids_for_each_segment(
        seq_lens: torch.Tensor,
        num_of_inputs: int,
        token_type_ids: torch.Tensor,
        buffered_position_ids: torch.Tensor
    ):

    cumsum_seq_lens = seq_lens.cumsum(dim=-1) 
    max_seq_lens = int(cumsum_seq_lens[:, -1].max())
    batch_size = seq_lens.size(0)
    position_ids = buffered_position_ids[:,:max_seq_lens].repeat(batch_size, 1)  
    offset = position_ids.new_zeros(size=position_ids.size(), dtype=position_ids.dtype)
    
    for i in range(1, num_of_inputs):
        temp_mask = position_ids.ge(cumsum_seq_lens[:, i - 1: i])
        temp_offset = temp_mask.long() * seq_lens[:, i - 1: i]
        offset = offset + temp_offset

    position_ids = position_ids - offset
    
    return position_ids

