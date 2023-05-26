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
import os
import json
import math
from comet.models import load_from_checkpoint
from pytorch_lightning import seed_everything
import jsonlines
import multiprocessing
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr, kendalltau
from nltk import sent_tokenize
from tqdm import tqdm
import numpy as np
import pandas as pd
from jsonargparse import ArgumentParser

def write_predict(write_path, data, eval_scores):
    for i in range(len(data)):
        data[i]['predict_scores'] = {"coherence": eval_scores['Entire_Summary'][i], "consistency": eval_scores['Sentence_by_Sentence'][i], "fluency": eval_scores['Sentence_by_Sentence'][i], "relevance": eval_scores['Entire_Summary'][i]}
    with open(write_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(data_path):
    with open(data_path) as f:
        data = json.loads(f.read())
    return data


def calculate_correlation(pred_score, human_score, dim, result):
    assert len(pred_score) == len(human_score)
    if dim not in result:
        result[dim] = [0] * 3
    result[dim][0] += pearsonr(pred_score, human_score)[0]
    result[dim][1] += spearmanr(pred_score, human_score)[0]
    result[dim][2] += kendalltau(pred_score, human_score)[0]
    return result


def print_correlations(result, output_url):
    output = pd.DataFrame(columns=['Dimensions', 'Pearson', 'Spearman', 'Kendall'])
    index = 0
    for dim in result:
        output.loc[index] = [dim, round(result[dim][0], 6), round(result[dim][1], 6), round(result[dim][2], 6)]
        index += 1
    output.to_csv(output_url + '.csv', index=None)


def get_unique_value(data, key):
    value = set()
    for i in range(len(data)):
        if data[i][key] not in value:
            value.add(data[i][key])
    return list(value)


def correlation_for_summ(data, output_url):
    dimensions = ['coherence', 'consistency', 'fluency', 'relevance']

    result = {}
    docs = get_unique_value(data, 'doc_id')
    for dim in dimensions:
        valid_cnt = 0
        for doc_idx in docs:
            pred_score, human_score = [], []
            for i in range(len(data)):
                if data[i]['doc_id'] == doc_idx:
                    pred_score.append(data[i]['predict_scores'][dim])
                    human_score.append(data[i]['scores'][dim])
            if len(set(pred_score)) == 1 or len(set(human_score)) == 1:
                continue
            result = calculate_correlation(pred_score, human_score, dim, result)
            valid_cnt += 1
        for j in range(3):
            result[dim][j] /= valid_cnt
    print_correlations(result, output_url)



parser = ArgumentParser(description="Command for testing UMSE model.")
parser.add_argument("--data", type=str, default=None, help='the path of data')
parser.add_argument("--model", type=str, default=None, help='the path of model')
parser.add_argument("--config", type=str, default=None, help='the path of model config')
parser.add_argument("--output", type=str, default=None, help='the path of output')
cfg = parser.parse_args()

def main() -> None:


    # load evaluation model (UMSE)
    model = load_from_checkpoint(cfg.model, cfg.config)
    device = torch.device("cuda")
    model.to(device)
    model.eval()


    # load SummEval dataset
    human_annotation = {}
    with open(cfg.data, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            pair_id = item['id'] + "-" + item['model_id']
            human_annotation[pair_id] = {
                'doc_id': item['id'],
                'system_id': item['model_id'],
                'source': item['text'],
                'reference': item['references'][0],
                'system_output': item['decoded'],
            }

            avg_expert = {'coherence': 0, 'consistency': 0, 'fluency': 0, 'relevance': 0}
            for expert in item['expert_annotations']:
                for k, v in expert.items():
                    avg_expert[k] += v / len(item['expert_annotations'])
            overall = 0
            for k in avg_expert.keys():
                overall += avg_expert[k] / 4
            avg_expert['overall'] = overall
            human_annotation[pair_id]['scores'] = avg_expert

    dataloader = DataLoader(
        dataset=list(human_annotation.values()),
        batch_size=1,
        num_workers=multiprocessing.cpu_count(),
    )
    Softmax = nn.Softmax()

    # predit scores (whole summary)
    eval_scores={
        'hyp-ref':{
            'Entire_Summary':[],
            'Sentence_by_Sentence':[]
        },
        'hyp-src': {
            'Entire_Summary': [],
            'Sentence_by_Sentence': []
        },
        'hyp-src-ref': {
            'Entire_Summary': [],
            'Sentence_by_Sentence': []
        }
    }
    for scenario in eval_scores.keys():

        if scenario=='hyp-src-ref':
            # Aggregation
            for mode in eval_scores['hyp-ref'].keys():
                for i in range(len(eval_scores['hyp-ref'][mode])):
                    eval_scores['hyp-src-ref'][mode].append((eval_scores['hyp-ref'][mode][i] + eval_scores['hyp-src'][mode][i]) / 2)

        else:
            for mode in eval_scores[scenario].keys():
                for batch_input in tqdm(dataloader):
                
                    if mode=='Entire_Summary':
                    
                        src_inputs = model.encoder.prepare_sample(batch_input["source"])
                        ref_inputs = model.encoder.prepare_sample(batch_input["reference"])
                        mt_inputs = model.encoder.prepare_sample(batch_input["system_output"])

                        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
                        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
                        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}

                        inputs = {**src_inputs, **mt_inputs, **ref_inputs}
                        for k in inputs.keys():
                            inputs[k] = inputs[k].to(device)

                        batch_prediction = model.forward(input_segments=scenario, **inputs)
                        scores = Softmax(batch_prediction['score'])[:, 1].view(-1).tolist()
                        eval_scores[scenario][mode].extend(scores)


                    elif mode=='Sentence_by_Sentence':
     
                        sent_list=sent_tokenize(batch_input["system_output"][0])
                        src_inputs = model.encoder.prepare_sample(batch_input["source"]*len(sent_list))
                        ref_inputs = model.encoder.prepare_sample(batch_input["reference"]*len(sent_list))
                        mt_inputs = model.encoder.prepare_sample(sent_list)
                        
                        
                        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
                        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
                        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
                        
                        inputs = {**src_inputs, **mt_inputs, **ref_inputs}
                        for k in inputs.keys():
                            inputs[k]=inputs[k].to(device)
                            
                        batch_prediction = model.forward(input_segments=scenario, **inputs)
                        scores=Softmax(batch_prediction['score'])[:,1].view(-1).tolist()
                        eval_scores[scenario][mode].append(np.mean(scores))
                                
         
        write_predict(cfg.output + scenario + '.json', list(human_annotation.values()), eval_scores[scenario])
        data = load_json(cfg.output + scenario + '.json')
        correlation_for_summ(data,cfg.output + scenario)




if __name__ == '__main__':
    main()
