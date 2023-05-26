import torch
from typing import Dict, List
from transformers import RobertaModel, RobertaPreTrainedModel,RobertaConfig,RobertaTokenizer

class PrefixEncoder(torch.nn.Module):

    def __init__(self, prefix_projection,pre_seq_len,hidden_size,num_hidden_layers,prefix_hidden_size):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values



class PrefixRobertaEncoder(RobertaPreTrainedModel):

    def __init__(self, pretrained_model):
        self.config = RobertaConfig.from_pretrained(pretrained_model)
        super().__init__(self.config)
        self.model = RobertaModel.from_pretrained(pretrained_model, add_pooling_layer=False)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
        self.pre_seq_len = 128
        self.hidden_dropout_prob=0.1
        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // self.config.num_attention_heads
        self.d_model = self.config.hidden_size 
        self.prefix_tokens = torch.ones(self.pre_seq_len).long().to(self.model.device)
        self.prefix_encoder = PrefixEncoder(prefix_projection=False, pre_seq_len=self.pre_seq_len, hidden_size=self.d_model, num_hidden_layers=self.n_layer, prefix_hidden_size=512)
        self.dropout = torch.nn.Dropout(self.hidden_dropout_prob)
        self.model.encoder.output_hidden_states = True
        
        for param in self.model.parameters():
            param.requires_grad=False

    def get_prompt(self,task, batch_size):
        
        if task=='hyp-src':
            prefix_token=list(range(0,self.pre_seq_len))
        
        elif task=='hyp-ref':
            prefix_token=list(range(self.pre_seq_len-1,-1,-1))
        
        elif task == 'hyp-src-ref':
            prefix_token_part1=list(range(1,self.pre_seq_len,2))
            prefix_token_part2=list(range(0,self.pre_seq_len,2))
            prefix_token_part1.extend(prefix_token_part2)
            prefix_token=prefix_token_part1
        
        self.prefix_tokens = torch.Tensor(prefix_token).long().to(self.model.device)

        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    
  
    @classmethod
    def from_pretrained(cls, pretrained_model: str):
        return PrefixRobertaEncoder(pretrained_model)
    
    def forward(self, task, batch_size, input_ids, attention_mask, token_type_ids, position_ids):
        

        batch_size=input_ids.shape[0]
        past_key_values = self.get_prompt(task, batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        last_hidden_states, _, all_layers  = self.model(input_ids = input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids, 
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=False,
                past_key_values=past_key_values)

        return {
             "sentemb": last_hidden_states[:, 0, :],
             "wordemb": last_hidden_states,
             "all_layers": all_layers,
             "attention_mask": attention_mask,
        }
   

    def prepare_sample(self, sample: List[str]) -> Dict[str, torch.Tensor]:
   
        tokenizer_output = self.tokenizer(
            sample,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_positions - 2,
        )
        return tokenizer_output
    
    def freeze(self) -> None:
        """Frezees the entire encoder."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfrezees the entire encoder."""
        for param in self.parameters():
            param.requires_grad = True
    
    @property
    def output_units(self):
        """Max number of tokens the encoder handles."""
        return self.model.config.hidden_size
    
    
    @property
    def num_layers(self):
        """Number of model layers available."""
        return self.model.config.num_hidden_layers + 1
    
    @property
    def max_positions(self):
        """Max number of tokens the encoder handles."""
        return self.model.config.max_position_embeddings

        
    def freeze_embeddings(self) -> None:
            """Frezees the embedding layer."""
            print()
            print('Freeze embedding function from Roberta.py')
            for param in self.model.embeddings.parameters():
                param.requires_grad = False
        
    def layerwise_lr(self, lr: float, decay: float, weight_decay: float = 0.0):
        """
        :param lr: Learning rate for the highest encoder layer.
        :param decay: decay percentage for the lower layers.
    
        :return: List of model parameters with layer-wise decay learning rate
        """
        # Embedding Layer
        opt_parameters = [
            {
                "params": self.model.embeddings.parameters(),
                "lr": lr * decay ** (self.num_layers),
                "weight_decay": weight_decay
            }
        ]
        # All layers
        opt_parameters += [
            {
                "params": self.model.encoder.layer[i].parameters(),
                "lr": lr * decay ** i,
                "weight_decay": weight_decay
            }
            for i in range(self.num_layers - 2, 0, -1)
        ]
        return opt_parameters
    
        
        



