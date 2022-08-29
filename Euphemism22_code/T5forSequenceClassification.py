#!/usr/bin/env python

"""
https://notebooks.githubusercontent.com/view/ipynb?azure_maps_enabled=false&color_mode=auto&commit=02a6207a6807ef8d5d60c0e5c11aac9f65c7cf55&
enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f676973742f61766964616c652f64633761323665623363666663393030373562
663130306531356235393530662f7261772f303261363230376136383037656638643564363063306535633131616163396636356337636635352f727574352d656e636f6465
722e6970796e62&enterprise_enabled=false&logged_in=false&nwo=avidale%2Fdc7a26eb3cffc90075bf100e15b5950f&path=rut5-encoder.ipynb&
repository_id=109476206&repository_type=Gist

https://colab.research.google.com/gist/avidale/dc7a26eb3cffc90075bf100e15b5950f/rut5-encoder.ipynb

"""
from transformers import T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, Embedding
import torch.nn as nn
import copy


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class T5ForSequenceClassification(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.shared = Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if attention_mask is None:
            total_output = outputs[0]  # batch, seq_len, emb_dim
            pooled_output = total_output.mean(dim=1)
        else:
            pooled_output = mean_pooling(outputs, attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
