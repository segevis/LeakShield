from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import BertConfig, BertForTokenClassification, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class HeBERTMultiTaskOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    token_loss: Optional[torch.Tensor] = None
    sequence_loss: Optional[torch.Tensor] = None
    token_logits: Optional[torch.Tensor] = None
    sequence_logits: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None


class HeBERTMultiTaskModel(BertPreTrainedModel):
    """One shared HeBERT encoder with token and sequence classification heads."""

    config_class = BertConfig

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        token_labels = int(getattr(config, "num_token_labels", getattr(config, "num_labels", 2)))
        sequence_labels = int(getattr(config, "num_sequence_labels", 2))
        config.num_token_labels = token_labels
        config.num_sequence_labels = sequence_labels
        config.architectures = [self.__class__.__name__]

        self.bert = BertModel(config, add_pooling_layer=False)
        dropout_rate = getattr(config, "classifier_dropout", None)
        if dropout_rate is None:
            dropout_rate = config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_rate)
        self.token_classifier = nn.Linear(config.hidden_size, token_labels)
        self.sequence_classifier = nn.Linear(config.hidden_size, sequence_labels)
        self.post_init()

    @classmethod
    def from_token_checkpoint(
        cls,
        checkpoint: str,
        *,
        num_sequence_labels: int = 2,
        token_loss_weight: float = 1.0,
        sequence_loss_weight: float = 1.0,
    ) -> "HeBERTMultiTaskModel":
        token_model = BertForTokenClassification.from_pretrained(checkpoint)
        config = token_model.config
        config.num_token_labels = token_model.classifier.out_features
        config.num_sequence_labels = num_sequence_labels
        config.token_loss_weight = float(token_loss_weight)
        config.sequence_loss_weight = float(sequence_loss_weight)
        config.id2tokenlabel = dict(config.id2label)
        config.tokenlabel2id = dict(config.label2id)
        config.id2sequencelabel = {0: "NON_LEAK", 1: "LEAK"}
        config.sequencelabel2id = {"NON_LEAK": 0, "LEAK": 1}

        model = cls(config)
        model.bert.load_state_dict(token_model.bert.state_dict(), strict=True)
        model.token_classifier.load_state_dict(token_model.classifier.state_dict(), strict=True)
        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        sequence_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> HeBERTMultiTaskOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = self.dropout(outputs.last_hidden_state)
        token_logits = self.token_classifier(hidden)
        sequence_logits = self.sequence_classifier(hidden[:, 0])

        token_loss = None
        sequence_loss = None
        if token_labels is not None and torch.any(token_labels != -100):
            token_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                token_logits.view(-1, self.config.num_token_labels), token_labels.view(-1)
            )
        if sequence_labels is not None and torch.any(sequence_labels != -100):
            sequence_loss = nn.CrossEntropyLoss(ignore_index=-100)(sequence_logits, sequence_labels)

        losses: list[torch.Tensor] = []
        if token_loss is not None:
            losses.append(float(getattr(self.config, "token_loss_weight", 1.0)) * token_loss)
        if sequence_loss is not None:
            losses.append(float(getattr(self.config, "sequence_loss_weight", 1.0)) * sequence_loss)
        loss = sum(losses) if losses else None

        result = HeBERTMultiTaskOutput(
            loss=loss,
            token_loss=token_loss,
            sequence_loss=sequence_loss,
            token_logits=token_logits,
            sequence_logits=sequence_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        if return_dict:
            return result
        return tuple(v for v in result.values() if v is not None)
