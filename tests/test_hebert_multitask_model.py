import torch
from transformers import BertConfig
from training.hebert_multitask_model import HeBERTMultiTaskModel


def tiny_model():
    config=BertConfig(vocab_size=50,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,num_labels=5)
    config.num_token_labels=5; config.num_sequence_labels=2
    return HeBERTMultiTaskModel(config)

def test_two_heads_shapes():
    model=tiny_model(); ids=torch.randint(0,50,(2,6)); mask=torch.ones_like(ids)
    out=model(input_ids=ids,attention_mask=mask)
    assert out.token_logits.shape==(2,6,5)
    assert out.sequence_logits.shape==(2,2)

def test_both_losses():
    model=tiny_model(); ids=torch.randint(0,50,(2,6)); mask=torch.ones_like(ids)
    tok=torch.tensor([[0,1,2,-100,-100,-100],[0,0,0,0,0,0]])
    seq=torch.tensor([1,0])
    out=model(input_ids=ids,attention_mask=mask,token_labels=tok,sequence_labels=seq)
    assert out.loss is not None and out.token_loss is not None and out.sequence_loss is not None

def test_sequence_only_loss():
    model=tiny_model(); ids=torch.randint(0,50,(2,6)); mask=torch.ones_like(ids)
    tok=torch.full((2,6),-100); seq=torch.tensor([1,0])
    out=model(input_ids=ids,attention_mask=mask,token_labels=tok,sequence_labels=seq)
    assert out.token_loss is None and out.sequence_loss is not None
