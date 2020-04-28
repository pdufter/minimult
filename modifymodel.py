from typing import Any
import torch


def zero_out_embeddings(embeddings: Any) -> None:
    myweights = embeddings.weight.data.detach().cpu().numpy()
    myweights[:, :] = 0.0
    embeddings.weight.data = torch.tensor(myweights.astype('float32'))


def fix_embeddings(embeddings: Any) -> None:
    embeddings.training = False
    embeddings.weight.requires_grad = False


def delete_position_segment_embeddings(model: Any) -> None:
    zero_out_embeddings(model.bert.embeddings.position_embeddings)
    fix_embeddings(model.bert.embeddings.position_embeddings)
    zero_out_embeddings(model.bert.embeddings.token_type_embeddings)
    fix_embeddings(model.bert.embeddings.token_type_embeddings)
