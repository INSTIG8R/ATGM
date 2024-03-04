import mxnet as mx
from bert_embedding import BertEmbedding
import torch
import numpy as np

def BERT(x):
    # #For using GPU - mxnet doesn't support cuda version above 11.2
    # ctx = mx.gpu(0)
    # bert_embedding = BertEmbedding(ctx=ctx)

    ctx = mx.cpu(0)
    bert_embedding = BertEmbedding(ctx=ctx)
    token = bert_embedding(x)
    token_np = np.array(token[0][1])
    token_torch = torch.Tensor(token_np)

    return token_torch 

# if __name__ == "__main__":
#     x = ["This is BERT module"]
#     x_embed = BERT(x)
#     print(x_embed.shape) # (4,768)