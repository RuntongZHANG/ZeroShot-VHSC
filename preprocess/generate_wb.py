import torch
import numpy as np
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

vsac_classes = ['background', 'cat', 'dog', 'horse', 'cow', 'sheep',
               'airplane', 'bicycle', 'motor', 'car', 'bus',
               'train', 'bear', 'elephant', 'giraffe', 'zebra', 'truck']

def single_templete(save_path, class_names, model):
    with torch.no_grad():
        texts = torch.cat([clip.tokenize("a photo of a {%s}"%c) for c in class_names]).cuda()
        text_embeddings = model.encode_text(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        np.save(save_path, text_embeddings.cpu().numpy())
    return text_embeddings

save_path='vsac_for_JoEm.npy'
text_embeddings = single_templete(save_path, vsac_classes, model)
