import re
import numpy as np
import torch
from datasets import load_dataset

MAX_LEN = 40

def tokenize(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()

def build_vocab(token_lists):
    vocab = {"<pad>": 0, "<unk>": 1}
    for toks in token_lists:
        for w in toks:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab

def encode(text, vocab):
    ids = [vocab.get(w, 1) for w in tokenize(text)][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

def embedding_matrix(vocab, dim=100):
    m = np.random.normal(0, 0.1, (len(vocab), dim)).astype("float32")
    m[0] = 0
    try:
        from huggingface_hub import snapshot_download
        from gensim.models import KeyedVectors
        path = snapshot_download("fse/glove-wiki-gigaword-100")
        glove = KeyedVectors.load(path + "/glove-wiki-gigaword-100.model")
        for w, i in vocab.items():
            if w in glove:
                m[i] = glove[w]
        print("GloVe loaded")
    except Exception as e:
        print("GloVe failed, using random embeddings:", e)
    return torch.tensor(m)

def to_tensors(split, vocab):
    a = torch.tensor([encode(s, vocab) for s in split["sentence1"]])
    b = torch.tensor([encode(s, vocab) for s in split["sentence2"]])
    y = torch.tensor(split["label"], dtype=torch.float32)
    return a, b, y

def load_data():
    ds = load_dataset("nyu-mll/glue", "mrpc")
    train_val = ds["train"].train_test_split(test_size=0.1, seed=42)
    train, val, test = train_val["train"], train_val["test"], ds["validation"]
    vocab = build_vocab(
        tokenize(s) for split in (train, val, test)
        for col in ("sentence1", "sentence2") for s in split[col]
    )
    return vocab, to_tensors(train, vocab), to_tensors(val, vocab), to_tensors(test, vocab)