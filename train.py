import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data import load_data, embedding_matrix
from model import SiameseBiLSTM

vocab, train, val, test = load_data()
emb = embedding_matrix(vocab)
model = SiameseBiLSTM(emb)
loss_fn = nn.BCEWithLogitsLoss()  # sigmoid + binary cross-entropy
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
loader = DataLoader(TensorDataset(*train), batch_size=32, shuffle=True)

def predict(data):
    model.eval()
    with torch.no_grad():
        return (torch.sigmoid(model(data[0], data[1])) > 0.5).int().numpy()

best_f1, patience, bad = 0, 4, 0
for epoch in range(20):
    model.train()
    for a, b, y in loader:
        opt.zero_grad()
        loss = loss_fn(model(a, b), y)
        loss.backward()
        opt.step()
    f1 = f1_score(val[2].numpy(), predict(val))
    print(f"epoch {epoch+1}  loss {loss.item():.3f}  val F1 {f1:.3f}")
    if f1 > best_f1:
        best_f1, bad = f1, 0
        torch.save(model.state_dict(), "model.pt")
    else:
        bad += 1
        if bad >= patience:
            print("early stopping")
            break

pickle.dump(vocab, open("vocab.pkl", "wb"))
model.load_state_dict(torch.load("model.pt"))
pred, true = predict(test), test[2].numpy()
print("Accuracy :", accuracy_score(true, pred))
print("Precision:", precision_score(true, pred))
print("Recall   :", recall_score(true, pred))
print("F1       :", f1_score(true, pred))
print("Confusion matrix:\n", confusion_matrix(true, pred))