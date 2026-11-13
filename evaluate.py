import pickle
import torch
from data import encode
from model import SiameseBiLSTM

vocab = pickle.load(open("vocab.pkl", "rb"))
model = SiameseBiLSTM(torch.zeros(len(vocab), 100))
model.load_state_dict(torch.load("model.pt"))
model.eval()

def similarity(s1, s2):
    a = torch.tensor([encode(s1, vocab)])
    b = torch.tensor([encode(s2, vocab)])
    with torch.no_grad():
        return torch.sigmoid(model(a, b)).item()

pairs = [
    ("The company reported record profits this year.", "This year the firm posted its highest ever earnings."),
    ("The company reported record profits this year.", "The company reported record losses this year."),
    ("The students finished the assignment before the deadline.", "The assignment was completed by the students ahead of time."),
    ("The students finished the assignment before the deadline.", "The weather in Kolkata is humid in July."),
    ("She bought a new laptop for college.", "She purchased a new computer for university."),
]
for s1, s2 in pairs:
    print(f"{similarity(s1, s2):.2f}  |  {s1}  <->  {s2}")