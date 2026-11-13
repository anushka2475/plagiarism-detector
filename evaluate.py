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

print("Plagiarism-by-Meaning Detector")
print("Type two sentences and I'll tell you if they mean the same thing.")

while True:
    s1 = input("\nSentence 1 (or 'q' to quit): ")
    if s1.strip().lower() == "q":
        break
    s2 = input("Sentence 2: ")
    score = similarity(s1, s2)
    verdict = "SAME MEANING (possible copy)" if score > 0.5 else "DIFFERENT"
    print(f"Similarity: {score:.2f} -> {verdict}")