"""Fine-tune DistilBERT on a tiny toy sentiment dataset."""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SENTENCES = [
    ("I love this product", 1),
    ("This is terrible", 0),
    ("Absolutely amazing experience", 1),
    ("Worst purchase ever", 0),
    ("Pretty good, would buy again", 1),
    ("Not good at all", 0),
] * 10


class ToyDataset(Dataset):
    def __init__(self, tokenizer, max_len: int = 32):
        self.items = [
            (tokenizer(text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt"), label)
            for text, label in SENTENCES
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        enc, label = self.items[idx]
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.long),
        )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)

    ds = ToyDataset(tokenizer)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(2):
        for ids, mask, labels in dl:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
        print(f"epoch {epoch + 1}: loss {out.loss.item():.4f}")


if __name__ == "__main__":
    main()
