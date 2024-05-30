import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

with open('faq_dataset.json', 'r', encoding='utf-8') as f:
    faq = json.load(f)

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

tags = []
trains = []

for x in faq:
    tag = x['tag']
    tags.append(tag)
    for pattern in x['questions']:
        trains.append((pattern, tag))

tags = sorted(set(tags))


class DataSet(Dataset):
    def __init__(self, trains, tags, tokenizer, max_len=50):
        self.trains = trains
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}

    def __len__(self):
        return len(self.trains)

    def __getitem__(self, idx):
        sentence, tag = self.trains[idx]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        label = torch.tensor(self.tag_to_idx[tag], dtype=torch.long)
        return input_ids, attention_mask, label


dataset = DataSet(trains, tags, tokenizer)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IntentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(IntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


model = IntentClassifier(len(tags)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions.double() / len(train_loader.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

data = {
    "model_state": model.state_dict(),
    "tags": tags,
    "tokenizer": tokenizer,
    "faq": faq
}

torch.save(data, "server/model.pth")
