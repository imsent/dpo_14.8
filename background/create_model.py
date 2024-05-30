import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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


train_data, temp_data = train_test_split(trains, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


train_dataset = DataSet(train_data, tags, tokenizer)
val_dataset = DataSet(val_data, tags, tokenizer)
test_dataset = DataSet(test_data, tags, tokenizer)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


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

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

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
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy.item())
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')


    model.eval()
    val_loss = 0
    val_correct_predictions = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            val_correct_predictions += torch.sum(preds == labels)

    val_avg_loss = val_loss / len(val_loader)
    val_accuracy = val_correct_predictions.double() / len(val_loader.dataset)
    val_losses.append(val_avg_loss)
    val_accuracies.append(val_accuracy.item())
    print(f'Validation Loss: {val_avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')


model.eval()
test_loss = 0
test_correct_predictions = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        test_correct_predictions += torch.sum(preds == labels)

test_avg_loss = test_loss / len(test_loader)
test_accuracy = test_correct_predictions.double() / len(test_loader.dataset)
print(f'Test Loss: {test_avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')


data = {
    "model_state": model.state_dict(),
    "tags": tags,
    "tokenizer": tokenizer,
    "faq": faq
}

torch.save(data, "server/model.pth")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()


print("Final Test Results:")
print(f'Test Loss: {test_avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
