import random
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn


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


data = torch.load("model.pth", map_location=torch.device('cpu'))

model_state = data["model_state"]
tags = data["tags"]
faq = data['faq']
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

model = IntentClassifier(len(tags))
model.load_state_dict(model_state)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def classify_question(sentence):
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=50,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    _, prediction = torch.max(outputs, dim=1)
    return tags[prediction.item()]


def get_response(predict):
    for intent_data in faq:
        if intent_data['tag'] == predict:
            return random.choice(intent_data['answers'])


async def answer(text):
    predict = classify_question(text)
    response = get_response(predict)
    return response

