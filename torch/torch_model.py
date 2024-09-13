import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, random_split, TensorDataset
import joblib
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

filename = "tarot_classification_data.csv"
dataset = pd.read_csv(filename)
ps = PorterStemmer()

tokenizer = get_tokenizer("basic_english")

def tokenize_text(data_iter):
    tokens = []
    for text in data_iter:
        tokens.append(tokenizer(text))
    return tokens

tokens = tokenize_text(dataset['Text'])
vocab = build_vocab_from_iterator(tokens, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def text_to_indices(text):
    tokens = tokenizer(text)
    indices = vocab(tokens)
    return indices

def label_to_index(label):
    return int(label)

X = [text_to_indices(text) for text in dataset['Text']]
Y = [label_to_index(label) for label in dataset['Label']]

X = [torch.tensor(x, dtype=torch.int64) for x in X]
Y = torch.tensor(Y, dtype=torch.int64)
X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(split_valid_, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

class LSTMTextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_class):
        super(LSTMTextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, text):
        embedded = self.embedding(text)
        output, _ = self.lstm(embedded)
        last_hidden = output[:, -1, :]
        return F.log_softmax(self.fc(last_hidden), dim=1)

vocab_size = len(vocab)
embed_dim = 300
hidden_size = 900
output_size = 9
model = LSTMTextClassificationModel(vocab_size, embed_dim, hidden_size, output_size).to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500

    for idx, (text, label) in enumerate(dataloader):
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (text, label) in enumerate(dataloader):
            text, label = text.to(device), label.to(device)
            predicted_label = model(text)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

epoches = 10
total_accu = None

for epoch in range(1, epoches + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print(
        "|epoch {:3d}  valid accuracy {:8.3f} ".format(
            epoch, accu_val
        )
    )

accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))

torch.save(model.state_dict(), 'new_model.pth')
joblib.dump(vocab, 'vocab.pkl')

def predict(text):
    indices = text_to_indices(text)
    text_tensor = torch.tensor(indices, dtype=torch.int64).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(text_tensor)
        return output.argmax(1).item()

labels = {0: '5 Card Tarot Spread for Decision Making', 1: '10 Card Tarot Spread for Self Growth and Personal Development', 2: 'Repeating Tarot Card Spread', 3: 'Are You Ready for Love Tarot Spread', 4: 'Chakras Love Tarot Spread', 5: 'New Years Tarot Spread', 6: 'Finding Love Relationship Tarot', 7: 'Facing Challenges Career Tarot Spread'}

ex_text_str = "what will my career be like"
predicted_label = predict(ex_text_str)
print(f"This text is classified as: {labels[predicted_label]}")
