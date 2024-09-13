import torch
import torch.nn as nn
from torch.nn import functional as F
import joblib
from torchtext.data.utils import get_tokenizer
vocab = joblib.load('vocab.pkl')

embedding_dim = 300
vocab_size = len(vocab)
hidden_size = 900
output_size = 9


class LSTMTextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_class):
        super(LSTMTextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, text):
        embedded = self.embedding(text)
        output, _ = self.lstm(embedded)
        last_hidden = output[:, -1, :]  # Use the last hidden state
        return F.log_softmax(self.fc(last_hidden), dim=1)

model = LSTMTextClassificationModel(vocab_size, embedding_dim, hidden_size, output_size)
model.load_state_dict(torch.load('new_model.pth'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

tokenizer = get_tokenizer("basic_english")
text_pipeline = lambda x: vocab(tokenizer(x))

sample = ["what will my career be like"]
sample_processed = [text_pipeline(text) for text in sample]

sample_tensor = torch.nn.utils.rnn.pad_sequence(
    [torch.tensor(x, dtype=torch.int64) for x in sample_processed],
    batch_first=True,
    padding_value=0
).to(device)

labels = {0: '5 Card Tarot Spread for Decision Making', 1: '10 Card Tarot Spread for Self Growth and Personal Development',
          2: 'Repeating Tarot Card Spread', 3: 'Are You Ready for Love Tarot Spread',
          4: 'Chakras Love Tarot Spread', 5: 'New Years Tarot Spread',
          6: 'Finding Love Relationship Tarot', 7: 'Facing Challenges Career Tarot Spread'}

with torch.no_grad():
    sentiment = model(sample_tensor)
    probabilities = torch.exp(sentiment)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    print(f"Predicted class: {labels.get(predicted_class, 'Unknown')}")
