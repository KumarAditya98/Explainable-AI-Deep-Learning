import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Model, GPT2Tokenizer
from torch.utils.data import DataLoader, TensorDataset


# Define the GPT-2 based Discriminator
class GPT2Discriminator(nn.Module):
    def __init__(self, gpt_model_name: str, hidden_size: int, max_seq_len: int):
        super(GPT2Discriminator, self).__init__()

        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.classifier = nn.Linear(hidden_size * max_seq_len, 2)

    def forward(self, input_ids, attention_mask):
        gpt_output, _ = self.gpt2model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        batch_size = gpt_output.shape[0]
        linear_output = self.classifier(gpt_output.view(batch_size, -1))
        return linear_output


# Tokenize and encode sequences
def encode_sequences(texts, tokenizer, max_length):
    return tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")


# Sample Data
ground_truth_texts = []
machine_texts = []
all_texts = ground_truth_texts + machine_texts
labels = [1] * len(ground_truth_texts) + [0] * len(machine_texts)

# Tokenization
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
encoded_data = encode_sequences(all_texts, tokenizer, max_length=512)
dataset = TensorDataset(encoded_data['input_ids'], encoded_data['attention_mask'], torch.tensor(labels))

# Instantiate the model and set hyperparameters
model = GPT2Discriminator(gpt_model_name='gpt2-medium', hidden_size=768, max_seq_len=512)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
epochs = 3
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training loop
for epoch in range(epochs):
    for input_ids, attention_mask, label in dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")


