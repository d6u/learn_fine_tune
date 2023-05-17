from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Consider a sentence: "Hello, my name is ChatGPT"
sentence = "Hello, my name is ChatGPT"

# Tokenize the sentence
tokens = tokenizer.encode(sentence, return_tensors='pt')

# Create inputs and labels
inputs = tokens[:,:-1]
labels = tokens[:,1:]

# Number of training epochs
epochs = 5

# Learning rate
learning_rate = 1e-5

# Device to train on (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model.to(device)

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Total number of training steps
total_steps = len(inputs) * epochs

# Set up the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

model.train()

for epoch in range(epochs):
    for i in range(len(inputs)):
        # Clear out the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs[i].to(device), labels=labels[i].to(device))
        loss = outputs[0]

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Epoch {epoch + 1} completed")

model.save_pretrained("my_fine_tuned_model")
