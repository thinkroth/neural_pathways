import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the single-node neural network
class SingleNodeNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleNodeNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

# Define the training function
def train(model, dataloader, train_criterion, optimizer, train_device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(train_device), labels.to(train_device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = train_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Define the evaluation function
def evaluate(model, dataloader, train_device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(train_device), labels.to(train_device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    accuracy = accuracy_score(targets, predictions)
    return accuracy

# Prepare the data (integers)
integers = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=torch.float32)

next_integers = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long)

# Create DataLoader for integers
train_dataset_integers = TensorDataset(integers, next_integers)
train_loader_integers = DataLoader(train_dataset_integers, batch_size=10, shuffle=True)

# Prepare the data (Roman numerals)
roman_numerals = torch.tensor([
    [0, 0, 0, 0],  # 0
    [1, 0, 0, 0],  # I
    [2, 0, 0, 0],  # II
    [3, 0, 0, 0],  # III
    [-1, 1, 0, 0],  # IV
    [0, 1, 0, 0],  # V
    [1, 1, 0, 0],  # VI
    [2, 1, 0, 0],  # VII
    [3, 1, 0, 0],  # VIII
    [-1, 0, 1, 0]   # IX
], dtype=torch.float32)

next_roman_numerals = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long)

# Create DataLoader for Roman numerals
train_dataset_roman = TensorDataset(roman_numerals, next_roman_numerals)
train_loader_roman = DataLoader(train_dataset_roman, batch_size=10, shuffle=True)

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model (integers)
INPUT_SIZE_INTEGERS = 10  # One-hot encoded vectors have 10 dimensions
OUTPUT_SIZE_INTEGERS = 11  # Matches the number of classes (1-10)
LEARNING_RATE = 0.1
PATIENCE = 10

MODEL_INTEGERS = SingleNodeNet(INPUT_SIZE_INTEGERS, OUTPUT_SIZE_INTEGERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_integers = optim.Adam(MODEL_INTEGERS.parameters(), lr=LEARNING_RATE)

best_loss = float('inf')
COUNTER = 0
NUM_EPOCHS_INTEGERS = 0

while True:
    epoch_loss = train(MODEL_INTEGERS, train_loader_integers, criterion, optimizer_integers, device)
    NUM_EPOCHS_INTEGERS += 1
    print(f"Epoch [{NUM_EPOCHS_INTEGERS}], Loss: {epoch_loss:.4f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        COUNTER = 0
    else:
        COUNTER += 1
        if COUNTER >= PATIENCE:
            print(f"Early stopping at epoch {NUM_EPOCHS_INTEGERS}")
            break

# Train the model (Roman numerals)
INPUT_SIZE_ROMAN = 4
OUTPUT_SIZE_ROMAN = 11

MODEL_ROMAN = SingleNodeNet(INPUT_SIZE_ROMAN, OUTPUT_SIZE_ROMAN).to(device)
optimizer_roman = optim.Adam(MODEL_ROMAN.parameters(), lr=LEARNING_RATE)

best_loss = float('inf')
COUNTER = 0
NUM_EPOCHS_ROMAN = 0

while True:
    epoch_loss = train(MODEL_ROMAN, train_loader_roman, criterion, optimizer_roman, device)
    NUM_EPOCHS_ROMAN += 1
    print(f"Epoch [{NUM_EPOCHS_ROMAN}], Loss: {epoch_loss:.4f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        COUNTER = 0
    else:
        COUNTER += 1
        if COUNTER >= PATIENCE:
            print(f"Early stopping at epoch {NUM_EPOCHS_ROMAN}")
            break

# Evaluate the models on the training sets
accuracy_integers = evaluate(MODEL_INTEGERS, train_loader_integers, device)
accuracy_roman = evaluate(MODEL_ROMAN, train_loader_roman, device)

print("Integer Model Performance:")
print(f"Accuracy: {accuracy_integers:.4f}")
print(f"Total Epochs: {NUM_EPOCHS_INTEGERS}")

print("Roman Numeral Model Performance:")
print(f"Accuracy: {accuracy_roman:.4f}")
print(f"Total Epochs: {NUM_EPOCHS_ROMAN}")

# Analyze the learned representations
print("\nInteger Model Weights:")
print(MODEL_INTEGERS.fc.weight)
print("\nRoman Numeral Model Weights:")
print(MODEL_ROMAN.fc.weight)

# Entropy Analysis
def calculate_entropy(model):
    weights = model.fc.weight.detach().cpu().numpy()
    probs = np.exp(weights) / np.sum(np.exp(weights))
    return entropy(probs.flatten())

integer_entropy = calculate_entropy(MODEL_INTEGERS)
roman_entropy = calculate_entropy(MODEL_ROMAN)

print(f"\nInteger Model Entropy: {integer_entropy:.4f}")
print(f"Roman Numeral Model Entropy: {roman_entropy:.4f}")

# Representation Similarity Analysis (RSA)
def perform_rsa(model, data):
    hidden_states = []
    with torch.no_grad():
        for x in data:
            x = x.to(device)
            hidden_states.append(model.fc(x).cpu().numpy())
    hidden_states = np.array(hidden_states)
    similarity_matrix = 1 - pdist(hidden_states, metric='cosine')
    similarity_matrix = squareform(similarity_matrix)
    return similarity_matrix

integer_similarity = perform_rsa(MODEL_INTEGERS, integers)
roman_similarity = perform_rsa(MODEL_ROMAN, roman_numerals)

# KL Divergence Analysis
def calculate_kl_divergence(model1, model2):
    weights1 = model1.fc.weight.detach().cpu().numpy()
    probs1 = np.exp(weights1) / np.sum(np.exp(weights1), axis=1, keepdims=True)

    weights2 = model2.fc.weight.detach().cpu().numpy()
    probs2 = np.exp(weights2) / np.sum(np.exp(weights2), axis=1, keepdims=True)

    kl_div = 0.0
    for p1 in probs1:
        p2 = probs2[:, :p1.shape[0]]
        p2 = p2.reshape(-1)[:p1.shape[0]]
        kl_div += np.sum(p1 * np.log(p1 / p2))

    kl_div /= probs1.shape[0]
    return kl_div

kl_divergence = calculate_kl_divergence(MODEL_INTEGERS, MODEL_ROMAN)
print(f"\nKL Divergence between Integer and Roman Numeral Models: {kl_divergence:.4f}")

# Visualization of Learned Representations
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(
    integer_similarity,
    cmap='coolwarm',
    square=True,
    annot=True,
    fmt='.2f',
    cbar_kws={'shrink': 0.7}
)

plt.title('Integer Representation Similarity')
plt.xlabel('Samples')
plt.ylabel('Samples')

plt.subplot(1, 2, 2)
sns.heatmap(
    roman_similarity,
    cmap='coolwarm',
    square=True,
    annot=True,
    fmt='.2f',
    cbar_kws={'shrink': 0.7}
)
plt.title('Roman Numeral Representation Similarity')
plt.xlabel('Samples')
plt.ylabel('Samples')

plt.tight_layout()
plt.show()
