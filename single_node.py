import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, mutual_info_score
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------
# Model Definitions
# ----------------------------

class SingleNodeNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleNodeNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.fc(x)

class DigitTokenizedNet(nn.Module):
    def __init__(self, num_digits, embedding_dim, hidden_dim, output_size, dropout_p=0.1):
        super(DigitTokenizedNet, self).__init__()
        self.num_digits = num_digits
        # Embedding for digits (0-9)
        self.digit_embedding = nn.Embedding(10, embedding_dim)
        # Learnable positional embeddings (for positions 0 to num_digits-1)
        self.position_embedding = nn.Embedding(num_digits, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        # Combine the embeddings by flattening and feed to an MLP
        self.fc1 = nn.Linear(num_digits * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):
        # x: shape (batch, num_digits) and dtype torch.long
        digit_emb = self.digit_embedding(x)  # (batch, num_digits, embedding_dim)
        # Create a tensor for position indices: [0, 1, ..., num_digits-1]
        batch_size = x.size(0)
        positions = torch.arange(0, self.num_digits, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # (batch, num_digits, embedding_dim)
        # Add digit and positional embeddings
        emb = digit_emb + pos_emb
        emb = self.dropout(emb)
        emb = emb.view(x.size(0), -1)  # flatten to (batch, num_digits * embedding_dim)
        out = self.fc1(emb)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ----------------------------
# Training and Evaluation Functions
# ----------------------------

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return accuracy_score(targets, predictions)

def train_model(model, train_loader, device, learning_rate=0.1, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    counter = 0
    num_epochs = 0
    while True:
        epoch_loss = train(model, train_loader, criterion, optimizer, device)
        num_epochs += 1
        print(f"Epoch [{num_epochs}], Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {num_epochs}")
                break
    return num_epochs

# ----------------------------
# Helper for Analysis Functions
# ----------------------------

def get_final_weight(model):
    if hasattr(model, 'fc'):
        return model.fc.weight.detach().cpu().numpy()
    elif hasattr(model, 'fc2'):
        return model.fc2.weight.detach().cpu().numpy()
    else:
        raise ValueError("Model does not have a recognized final linear layer.")

def calculate_entropy(model):
    weights = get_final_weight(model)
    probs = np.exp(weights) / np.sum(np.exp(weights))
    return entropy(probs.flatten())

def calculate_l2_norm(model):
    weight = get_final_weight(model)
    return np.linalg.norm(weight, ord=2)

def calculate_kl_divergence(model1, model2):
    weights1 = get_final_weight(model1).flatten()
    weights2 = get_final_weight(model2).flatten()
    min_length = min(len(weights1), len(weights2))
    weights1 = weights1[:min_length]
    weights2 = weights2[:min_length]
    def to_probs(w):
        exp_w = np.exp(w - np.max(w))
        return exp_w / np.sum(exp_w)
    p = to_probs(weights1)
    q = to_probs(weights2)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    return (kl_pq + kl_qp) / 2

def perform_rsa(model, data, device):
    hidden_states = []
    with torch.no_grad():
        for x in data:
            # Ensure input x has a batch dimension
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            x = x.to(device)
            hidden_states.append(model(x).cpu().numpy())
    hidden_states = np.concatenate(hidden_states, axis=0)
    similarity_matrix = 1 - pdist(hidden_states, metric='cosine')
    similarity_matrix = squareform(similarity_matrix)
    return similarity_matrix

def calculate_mutual_information(model, dataloader, device):
    model.eval()
    all_inputs = []
    all_outputs = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_inputs.extend(inputs.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    all_inputs = np.array(all_inputs)
    all_outputs = np.array(all_outputs)
    mi_scores = []
    for i in range(all_inputs.shape[1]):
        mi = mutual_info_score(all_inputs[:, i], np.argmax(all_outputs, axis=1))
        mi_to_bits = mi / np.log(2)
        mi_scores.append(mi_to_bits)
    return np.mean(mi_scores)

# ----------------------------
# Data Preparation
# ----------------------------

# 1. Categorical Representation (One-Hot)
categorical_data = torch.tensor([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
], dtype=torch.float32)

# 2. Roman Numeral Representation
roman_numerals = torch.tensor([
    [1, 0, 0, 0],  
    [2, 0, 0, 0],  
    [3, 0, 0, 0],  
    [-1, 1, 0, 0], 
    [0, 1, 0, 0],  
    [1, 1, 0, 0],  
    [2, 1, 0, 0],  
    [3, 1, 0, 0],  
    [-1, 0, 1, 0], 
    [0, 0, 1, 0]   
], dtype=torch.float32)

# 3. Raw Integer Representation
raw_integer_inputs = torch.tensor(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=torch.float32
)

# 4. Digit Tokenized Representation (as sequences of digit indices)
def tokenize_integer_to_digits(n, max_digits=2):
    s = str(n).zfill(max_digits)
    return [int(char) for char in s]

digit_tokenized_list = [tokenize_integer_to_digits(i) for i in range(1, 11)]
digit_tokenized_data = torch.tensor(digit_tokenized_list, dtype=torch.long)

# Target next values for next-value prediction (range: 2 to 11)
next_values = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long)

# Create DataLoaders
BATCH_SIZE = 10
train_dataset_categorical = TensorDataset(categorical_data, next_values)
train_loader_categorical = DataLoader(train_dataset_categorical, batch_size=BATCH_SIZE, shuffle=True)

train_dataset_roman = TensorDataset(roman_numerals, next_values)
train_loader_roman = DataLoader(train_dataset_roman, batch_size=BATCH_SIZE, shuffle=True)

train_dataset_raw_integer = TensorDataset(raw_integer_inputs, next_values)
train_loader_raw_integer = DataLoader(train_dataset_raw_integer, batch_size=BATCH_SIZE, shuffle=True)

train_dataset_digit_tokenized = TensorDataset(digit_tokenized_data, next_values)
train_loader_digit_tokenized = DataLoader(train_dataset_digit_tokenized, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# Device and Hyperparameters
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.1
PATIENCE = 10
# Output size is 12 because classes 0..11 exist.
# Note: Our target next values are in {2,3,...,11}, so valid predictions should be >=2.
OUTPUT_SIZE = next_values.max().item() + 1

# ----------------------------
# Model Initialization
# ----------------------------

MODEL_CATEGORICAL = SingleNodeNet(categorical_data.shape[1], OUTPUT_SIZE).to(device)
MODEL_ROMAN = SingleNodeNet(roman_numerals.shape[1], OUTPUT_SIZE).to(device)
MODEL_RAW_INTEGER = SingleNodeNet(raw_integer_inputs.shape[1], OUTPUT_SIZE).to(device)
# Revised Digit Tokenized Model with positional encoding and dropout.
MODEL_DIGIT_TOKENIZED = DigitTokenizedNet(num_digits=2, embedding_dim=8, hidden_dim=32, output_size=OUTPUT_SIZE, dropout_p=0.1).to(device)

# ----------------------------
# Training Phase
# ----------------------------

print("Training Categorical (One-Hot) Model:")
NUM_EPOCHS_CATEGORICAL = train_model(MODEL_CATEGORICAL, train_loader_categorical, device, LEARNING_RATE, PATIENCE)

print("\nTraining Roman Numeral Model:")
NUM_EPOCHS_ROMAN = train_model(MODEL_ROMAN, train_loader_roman, device, LEARNING_RATE, PATIENCE)

print("\nTraining Raw Integer Model:")
NUM_EPOCHS_RAW_INTEGER = train_model(MODEL_RAW_INTEGER, train_loader_raw_integer, device, LEARNING_RATE, PATIENCE)

print("\nTraining Digit Tokenized Model:")
NUM_EPOCHS_DIGIT_TOKENIZED = train_model(MODEL_DIGIT_TOKENIZED, train_loader_digit_tokenized, device, LEARNING_RATE, PATIENCE)

# ----------------------------
# Evaluation Phase
# ----------------------------

accuracy_categorical = evaluate(MODEL_CATEGORICAL, train_loader_categorical, device)
accuracy_roman = evaluate(MODEL_ROMAN, train_loader_roman, device)
accuracy_raw_integer = evaluate(MODEL_RAW_INTEGER, train_loader_raw_integer, device)
accuracy_digit_tokenized = evaluate(MODEL_DIGIT_TOKENIZED, train_loader_digit_tokenized, device)

print("\nModel Performances:")
print(f"Categorical (One-Hot) - Accuracy: {accuracy_categorical:.4f}, Total Epochs: {NUM_EPOCHS_CATEGORICAL}")
print(f"Roman Numeral       - Accuracy: {accuracy_roman:.4f}, Total Epochs: {NUM_EPOCHS_ROMAN}")
print(f"Raw Integer         - Accuracy: {accuracy_raw_integer:.4f}, Total Epochs: {NUM_EPOCHS_RAW_INTEGER}")
print(f"Digit Tokenized     - Accuracy: {accuracy_digit_tokenized:.4f}, Total Epochs: {NUM_EPOCHS_DIGIT_TOKENIZED}")

# ----------------------------
# Analysis Functions Usage
# ----------------------------

entropy_categorical = calculate_entropy(MODEL_CATEGORICAL)
entropy_roman = calculate_entropy(MODEL_ROMAN)
entropy_raw_integer = calculate_entropy(MODEL_RAW_INTEGER)
entropy_digit_tokenized = calculate_entropy(MODEL_DIGIT_TOKENIZED)

print("\nEntropy Analysis:")
print(f"Categorical (One-Hot) Model Entropy: {entropy_categorical:.4f}")
print(f"Roman Numeral Model Entropy: {entropy_roman:.4f}")
print(f"Raw Integer Model Entropy: {entropy_raw_integer:.4f}")
print(f"Digit Tokenized Model Entropy: {entropy_digit_tokenized:.4f}")

print("\nKL Divergence Analysis:")
print(f"Categorical vs Roman: {calculate_kl_divergence(MODEL_CATEGORICAL, MODEL_ROMAN):.4f}")
print(f"Categorical vs Raw Integer: {calculate_kl_divergence(MODEL_CATEGORICAL, MODEL_RAW_INTEGER):.4f}")
print(f"Categorical vs Digit Tokenized: {calculate_kl_divergence(MODEL_CATEGORICAL, MODEL_DIGIT_TOKENIZED):.4f}")
print(f"Roman vs Raw Integer: {calculate_kl_divergence(MODEL_ROMAN, MODEL_RAW_INTEGER):.4f}")
print(f"Roman vs Digit Tokenized: {calculate_kl_divergence(MODEL_ROMAN, MODEL_DIGIT_TOKENIZED):.4f}")
print(f"Raw Integer vs Digit Tokenized: {calculate_kl_divergence(MODEL_RAW_INTEGER, MODEL_DIGIT_TOKENIZED):.4f}")

rsa_categorical = perform_rsa(MODEL_CATEGORICAL, categorical_data, device)
rsa_roman = perform_rsa(MODEL_ROMAN, roman_numerals, device)
rsa_raw_integer = perform_rsa(MODEL_RAW_INTEGER, raw_integer_inputs, device)
rsa_digit_tokenized = perform_rsa(MODEL_DIGIT_TOKENIZED, digit_tokenized_data, device)

# ----------------------------
# Generalization Testing
# ----------------------------

# Create test inputs for unseen number 11.
categorical_11 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
roman_11 = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
raw_integer_11 = torch.tensor([[11]], dtype=torch.float32)
digit_tokenized_11 = torch.tensor([tokenize_integer_to_digits(11)], dtype=torch.long)

def test_model(model, input_tensor, representation_name, device):
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class].item()
    print(f"\n{representation_name} Model:")
    # Print raw predicted index.
    print(f"Predicted next value (class index): {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("Probabilities for each class (index: probability):")
    for i, prob in enumerate(probabilities[0]):
        print(f"{i}: {prob.item():.4f}")

test_model(MODEL_CATEGORICAL, categorical_11, "Categorical (One-Hot)", device)
test_model(MODEL_ROMAN, roman_11, "Roman Numeral", device)
test_model(MODEL_RAW_INTEGER, raw_integer_11, "Raw Integer", device)
test_model(MODEL_DIGIT_TOKENIZED, digit_tokenized_11, "Digit Tokenized", device)

mi_categorical = calculate_mutual_information(MODEL_CATEGORICAL, train_loader_categorical, device)
mi_roman = calculate_mutual_information(MODEL_ROMAN, train_loader_roman, device)
mi_raw_integer = calculate_mutual_information(MODEL_RAW_INTEGER, train_loader_raw_integer, device)
mi_digit_tokenized = calculate_mutual_information(MODEL_DIGIT_TOKENIZED, train_loader_digit_tokenized, device)

l2_categorical = calculate_l2_norm(MODEL_CATEGORICAL)
l2_roman = calculate_l2_norm(MODEL_ROMAN)
l2_raw_integer = calculate_l2_norm(MODEL_RAW_INTEGER)
l2_digit_tokenized = calculate_l2_norm(MODEL_DIGIT_TOKENIZED)

print("\nMutual Information Analysis (in Bits):")
print(f"Categorical (One-Hot) Model MI: {mi_categorical:.4f}")
print(f"Roman Numeral Model MI: {mi_roman:.4f}")
print(f"Raw Integer Model MI: {mi_raw_integer:.4f}")
print(f"Digit Tokenized Model MI: {mi_digit_tokenized:.4f}")

print("\nL2 Norm of Weights:")
print(f"Categorical (One-Hot) Model L2 Norm: {l2_categorical:.4f}")
print(f"Roman Numeral Model L2 Norm: {l2_roman:.4f}")
print(f"Raw Integer Model L2 Norm: {l2_raw_integer:.4f}")
print(f"Digit Tokenized Model L2 Norm: {l2_digit_tokenized:.4f}")

plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
sns.heatmap(rsa_categorical, cmap='coolwarm', square=True, annot=True, fmt='.2f', cbar_kws={'shrink': 0.7})
plt.title('Categorical (One-Hot) RSA')
plt.xlabel('Samples')
plt.ylabel('Samples')

plt.subplot(1, 4, 2)
sns.heatmap(rsa_roman, cmap='coolwarm', square=True, annot=True, fmt='.2f', cbar_kws={'shrink': 0.7})
plt.title('Roman Numeral RSA')
plt.xlabel('Samples')
plt.ylabel('Samples')

plt.subplot(1, 4, 3)
sns.heatmap(rsa_raw_integer, cmap='coolwarm', square=True, annot=True, fmt='.2f', cbar_kws={'shrink': 0.7})
plt.title('Raw Integer RSA')
plt.xlabel('Samples')
plt.ylabel('Samples')

plt.subplot(1, 4, 4)
sns.heatmap(rsa_digit_tokenized, cmap='coolwarm', square=True, annot=True, fmt='.2f', cbar_kws={'shrink': 0.7})
plt.title('Digit Tokenized RSA')
plt.xlabel('Samples')
plt.ylabel('Samples')

plt.tight_layout()
plt.show()
