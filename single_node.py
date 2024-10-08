import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, mutual_info_score
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SingleNodeNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleNodeNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

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
    return accuracy_score(targets, predictions)

one_hot_data = torch.tensor([
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
],dtype=torch.float32)

roman_numerals = torch.tensor([
    [1, 0, 0, 0],  # I
    [2, 0, 0, 0],  # II
    [3, 0, 0, 0],  # III
    [-1, 1, 0, 0], # IV
    [0, 1, 0, 0],  # V
    [1, 1, 0, 0],  # VI
    [2, 1, 0, 0],  # VII
    [3, 1, 0, 0],  # VIII
    [-1, 0, 1, 0], # IX
    [0, 0, 1, 0]   # X
], dtype=torch.float32)

integer_inputs = torch.tensor(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=torch.float32
)

next_values = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long)

train_dataset_one_hot = TensorDataset(one_hot_data, next_values)
train_loader_one_hot = DataLoader(train_dataset_one_hot, batch_size=10, shuffle=True)

train_dataset_roman = TensorDataset(roman_numerals, next_values)
train_loader_roman = DataLoader(train_dataset_roman, batch_size=10, shuffle=True)

train_dataset_integer = TensorDataset(integer_inputs, next_values)
train_loader_integer = DataLoader(train_dataset_integer, batch_size=10, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 0.1
PATIENCE = 10
OUTPUT_SIZE = next_values.max().item() + 1

def train_model(model, train_loader, input_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
            if counter >= PATIENCE:
                print(f"Early stopping at epoch {num_epochs}")
                break
    
    return num_epochs

MODEL_ONE_HOT = SingleNodeNet(one_hot_data.shape[1], OUTPUT_SIZE).to(device)
MODEL_ROMAN = SingleNodeNet(roman_numerals.shape[1], OUTPUT_SIZE).to(device)
MODEL_INTEGER = SingleNodeNet(integer_inputs.shape[1], OUTPUT_SIZE).to(device)

print("Training One-Hot Encoded Model:")
NUM_EPOCHS_ONE_HOT = train_model(MODEL_ONE_HOT, train_loader_one_hot, one_hot_data.shape[1])
print("\nTraining Roman Numeral Model:")
NUM_EPOCHS_ROMAN = train_model(MODEL_ROMAN, train_loader_roman, roman_numerals.shape[1])
print("\nTraining Integer Input Model:")
NUM_EPOCHS_INTEGER = train_model(MODEL_INTEGER, train_loader_integer, integer_inputs.shape[1])

accuracy_one_hot = evaluate(MODEL_ONE_HOT, train_loader_one_hot, device)
accuracy_roman = evaluate(MODEL_ROMAN, train_loader_roman, device)
accuracy_integer = evaluate(MODEL_INTEGER, train_loader_integer, device)

print("\nModel Performances:")
print(f"One-Hot Encoded - Accuracy: {accuracy_one_hot:.4f}, Total Epochs: {NUM_EPOCHS_ONE_HOT}")
print(f"Roman Numeral - Accuracy: {accuracy_roman:.4f}, Total Epochs: {NUM_EPOCHS_ROMAN}")
print(f"Integer Input - Accuracy: {accuracy_integer:.4f}, Total Epochs: {NUM_EPOCHS_INTEGER}")

print("\nLearned Weights:")
print("One-Hot Encoded Model:")
print(MODEL_ONE_HOT.fc.weight)
print("\nRoman Numeral Model:")
print(MODEL_ROMAN.fc.weight)
print("\nInteger Input Model:")
print(MODEL_INTEGER.fc.weight)

def calculate_entropy(model):
    weights = model.fc.weight.detach().cpu().numpy()
    probs = np.exp(weights) / np.sum(np.exp(weights))
    return entropy(probs.flatten())

one_hot_entropy = calculate_entropy(MODEL_ONE_HOT)
roman_entropy = calculate_entropy(MODEL_ROMAN)
integer_entropy = calculate_entropy(MODEL_INTEGER)

print("\nEntropy Analysis:")
print(f"One-Hot Encoded Model Entropy: {one_hot_entropy:.4f}")
print(f"Roman Numeral Model Entropy: {roman_entropy:.4f}")
print(f"Integer Input Model Entropy: {integer_entropy:.4f}")

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

one_hot_similarity = perform_rsa(MODEL_ONE_HOT, one_hot_data)
roman_similarity = perform_rsa(MODEL_ROMAN, roman_numerals)
integer_similarity = perform_rsa(MODEL_INTEGER, integer_inputs)

def calculate_kl_divergence(model1, model2):
    weights1 = model1.fc.weight.detach().cpu().numpy().flatten()
    weights2 = model2.fc.weight.detach().cpu().numpy().flatten()

    # Ensure weights have the same length
    min_length = min(len(weights1), len(weights2))
    weights1 = weights1[:min_length]
    weights2 = weights2[:min_length]

    # Convert to probabilities
    def to_probs(w):
        exp_w = np.exp(w - np.max(w))  # Subtract max for numerical stability
        return exp_w / np.sum(exp_w)

    p = to_probs(weights1)
    q = to_probs(weights2)

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    # Calculate KL divergence
    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    
    # Return the average of both directions (symmetric KL divergence)
    return (kl_pq + kl_qp) / 2

print("\nKL Divergence Analysis:")
print(f"One-Hot vs Roman: {calculate_kl_divergence(MODEL_ONE_HOT, MODEL_ROMAN):.4f}")
print(f"One-Hot vs Integer: {calculate_kl_divergence(MODEL_ONE_HOT, MODEL_INTEGER):.4f}")
print(f"Roman vs Integer: {calculate_kl_divergence(MODEL_ROMAN, MODEL_INTEGER):.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(one_hot_similarity, cmap='coolwarm', square=True, annot=True, fmt='.2f', cbar_kws={'shrink': 0.7})
plt.title('One-Hot Encoded Representation Similarity')
plt.xlabel('Samples')
plt.ylabel('Samples')

plt.subplot(1, 3, 2)
sns.heatmap(roman_similarity, cmap='coolwarm', square=True, annot=True, fmt='.2f', cbar_kws={'shrink': 0.7})
plt.title('Roman Numeral Representation Similarity')
plt.xlabel('Samples')
plt.ylabel('Samples')

plt.subplot(1, 3, 3)
sns.heatmap(integer_similarity, cmap='coolwarm', square=True, annot=True, fmt='.2f', cbar_kws={'shrink': 0.7})
plt.title('Integer Input Representation Similarity')
plt.xlabel('Samples')
plt.ylabel('Samples')

plt.tight_layout()
plt.show()

# Generalization Testing Section

print("\n--- Generalization Testing ---")

# Test data for each representation
one_hot_11 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
roman_11 = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)  # XI in Roman numeral representation
integer_11 = torch.tensor([[11]], dtype=torch.float32)

def test_model(model, input_tensor, representation_name):
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class].item()
    
    print(f"\n{representation_name} Model:")
    print(f"Predicted next value: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("Probabilities for each class:")
    for i, prob in enumerate(probabilities[0]):
        print(f"{i+1}: {prob.item():.4f}")

# Test each model
test_model(MODEL_ONE_HOT, one_hot_11, "One-Hot Encoded")
test_model(MODEL_ROMAN, roman_11, "Roman Numeral")
test_model(MODEL_INTEGER, integer_11, "Integer Input")

def calculate_mutual_information(model, dataloader):
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
        # Convert to Bits to see how close we are to the maximum Shannon entropy
        mi_to_bits = mi / np.log(2)
        mi_scores.append(mi_to_bits)
    
    return np.mean(mi_scores)

def calculate_l2_norm(model):
    return torch.norm(model.fc.weight, p=2).item()

# Calculate and print mutual information
mi_one_hot = calculate_mutual_information(MODEL_ONE_HOT, train_loader_one_hot)
mi_roman = calculate_mutual_information(MODEL_ROMAN, train_loader_roman)
mi_integer = calculate_mutual_information(MODEL_INTEGER, train_loader_integer)

print("\nMutual Information Analysis (in Bits):")
print(f"One-Hot Encoded Model MI: {mi_one_hot:.4f}")
print(f"Roman Numeral Model MI: {mi_roman:.4f}")
print(f"Integer Input Model MI: {mi_integer:.4f}")

# Calculate and print L2 norm of weights
l2_one_hot = calculate_l2_norm(MODEL_ONE_HOT)
l2_roman = calculate_l2_norm(MODEL_ROMAN)
l2_integer = calculate_l2_norm(MODEL_INTEGER)

print("\nL2 Norm of Weights:")
print(f"One-Hot Encoded Model L2 Norm: {l2_one_hot:.4f}")
print(f"Roman Numeral Model L2 Norm: {l2_roman:.4f}")
print(f"Integer Input Model L2 Norm: {l2_integer:.4f}")

# Visualization of Mutual Information and L2 Norm
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
models = ['One-Hot', 'Roman', 'Integer']
mi_scores = [mi_one_hot, mi_roman, mi_integer]
plt.bar(models, mi_scores)
plt.title('Mutual Information by Model')
plt.ylabel('Mutual Information')

plt.subplot(1, 2, 2)
l2_norms = [l2_one_hot, l2_roman, l2_integer]
plt.bar(models, l2_norms)
plt.title('L2 Norm of Weights by Model')
plt.ylabel('L2 Norm')

plt.tight_layout()
plt.show()
