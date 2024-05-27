import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
import time
from train_utils import fit_epoch, eval_epoch, train, predict
from models import NetworkA, NetworkB, NetworkC, NetworkD
from models.layers import TTLinear, TTConv


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    return accuracy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = FashionMNIST(root='./data', train=True, download=True, transform=ToTensor())

train_size = int(0.8 * len(dataset))
valid_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_A = NetworkA().to(DEVICE)
model_B = NetworkB().to(DEVICE)
model_C = NetworkC().to(DEVICE)
model_D = NetworkD().to(DEVICE)

epochs = 50
history_A = train(train_loader, valid_loader, model_A, epochs, batch_size)
initial_accuracy = evaluate_model(model_A, test_loader)
print(f'Test A: {initial_accuracy:.4f}')

history_B = train(train_loader, valid_loader, model_B, epochs, batch_size)
initial_accuracy = evaluate_model(model_B, test_loader)
print(f'Test B: {initial_accuracy:.4f}')

history_C = train(train_loader, valid_loader, model_C, epochs, batch_size)
initial_accuracy = evaluate_model(model_C, test_loader)
print(f'Test C: {initial_accuracy:.4f}')

history_D = train(train_loader, valid_loader, model_D, epochs, batch_size)
initial_accuracy = evaluate_model(model_D, test_loader)
print(f'Test D: {initial_accuracy:.4f}')

# TT-Cross
start_time_A = time.time()

weight_matrices_A = [model_A.layer1.weight.data, model_A.layer2.weight.data, model_A.layer3.weight.data]
tt_model_A_layer1 = TTLinear(mat_ranks=[1, 7, 7, 7, 1], inp_modes=[7, 4, 4, 4, 7], out_modes=[8, 8, 8, 8, 4], weight_matrix=weight_matrices_A[0])
tt_model_A_layer2 = TTLinear(mat_ranks=[1, 7, 7, 7, 1], inp_modes=[8, 8, 8, 8, 4], out_modes=[8, 8, 8, 4, 4], weight_matrix=weight_matrices_A[1])
tt_model_A_layer3 = TTLinear(mat_ranks=[1, 7, 7, 7, 1], inp_modes=[8, 8, 8, 4, 4], out_modes=[10], weight_matrix=weight_matrices_A[2])

end_time_A = time.time()

start_time_B = time.time()

weight_matrices_B = [model_B.layer1.weight.data, model_B.layer2.weight.data, model_B.layer3.weight.data, model_B.layer4.weight.data, model_B.layer5.weight.data]
tt_model_B_layer1 = TTLinear(mat_ranks=[1, 7, 7, 7, 1], inp_modes=[7, 4, 4, 4, 7], out_modes=[8, 8, 8, 8, 4], weight_matrix=weight_matrices_B[0])
tt_model_B_layer2 = TTLinear(mat_ranks=[1, 7, 7, 7, 1], inp_modes=[8, 8, 8, 8, 4], out_modes=[8, 8, 8, 4, 4], weight_matrix=weight_matrices_B[1])
tt_model_B_layer3 = TTLinear(mat_ranks=[1, 7, 7, 7, 1], inp_modes=[8, 8, 8, 4, 4], out_modes=[8, 8, 4, 4, 4], weight_matrix=weight_matrices_B[2])
tt_model_B_layer4 = TTLinear(mat_ranks=[1, 7, 7, 7, 1], inp_modes=[8, 8, 4, 4, 4], out_modes=[8, 4, 4, 4, 4], weight_matrix=weight_matrices_B[3])
tt_model_B_layer5 = TTLinear(mat_ranks=[1, 7, 7, 7, 1], inp_modes=[8, 4, 4, 4, 4], out_modes=[10], weight_matrix=weight_matrices_B[4])

end_time_B = time.time()

print("TT-Cross for model A:", end_time_A - start_time_A)
print("TT-Cross for model B:", end_time_B - start_time_B)

# Pruning 
start_time_prune_A = time.time()

prune.ln_structured(
    model_A.layer1, 'weight', amount=0.916, dim=1, n=float('-inf')
)
prune.ln_structured(
    model_A.layer2, 'weight', amount=0.916, dim=1, n=float('-inf')
)
prune.ln_structured(
    model_A.layer3, 'weight', amount=0.916, dim=1, n=float('-inf')
)

prune.remove(model_A.layer1, 'weight')
prune.remove(model_A.layer2, 'weight')
prune.remove(model_A.layer3, 'weight')

end_time_prune_A = time.time()

start_time_prune_B = time.time()

prune.ln_structured(
    model_B.layer1, 'weight', amount=0.944, dim=1, n=float('-inf')
)
prune.ln_structured(
    model_B.layer2, 'weight', amount=0.944, dim=1, n=float('-inf')
)
prune.ln_structured(
    model_B.layer3, 'weight', amount=0.944, dim=1, n=float('-inf')
)
prune.ln_structured(
    model_B.layer4, 'weight', amount=0.944, dim=1, n=float('-inf')
)
prune.ln_structured(
    model_B.layer5, 'weight', amount=0.944, dim=1, n=float('-inf')
)

prune.remove(model_B.layer1, 'weight')
prune.remove(model_B.layer2, 'weight')
prune.remove(model_B.layer3, 'weight')
prune.remove(model_B.layer4, 'weight')
prune.remove(model_B.layer5, 'weight')

end_time_prune_B = time.time()

print('Finished Pruning')
print(f'Pruning time for model A: {end_time_prune_A - start_time_prune_A:.3f} seconds')
print(f'Pruning time for model B: {end_time_prune_B - start_time_prune_B:.3f} seconds')

pruned_accuracy_A = evaluate_model(model_A, test_loader)
pruned_accuracy_B = evaluate_model(model_B, test_loader)
print(f'Accuracy after pruning for model A: {pruned_accuracy_A:.4f}')
print(f'Accuracy after pruning for model B: {pruned_accuracy_B:.4f}')
