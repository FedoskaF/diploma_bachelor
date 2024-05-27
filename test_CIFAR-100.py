import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from tqdm import tqdm
import time
from train_utils import fit_epoch, eval_epoch, train, predict
from models import vgg16, VGG16_TTConv, VGG16_TT_All
from models.layers import TTLinear


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            outputs = model(inputs.to(DEVICE))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    return accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CIFAR100(root='./data', train=True, download=True, transform=ToTensor())

train_size = int(0.6 * len(dataset))
valid_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_vgg16 = VGG16().to(DEVICE)
model_vgg16_ttconv = VGG16_TTConv().to(DEVICE)
model_vgg16_tt_all = VGG16_TT_All().to(DEVICE)

epochs = 15
history_vgg16 = train(train_loader, valid_loader, model_vgg16, epochs, batch_size)
initial_accuracy = evaluate_model(model_vgg16, test_loader)
print(f'Test VGG16: {initial_accuracy:.4f}')

history_vgg16_ttconv = train(train_loader, valid_loader, model_vgg16_ttconv, epochs, batch_size)
initial_accuracy = evaluate_model(model_vgg16_ttconv, test_loader)
print(f'Test VGG16_TTConv: {initial_accuracy:.4f}')

history_vgg16_tt_all = train(train_loader, valid_loader, model_vgg16_tt_all, epochs, batch_size)
initial_accuracy = evaluate_model(model_vgg16_tt_all, test_loader)
print(f'Test VGG16_TT_All: {initial_accuracy:.4f}')

# Pruning
start_time_prune_vgg16 = time.time()

for module in model_vgg16.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.ln_structured(module, 'weight', amount=0.968, dim=1, n=float('-inf'))
        prune.remove(module, 'weight')

end_time_prune_vgg16 = time.time()
print(f'Pruning time for VGG16: {end_time_prune_vgg16 - start_time_prune_vgg16:.3f} seconds')

pruned_accuracy_vgg16 = evaluate_model(model_vgg16, test_loader)
print(f'Accuracy after pruning for VGG16: {pruned_accuracy_vgg16:.4f}')
