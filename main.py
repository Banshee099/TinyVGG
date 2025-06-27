import torch
from train import train_step
from test import test_step
from tiny_VGG import FashionMNISTModelV2
from torchvision import datasets
import torchvision
from Tiny_VGG.eval import evaluate
from torch.utils.data import DataLoader
from torch import nn
from Tiny_VGG.accuracy_fn import accuracy_fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data = datasets.FashionMNIST(
    root='../data',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root='../data',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)


BATCH_SIZE=32
train_data_loader = DataLoader(dataset=train_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True)
test_data_loader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)
class_name = train_data.classes
model = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_name)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 3
for epoch in range(epochs):
    print("Epoch: {}".format(epoch))
    print("In training Step")
    train_step(model,
               train_data_loader,
               loss_fn,
               optimizer,
               accuracy_fn,
               device)
    print("In testing Step")
    test_step(model,
              test_data_loader,
              loss_fn,
              accuracy_fn,
              device)
print("Finished Training")
torch.save(model.state_dict(), "model.pth")
print("")
print("Model Evaluation")
print(evaluate(model, test_data_loader, loss_fn, accuracy_fn,device))