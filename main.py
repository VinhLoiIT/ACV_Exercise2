import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict
from torchvision import datasets, transforms, models

import os
import argparse

data_dir = 'data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/val'
test_dir = data_dir + '/test'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: Build and train your network
epochs = 50
lr = 0.001
print_every = 20
batch_size = 64

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
testval_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_trainset = datasets.ImageFolder(train_dir, transform=train_transforms)
image_testset = datasets.ImageFolder(test_dir, transform=testval_transforms)
image_valset = datasets.ImageFolder(valid_dir, transform=testval_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
image_trainloader = torch.utils.data.DataLoader(image_trainset, batch_size=64, shuffle=True)
image_testloader = torch.utils.data.DataLoader(image_testset, batch_size=64, shuffle=True)
image_valloader = torch.utils.data.DataLoader(image_valset, batch_size=64, shuffle=True)

# Freeze parameters so we don't backprop through them
hidden_layers = [10240, 1024]
def make_model(hidden_layers, lr):
    model = models.densenet161(pretrained=True)
    input_size = 2208
    output_size = 3 # n_classes
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('dropout',nn.Dropout(0.5)),
                              ('fc1', nn.Linear(input_size, hidden_layers[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(hidden_layers[1], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    return model

def cal_accuracy(model, optimizer, criterion, dataloader):
    validation_loss = 0
    accuracy = 0
    for i, (inputs,labels) in enumerate(dataloader):
      optimizer.zero_grad()
      inputs, labels = inputs.to(device) , labels.to(device)
      model.to(device)
      with torch.no_grad():    
          outputs = model.forward(inputs)
          validation_loss = criterion(outputs,labels)
          ps = torch.exp(outputs).data
          equality = (labels.data == ps.max(1)[1])
          accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
    validation_loss = validation_loss / len(dataloader)
    accuracy = accuracy /len(dataloader)
    
    return validation_loss, accuracy

def train(model, image_trainloader, image_valloader, epochs, print_every, save_dir):
    print_every = print_every
    steps = 0

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    # change to cuda
    model.to(device)
    model.train()

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(image_trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                val_loss, train_ac = cal_accuracy(model, optimizer, criterion, image_valloader)
                print("Epoch: {}/{}... | ".format(e+1, epochs),
                      "Loss: {:.4f} | ".format(running_loss/print_every),
                      "Validation Loss {:.4f} | ".format(val_loss),
                      "Accuracy {:.4f}".format(train_ac))
                
                model.train()
                running_loss = 0
        save_model(e, model, save_dir)

def testing(model, dataloader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in image_testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _ , prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels.data).sum().item()
        print('Accuracy on the test set: %d %%' % (100 * correct / total))

def save_model(epoch, model, save_dir='./'):
  state = {
    'structure' :'densenet161',
    'learning_rate': lr,
    'epochs': epochs,
    'hidden_layers':hidden_layers,
    'state_dict':model.state_dict(),
  }
  torch.save(state, f'{save_dir}/checkpoint_{epoch}.pth')

def loading_checkpoint(path):
    
    # Loading the parameters
    state = torch.load(path, map_location=torch.device(device))
    lr = state['learning_rate']
    structure = state['structure']
    hidden_layers = state['hidden_layers']
    epochs = state['epochs']
    
    # Building the model from checkpoints
    model = make_model(hidden_layers, lr)
    model.load_state_dict(state['state_dict'])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'], type=str)
    parser.add_argument('--checkpoint', default='')
    
    args = parser.parse_args()

    if args.mode == 'train':
        checkpoints_dir = os.path.join(data_dir, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        model = make_model(hidden_layers, lr)
        train(model, image_trainloader, image_valloader, epochs, print_every, checkpoints_dir)
    elif args.mode == 'test':
        model = loading_checkpoint(args.checkpoint)
        testing(model, image_testloader)

# review_loader = torch.utils.data.DataLoader(image_testset, batch_size=10, shuffle=True)
# n_rows = 2
# n_cols = 5
# fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, sharey='all', sharex='all', figsize=(15,8))
# samples = next(iter(review_loader))
  
# images, gt_labels = samples
# predict_labels = testing_images(model, images).to('cpu').numpy()

# images = [image.to('cpu').permute(1, 2, 0).numpy() for image in images]
# gt_labels = [image_testloader.dataset.classes[label] for label in gt_labels]
# predict_labels = [image_testloader.dataset.classes[label] for label in predict_labels]

# for i, (image, predict_label, gt_label) in enumerate(zip(images, predict_labels, gt_labels)):
#   title = f'{predict_label} / {gt_label}'
#   axes[i // n_cols, i % n_cols].set_title(title)
#   axes[i // n_cols, i % n_cols].imshow(image)

# fig.savefig('./results.png', dpi=300)

