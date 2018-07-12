import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

#1. Basic autograd example1
#Create Tensors.
x = torch.tensor(1, requires_grad = True)
w = torch.tensor(2, requires_grad = True)
b = torch.tensor(3, requires_grad = True)

#Computational graph.
y = w * x + b

#Compute gradients.
y.backward()

#Print out gradients.
print(x.grad) #tensor(2)
print(w.grad) #tensor(1)
print(b.grad) #tensor(1)

#2. Basic autograd exmaple 2
#Create Tensors.
x = torch.randn(10, 3)
y = torch.randn(10, 2)

#Build a fully connected layer.
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

#Build loss function and optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)

#Compute loss
loss = criterion(pred, y)
print('loss: ', loss.item())

#Backward
loss.backward()

print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

optimizer.step()
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())

#3. Loading data from numpy
#Create a numpy array.
x = np.array([[1, 2], [3, 4]])

#Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)

#Convert the torch tensor to a numpy array
z = y.numpy()

#4. Input pipline
train_dataset = torchvision.datasets.CIFAR10(root = '../../data/',
                                             train = True,
                                             transform = transforms.ToTensor(),
                                             download = True)

#Fetch one data pair.
image, label = train_dataset[0]
print(image.size())
print(label)

#Data loader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = 64,
                                           shuffle = True)

data_iter = iter(train_loader)
images, labels = data_iter.next()

for images. labels in train_loader:
    pass


#5. Input pipline for custom dataset

#6. Pretrained model
#ResNet-18
resnet = torchvision.models.resnet18(pretrained = True)

for param in resnet.parameters():
    param.required_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 100)

images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size()) #(64, 100)

#7. Save and load the model
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

#권장 : parameters만 저장
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))