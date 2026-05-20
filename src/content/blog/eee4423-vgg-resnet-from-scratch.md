---
title: "Implementing VGGNet and ResNet From Scratch for Image Recognition"
description: "A comparative reimplementation of VGG and ResNet, focused on depth, optimization, and residual learning."
pubDate: "May 18 2026"
heroImage: "/blog/eee4423/cnn_vgg_resnet/GiwonShin_Lab03_files/GiwonShin_Lab03_15_0.png"
badge: "EEE4423"
tags: ["EEE4423", "VGG", "ResNet", "Image Classification", "PyTorch"]
---

## Paper Context

The background section for this project revolved around two papers that changed how people think about depth in convolutional networks. VGGNet established that aggressively stacking small 3x3 convolutions could outperform shallower designs, while ResNet showed that simply making networks deeper is not enough if the optimization path becomes unstable.

Residual connections were the key idea to carry into code. Once the skip path is in place, the comparison stops being just VGG versus ResNet on paper and becomes a question of how depth, optimization, and training budget show up in practice.

## Implementation Walkthrough

The sections below compare the two implementations, the shared training setup, and the accuracy trends on the reduced CIFAR-10 split.

## VGGNet with PyTorch

### Implementing VGGNet

> 1. **Dataset**
>> - Images from the first three categories in CIFAR-10. (Due to the computational constraints.) <br>
 Three categories : plane, car, bird / The number of training images : 15,000 / The number of test images : 3,000
>> - Augmented with flipping and random cropping.
>
> 2. **Network architecture**
>> - Type-D configuration in the paper (+ 3-way classifier after convolutional layers).
>> - ReLU activation.
>> - No dropout for simplicity.
>> - This implementation applies **batch-normalization** after every convolution, even though it is not used in the original paper, to make optimization more stable.
>
> 3. **Loss function**
>> - Cross-entropy loss between outputs & ground-truths. <br>
 Note that `nn.CrossEntroyLoss` takes logits before softmax as network outputs and scalar index (not one-hot vector) as ground-truths.<br>
 See https://pytorch.org/docs/stable/nn.html#crossentropyloss for details.
>
> 4. **Training**
>> - Default weight initialization for simplicity.
>> - SGD optimizer with `learning rate = 1e-2`, `momentum = 0.9`, and `weight_decay = 5e-4`.
>> - 20 epochs without learning rate scheduling.
>
> 5. **Evaluation metric**
>> - Classification accuracy (i.e., the percentage of correct predictions).
>
>

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision
import time
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
```

```python

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = dsets.CIFAR10(root='../dataset/lab03',
                            train=True,
                            transform=transform_train,
                            download=True)

test_dataset = dsets.CIFAR10(root='../dataset/lab03',
                           train=False,
                           transform=transform_test)
```
```python
# reducing the dataset
reduced_train_dataset = []
for images, labels in train_dataset:
    if labels < 3:
        reduced_train_dataset.append((images, labels))

reduced_test_dataset = []
for images, labels in test_dataset:
    if labels < 3:
        reduced_test_dataset.append((images, labels))
```

```python
print("The number of training images : ", len(reduced_train_dataset))
print("The number of test images : ", len(reduced_test_dataset))
```
```python

train_loader = torch.utils.data.DataLoader(dataset=reduced_train_dataset,
                                           batch_size=128,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=reduced_test_dataset,
                                          batch_size=100,
                                          shuffle=False)

class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```
### Visualize a few images

```python
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
```

```python
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
```

```python
train_loader_sample = torch.utils.data.DataLoader(dataset=reduced_train_dataset,
                                           batch_size=4,
                                           shuffle=True)

# Get a batch of training data
inputs, classes = next(iter(train_loader_sample))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
```

![png](/blog/eee4423/cnn_vgg_resnet/GiwonShin_Lab03_files/GiwonShin_Lab03_15_0.png)

### VGG-16 implementation

```python

'''
## Constraints

### Dataset
dataset = cifar-10 (3 x 224 x 2s24)
train = 15000, test = 3000
3 target classes plane car, bird
augment = [flipping, random cropping]

### Architecture
config = Type-D config +(3 way classifier after conv layers)
activation = ReLU
dropout = False
batch-norm = True, after every convolution

### Loss
loss = Cross-entropy
nn.CrossEntropyLoss takes logits before softmax and scalar index (not one-hot vector) as gt

### Training
weight initialization = default(?)
optimizer = SGD(learning rate = 1e-2, momentum = 0.9, weight decay = 5e-4)
epochs = 20
lr_scheduling = False

### Evaluation metric
metric = classification accuracy (percentage of correct predictions; correct/# of predictions)

'''

class CNN_builder(nn.Module): # customable CNN model class
    def __init__(self, config, input_channel=3):
        super(CNN_builder, self).__init__()
        self.channel = input_channel
        self.layers = self.build_layers(config)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x

    def build_layers(self, config):
        layers = []
        for layer_config in config:  # dictionary in list
            layer_type = layer_config.pop("type")
            if layer_type == "conv2d":
                layers.append(self.build_conv2d_layer(**layer_config))
            elif layer_type == "residual":
                layers.append(self.build_residual_layer(**layer_config))
            elif layer_type == "pool":
                layers.append(self.build_pool_layer(**layer_config))
            elif layer_type == "linear":
                layers.append(self.build_linear_layer(**layer_config))
            elif layer_type == "dropout":
                layers.append(self.build_dropout_layer(**layer_config))
            elif layer_type == "vggblock":
                layers.append(self.build_vgg_layer(**layer_config))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        return nn.Sequential(*layers)

    # Below are the build functions for layers and blocks.

    def build_conv2d_layer(self, out_channels, kernel_size=3, stride=1, padding=0, normalization='batchnorm2d', activation='relu'):
        act = self.get_activation(activation)
        if normalization == False:
            bias = True
            conv = nn.Conv2d(self.channel, out_channels,
                             kernel_size, stride, padding, bias=bias)
            self.channel = out_channels  # update number of channels
            return nn.Sequential(conv, act)
        else:
            bias = False
            conv = nn.Conv2d(self.channel, out_channels,
                             kernel_size, stride, padding, bias=bias)
            self.channel = out_channels  # update number of channels
            norm = self.get_normalization(normalization)
            return nn.Sequential(conv, norm, act)

    def build_residual_layer(self, block, num_blocks, out_channels, stride):
        layers = [block(self.channel, out_channels, stride)]
        self.channel = out_channels  # update number of channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def build_vgg_layer(self, num_conv, out_channels, kernel_size=3, stride=1, padding=1, normalization='batchnorm2d', activation='relu'):
        layers = []
        for _ in range(num_conv):
            layers.append(self.build_conv2d_layer(out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, normalization=normalization, activation=activation))
        return nn.Sequential(*layers)

    def build_pool_layer(self, pool_type, kernel_size, stride=1, padding=0):
        if pool_type == "max":
            return nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type == "avg":
            return nn.AvgPool2d(kernel_size, stride, padding)
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")

    def build_linear_layer(self, out_features, in_features=None, activation=None):
        if in_features == None:
            in_features = self.channel
        linear = nn.Linear(in_features, out_features)
        if activation:
            act = self.get_activation(activation)
            return nn.Sequential(linear, act)
        else:
            return linear

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def get_normalization(self, normalization):
        if normalization == 'batchnorm2d':
            return nn.BatchNorm2d(self.channel)
        else:
            raise ValueError(f"Unknown activation: {normalization}")

    # Can create more build functions if needed below

def VGG(vgg_type=16):
    config = []
    vggnet16_config = [
        {"type": "vggblock", "num_conv": 2, "out_channels": 64},
        {"type": "pool", "pool_type": "max", "kernel_size": 2, "stride": 2},
        {"type": "vggblock", "num_conv": 2, "out_channels": 128},
        {"type": "pool", "pool_type": "max", "kernel_size": 2, "stride": 2},
        {"type": "vggblock", "num_conv": 3, "out_channels": 256},
        {"type": "pool", "pool_type": "max", "kernel_size": 2, "stride": 2},
        {"type": "vggblock", "num_conv": 3, "out_channels": 512},
        {"type": "pool", "pool_type": "max", "kernel_size": 2, "stride": 2},
        {"type": "vggblock", "num_conv": 3, "out_channels": 512},
        {"type": "pool", "pool_type": "max", "kernel_size": 2, "stride": 2},
        {"type": "linear", "out_features": 3},
    ]
    # add other types of vggnet if needed.
    if vgg_type == 16:
        config = vggnet16_config
    else:
        raise ValueError(f"Unknown config: \n{config}")
    return CNN_builder(config)

```
```python

model = VGG()
num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)
#  USE GPU FOR MODEL  #

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
```
```python

criterion = nn.CrossEntropyLoss()

learning_rate = 1e-2
momentum = 0.9
weight_decay = 5e-4

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum, weight_decay = weight_decay)
```
### Training the VGG 16 model and print test accuracy for every epochs

```python
num_epochs = 20

best_acc = 0 # keep track of the best performance

start_time = datetime.datetime.now() # measure running time
step = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # load resource to gpu
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()  # clear gradients
        loss.backward()  # calculate gradients
        optimizer.step()  # update parameters

    # Calculate Accuracy
    correct = 0
    total = 0
    # Iterate through test dataset
    for j, (images, labels) in enumerate(test_loader):

        # load resource to gpu
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)  # Forward pass only to get logits/output
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        if torch.cuda.is_available():  # if pred & label should be in cpu for computation
            correct += (predicted.cpu() == labels.cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100 * correct.item() / total

    if best_acc < accuracy:
        torch.save(model, "./vggnet_best.pth") # save best model

    # Print log every epoch
    print(f'Epochs: {epoch}. Loss: {loss.item()}. Accuracy: {accuracy} Elapsed time: {datetime.datetime.now()-start_time}')
    torch.save(model, "./vggnet_last.pth") # save last model

```
## ResNet with PyTorch

### Implementing ResNet

> 1. **Dataset**
>> - The same dataset used for VGGNet
>
> 2. **Network architecture**
>> - 50-layer ResNet with **bottleneck blocks**. <br>
 Note. The initial convolution layer (i.e., conv1) is different from the one in the paper &<br>
 the initial max-pooling layer is removed (because the size of CIFAR-10 images is too small).
>> - ReLU activation.
>> - Strided convolution for down-sampling instead of max-pooling layer. <br>
 Note. Once down-sampled, a $1\times1$ convolution/stride 2 is applied to residual for expanding the channel of the residual.
>> - No dropout for simplicity.
>> - Batch-normalization after every convolution.
>
> 3. **Loss function**
>> - Cross-entropy loss between outputs & ground-truths. <br>
>
> 4. **Training**
>> - Default weight initialization for simplicity.
>> - SGD optimizer with `learning rate = 1e-2`, `momentum = 0.9`, and `weight_decay = 5e-4`.
>> - 15 epochs without learning rate scheduling.
>
> 5. **Evaluation metric**
>> - Classification accuracy (i.e., the percentage of correct predictions).
>
>

### Implement ResNet50 and train it with the CIFAR 10 dataset

```python

class Bottleneck(nn.Module):  # need this as external class because of the residual connection
    expansion = 4  # channel expansion (f -> 4f)

    def __init__(self, in_channels, out_channels, stride=1):  # stride = 2 for dotted line
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # relu and residual inplace addition are not consecutive -> inplaceable
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

def ResNet(resnet_type = 50):
    resnet50_config = [
        {"type": "conv2d", "out_channels": 64,
            "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "residual", "block": Bottleneck,
            "num_blocks": 3, "out_channels": 256, "stride": 1},
        {"type": "residual", "block": Bottleneck,
            "num_blocks": 4, "out_channels": 512, "stride": 2},
        {"type": "residual", "block": Bottleneck,
            "num_blocks": 6, "out_channels": 1024, "stride": 2},
        {"type": "residual", "block": Bottleneck,
            "num_blocks": 3, "out_channels": 2048, "stride": 2},
        {"type": "pool", "pool_type": "avg", "kernel_size": 4, "stride": 1},
        {"type": "linear", "out_features": 3},
    ]
    # add other resnet types if needed.
    if resnet_type == 50:
        config = resnet50_config
    else:
        raise ValueError(f"Unknown config: \n{config}")
    return CNN_builder(config)

# same code as the VGGNet part
model = ResNet()
num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)

#  USE GPU FOR MODEL  #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

```
### Print test accuracy for every epochs.

```python

# given to use CE loss
criterion = nn.CrossEntropyLoss()

# given settings
learning_rate = 1e-2
momentum = 0.9
weight_decay = 5e-4

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum, weight_decay = weight_decay)

num_epochs = 15 # given condition
best_acc = 0 # used to track best performance

start_time = datetime.datetime.now() # measure the training time
step = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # load resource to gpu
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()  # clear gradients
        loss.backward()  # calculate gradients
        optimizer.step()  # update parameters

    # Calculate Accuracy
    correct = 0
    total = 0
    # Iterate through test dataset
    for j, (images, labels) in enumerate(test_loader):

        # load resource to gpu
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)  # Forward pass only to get logits/output
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        if torch.cuda.is_available():  # if pred & label should be in cpu for computation
            correct += (predicted.cpu() == labels.cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100 * correct.item() / total

    if best_acc < accuracy:
        torch.save(model, "./resnet_best.pth") # save best model

    # Print log every epoch
    print(f'Epochs: {epoch}. Loss: {loss.item()}. Accuracy: {accuracy} Elapsed time: {datetime.datetime.now()-start_time}')

    torch.save(model, "./resnet_last.pth") # save last model

```
# Discussion

This comparison ended up being as much about training budget as architecture.

## VGGNet
Initially, the plan was to make a model class for both VGGNet and ResNet, writing layers and blocks one by one. However, after reaching the ResNet part, I noticed the nuisance of the task, especially when the depth of the model deepens or the connection of the model get complicated.

To resove this issue, I made a general CNN class called "*CNN_builder*" which initialiezes using the the model configuration given in the form of '*list of dictionaries*' where each dictionary specifies a layer or a block of the network.
The input channel size is also given for the model to auto connect the number of input channels to output channels for each layer/block.

I have made this class easily expandable for custom block or network by making a builder function for each of the "*type*" from the config. Also by having a "*VGG*" function, this could easily expand to have multiple configurations for other types of VGGNet.

When designing the convolution layers, the application of '*BatchNorm2d*' was different condition from the official implementation. The "*Batch normalization*" suggested by Ioffe *et al* (2015) is a normalization method for deep neural networks which aims to overcome '*internal covariate shift*' problem. As mentioned in the paper, batch normalization already contains the term considering mean subtraction, therefore bias term for hidden layers redundant. Knowing this, I have designed the 2D convolutional layers without biases for both VGGNet and ResNet.

For the training loop, I kept the optimizer and augmentation settings described above and added checkpointing for both the best-accuracy model and the final-epoch model.

The results for the 20 epoch training had the best and final epoch accuracy of 92.9%. The speed of convergence to this accuracy was very fast because of the small image size of the CIFAR-10 dataset and adequate depth of VGGNet-16.

## ResNet
I have made the ResNet-50 model using the expandability of the CNN_builder class. For the residual connection of the bottleneck layer, I made an external block and placed them accordingly through the config. ResNet also didn't have batch normalization for the official implementation at the moment.

When implementing ResNet-50, the main ambiguity was where to apply spatial downsampling inside the bottleneck block. The paper states that downsampling is performed with stride-2 convolution instead of max pooling, but for bottleneck variants it is not fully explicit which of the three convolutions should carry that stride. I initially followed the architecture figure used for this project, but most library implementations place the stride on the middle 3x3 convolution. That choice is more consistent with the claim that 1x1 convolutions reduce and then restore channel dimensionality, and it also makes sense from a receptive-field perspective.
I trained ResNet-50 with both variants, but with only 15 epochs the performance gap was negligible.

On this setup, ResNet-50 peaked at 86.0%, below the VGGNet result. That gap is more likely explained by the training setup than by a weakness in the residual architecture itself.

ResNet-50 is deeper in depth and has approximately 60% more parameters than VGGNet-16, which can be interpreted as having higer capacity for learning. But the classification of low-resolution images(3x32x32) was too simple task for ResNet-50 to have advantage on. Also, despite having residual connection, 15 epochs were appearently not enough for ResNet-50 to fully converge.

Even for this simple task, if the training epochs and learning rate scheduling were properly set, ResNet-50 should have outperformed VGGNet-16. Furthermore, the performance gap will widen as the difficulty of the task increases.

Both models still show why they remained so influential: VGG for its simplicity, and ResNet for making much deeper networks practical to optimize.
