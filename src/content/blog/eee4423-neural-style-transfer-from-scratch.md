---
title: "Building Neural Style Transfer From Scratch"
description: "A code-first walkthrough of neural style transfer using VGG feature losses, Gram matrices, and iterative image optimization."
pubDate: "May 18 2026"
heroImage: "/blog/eee4423/styletransfernetwork/GIwonShin_Lab10_files/GIwonShin_Lab10_48_0.png"
badge: "EEE4423"
tags: ["EEE4423", "Style Transfer", "VGG", "Perceptual Loss", "PyTorch"]
---

## Paper Context

The background reading focused on the central insight behind neural style transfer: CNN representations separate semantic content and local texture strongly enough that style can be modeled through feature correlations rather than through direct pixel copying. That is why style transfer is usually explained through content losses on deeper features and style losses on Gram matrices from shallower layers.

That separation between content and texture is the part that shaped the implementation. Once VGG is treated as a fixed loss network, the experiment becomes less about a black-box stylization trick and more about how the optimization balances scene structure against texture statistics.

## Implementation Walkthrough

The sections below cover the loss network, the optimization loop, and the stylized images produced during the run.

# Neural Stylization with Pytorch
## Introduction


This implementation follows A Neural Algorithm of Artistic Style, which separates and recombines the content and style of natural images. The algorithm takes a content image and a style image, then optimizes a new image that preserves the content of the former while borrowing the artistic texture of the latter.

**Main idea**

To transfer the style image $\textit{I}_s$ onto a content image $\textit{I}_c$ we synthesize a new image that simultaneously matches the content representation of $\textit{I}_c$ and the style representation of $\textit{I}_s$ (Fig). Thus we jointly minimize the distance of the feature representations of a white noise image from the content representation of the photograph in one layer and the style representation of the painting defined on a number of layers of the Convolutional Neural Network.

```python
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
import numpy as np

from torchvision import datasets
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
```

**Style and content image**

```python
style_img_name = '../dataset/lab10/style/mosaic.jpg'
content_img_name = '../dataset/lab10/content/Tuebingen_Neckarfront.jpg'
```

```python
style_img = Image.open(style_img_name)
content_img = Image.open(content_img_name)
```

```python
plt.figure(figsize = (15,15))
plt.subplot(1,2,1)
plt.title('style image')
plt.imshow(style_img)
plt.subplot(1,2,2)
plt.title('content image')
plt.imshow(content_img)
```

 <matplotlib.image.AxesImage at 0x7f6cd1d60d10>

![png](/blog/eee4423/styletransfernetwork/GIwonShin_Lab10_files/GIwonShin_Lab10_12_1.png)

### Style representation

To obtain a representation of the style of an input image, we use a feature space designed to capture texture information. This feature space can be built on top of the filter responses in any layer of the network. It consists of the correlations between the different filter responses, where the expectation is taken over the spatial extent of the feature maps. These feature correlations are given by the Gram matrix $\mathit{G} \in \mathcal{R}^{\mathit{N_l} \times \mathit{N_l}}$, where $\mathit{G^l_{ij}}$ is the inner product between the vectorised feature maps $i$ and $j$ in layer $l$:

### <center> ${\mathit{G^l_{ij}} = \sum_{k}{F^l_{ik}F^l_{jk}}}$ </center>

```python
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
```

### Deep image representations

In this work we show how the generic feature representations learned by high-performing Convolutional Neural Networks can be used to independently process and manipulate the content and the style of natural images.

The image reperesentations were generated on the basis of the VGG network, which was trained to perform object recognition and localization. We use the feature space provided by the 16 convolutional and 5 pooling layers of the 19-layer VGG network.

```python
import torchvision.models.vgg as vgg

class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        # get vgg network
        self.vgg = vgg.vgg19(pretrained=False)

    def forward(self, x, layer_name):
        output = {}
        for name, module in self.vgg.features._modules.items():
            x = module(x)
            if name in layer_name:
                output[layer_name[name]] = x
        return output
```

```python
loss_net = LossNetwork().cuda()
checkpoint = torch.load('../pretrain/lab10/vgg19.pth')
loss_net.vgg.load_state_dict(checkpoint)
for param in loss_net.parameters():
    param.requires_grad = False
```

Content representation is on layer ‘conv4 2’ and the style representation is on layers ‘conv1 1’, ‘conv2 1’, ‘conv3 1’, ‘conv4 1’ and ‘conv5 1’

```python
style_layer_name = {
    '1': "conv1-1",
    '6': "conv2-1",
    '11': "conv3-1",
    '20': "conv4-1",
    '29': "conv5-1"
}

content_layer_name = {
    '22': "conv4-2"
}
```

**Pre and post processing for images**

```python

img_size = 512
prep = transforms.Compose([transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                        std=[1,1,1]),
])

post = transforms.Compose([
    transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                        std=[1,1,1]),
    transforms.Lambda(lambda x: torch.clamp(x,0,1)),
    transforms.ToPILImage()
])
```

```python
content = prep(content_img).cuda().unsqueeze(0)
style = prep(style_img).cuda().unsqueeze(0)

out_img = content.clone()
out_img.requires_grad = True

criterion = nn.MSELoss()
style_gt = [gram_matrix(f).detach() for f in loss_net(style, style_layer_name).values()]
content_gt = [A.detach() for A in loss_net(content, content_layer_name).values()]

show_iter = 50
optimizer = optim.LBFGS([out_img])
n_iter = [0]
```

**Hyper parameters**

```python
alpha = 1e0
beta = 1e7
max_iter = 500
```

**Optimizing image**

```python
while n_iter[0] <= max_iter:

    def closure():
        optimizer.zero_grad()

        style_layers = [gram_matrix(f) for f in loss_net(out_img, style_layer_name).values()]
        content_layers = [f for f in loss_net(out_img, content_layer_name).values()]

        style_loss = 0
        for i in range(len(style_layers)):
            style_loss += criterion(style_layers[i], style_gt[i])

        content_loss = criterion(content_layers[0], content_gt[0])

        loss = alpha * content_loss + beta * style_loss
        loss.backward()
        n_iter[0]+=1
        #print loss
        if n_iter[0]%show_iter == (show_iter-1):
            print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))

        return loss

    optimizer.step(closure)

#display result
```
**Visualize result images**

```python
out_img_hr = post(out_img.data[0].cpu().squeeze())

plt.figure(figsize = (15,15))

plt.subplot(1,3,1)
plt.title('original image')
plt.imshow(content_img)
plt.subplot(1,3,2)
plt.title('style image')
plt.imshow(style_img)
plt.subplot(1,3,3)
plt.title('transfered image')
plt.imshow(out_img_hr)
plt.show()
```

![png](/blog/eee4423/styletransfernetwork/GIwonShin_Lab10_files/GIwonShin_Lab10_28_0.png)

# Perceptual Losses for Style Transfer with Pytorch

**Main idea**

Previous method produces high-quality results, but is computationally expensive since each step of the optimization problem requires a forward and backward pass through the pretrained network. To overcome this computational burden, we train a feed-forward network to quickly approximate solutions to their optimization problem.

$\mathcal{L}_{total} = \alpha\sum_{}{\mathcal{l}_{content}} + \beta\sum_{}{\mathcal{l}_{style}} + \gamma\mathcal{l}_{TV}$

- $\mathcal{l}_{content}^{\phi,j} = ||\phi_j(\hat{y}) - \phi_j(y)||^2_2$, $\phi$ represents vgg feature

- $\mathcal{l}_{style}^{\phi,j} = ||G^{\phi}_{j}(\hat{y}) - G^{\phi}_{j}(y)||^2_F$, $G$ represents gram matrix.

- $\mathcal{l}_{TV}(y) = \sum_{i,j}{|y_{i+1,j}-y_{i,j}|+|y_{i,j+1}-y_{i,j}|}$

**Data loader**

```python
data_root = '../dataset-dllab/lab10/train'
image_size = 224
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder(data_root, transform)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```

### Image Transform Net


##### Residual block (channel, x)
- Conv 1: $3\times3$ Conv(in: channel, out: channel, padding: 1) with reflection padding
- Instance Norm
- Relu
- Conv 2: $3\times3$ Conv(in: channel, out: channel, padding: 1) with reflection padding
- Instance Norm
- Residual Connection

#### ImageTransformNet
- Conv $9\times9$ (in: 3, out: 32, padding: 4) with reflection padding
- Instance Norm
- Relu
- Conv $3\times3$ (in: 32, out: 64, padding: 1, **stride: 2**) with reflection padding
- Instance Norm
- Relu
- Conv $3\times3$ (in: 64, out:128, padding: 1, **stride: 2**) with reflection padding
- Instance Norm
- Relu
- 5 Residual block(channel: 128)
- $2\times$ Nearest Upsample
- Conv $3\times3$ (in: 128, out:64, padding: 1) with reflection padding
- Instance Norm
- Relu
- **$2\times$ Nearest Upsample**
- Conv $3\times3$ (in: 64, out:32, padding: 1) with reflection padding
- Instance Norm
- Relu
- Conv $9\times9$ (in: 32, out:3, padding: 4) with reflection padding

```python
class ImageTransformNet(torch.nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super(ResidualBlock, self).__init__()
                self.channels = channels
                self.relu = nn.ReLU()
                # Residual Block
                self.residual = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, 3, padding=1, padding_mode="reflect"),
                    nn.InstanceNorm2d(self.channels,),
                    nn.ReLU(),
                    nn.Conv2d(self.channels, self.channels, 3, padding=1, padding_mode="reflect"),
                    nn.InstanceNorm2d(self.channels,),
                )

            def forward(self, x):
                out = self.residual(x)
                out = out + x
                # out = self.relu(out) # ambiguous
                return out

        # conv block 1
        self.conv_1 = nn.Conv2d(3, 32, 9, padding=4, padding_mode="reflect")
        self.instnorm_1 = nn.InstanceNorm2d(32)
        self.relu = nn.ReLU()

        # conv block 2
        self.conv_2 = nn.Conv2d(32, 64, 3, padding=1,stride=2, padding_mode="reflect")
        self.instnorm_2 = nn.InstanceNorm2d(64)
        # relu

        # conv block 3
        self.conv_3 = nn.Conv2d(64, 128, 3, padding=1,stride=2, padding_mode="reflect")
        self.instnorm_3 = nn.InstanceNorm2d(128)
        # relu

        # residual blocks
        self.residual_1 = ResidualBlock(128)
        self.residual_2 = ResidualBlock(128)
        self.residual_3 = ResidualBlock(128)
        self.residual_4 = ResidualBlock(128)
        self.residual_5 = ResidualBlock(128)

        # upsample 1
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        # conv block 4
        self.conv_4 = nn.Conv2d(128, 64, 3, padding=1, padding_mode="reflect")
        self.instnorm_4 = nn.InstanceNorm2d(64)
        # relu

        # upsampe 2

        # conv block 5
        self.conv_5 = nn.Conv2d(64, 32, 3, padding=1, padding_mode="reflect")
        self.instnorm_5 = nn.InstanceNorm2d(32)
        # relu

        # conv block 6
        self.conv_6 = nn.Conv2d(32, 3, 9, padding=4, padding_mode="reflect")

    def forward(self, X):
        X = self.conv_1(X)
        X = self.instnorm_1(X)
        X = self.relu(X)

        X = self.conv_2(X)
        X = self.instnorm_2(X)
        X = self.relu(X)

        X = self.conv_3(X)
        X = self.instnorm_3(X)
        X = self.relu(X)

        X = self.residual_1(X)
        X = self.residual_2(X)
        X = self.residual_3(X)
        X = self.residual_4(X)
        X = self.residual_5(X)

        X = self.upsample(X)

        X = self.conv_4(X)
        X = self.instnorm_4(X)
        X = self.relu(X)

        X = self.upsample(X)

        X = self.conv_5(X)
        X = self.instnorm_5(X)
        X = self.relu(X)

        X = self.conv_6(X)
        return X

```

```python
transformer = ImageTransformNet().cuda()
```

### Find style and content representation layers in loss net

```python
style_layer_name = {
    '3': "relu1-2",
    '8': "relu2-2",
    '17': "relu3-4", # different from paper since we use VGG-19
    '26': "relu4-4", # different from paper since we use VGG-19
}

content_layer_name = {
    '8': "relu2-2",
}
```

```python
style = transform(style_img).cuda().unsqueeze(0)
style_gt = [gram_matrix(f).detach() for f in loss_net(style, style_layer_name).values()]
```

**Hyper parameters**

```python
# You may adjust hyper paramters
alpha = 1e0
beta = 1e4
gamma = 1e-5

LR = 1e-3
steps = 2000
optimizer = optim.Adam(transformer.parameters(), LR)
```

### Training the image transformNet

- Print total loss, content loss, style loss and total variation loss for every 50 iterations
- Style loss should contain all of the layers listed above
- Use alpha, beta, gamma as coefficient

```python
transformer.train()
n_iter = 0
total_running_loss = 0.0
style_running_loss = 0.0
content_running_loss = 0.0
tv_running_loss = 0.0

class Found(Exception): pass
try:
    while True:
        for x, _ in train_loader:
            x = x.cuda()
            optimizer.zero_grad()

            # Forward
            y = transformer(x)
            content_gt = [A.detach() for A in loss_net(x, content_layer_name).values()]

            style_layers = [gram_matrix(f) for f in loss_net(y, style_layer_name).values()]
            content_layers = [f for f in loss_net(y, content_layer_name).values()]

            # style loss
            for i in range(len(style_layers)):
                if i == 0:
                    style_loss = criterion(style_layers[i], style_gt[i].repeat([x.shape[0], 1, 1]))
                else:
                    style_loss += criterion(style_layers[i], style_gt[i].repeat([x.shape[0], 1, 1]))

            # content_loss
            content_loss = criterion(content_layers[0], content_gt[0])

            # total variance loss
            i_diff = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
            j_diff = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
            tv_loss = torch.sum(i_diff) + torch.sum(j_diff)

            # total loss
            total_loss = alpha * content_loss + beta * style_loss + gamma * tv_loss

            # backwards
            total_loss.backward()

            # update
            optimizer.step()
            n_iter += 1
            total_running_loss += total_loss.item()
            style_running_loss += style_loss.item()
            content_running_loss += content_loss.item()
            tv_running_loss += tv_loss.item()

            # training log
            if n_iter%show_iter == (show_iter-1):
                print(f'Iteration: {n_iter+1}, total loss: {total_running_loss/show_iter}, content loss: {alpha *content_running_loss/show_iter}, style loss: {beta *style_running_loss/show_iter}, tv loss: {gamma *tv_running_loss/show_iter}')
                total_running_loss = 0.0
                style_running_loss = 0.0
                content_running_loss = 0.0
                tv_running_loss = 0.0

            if n_iter >= steps:
                raise Found

except Found:
    pass

```
### Results and discussion
- Compare the result of neural style with yours
- Adjust the hyper parameter and analyze each result
- Use transform function before and after inference

```python
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                         std=[1/0.229, 1/0.224, 1/0.255]),
    transforms.Lambda(lambda x: torch.clamp(x,0,1)),
    transforms.ToPILImage()
])

test_img = test_transform(content_img).cuda().unsqueeze(0)
transformer.eval()
with torch.no_grad():
    test_result = transformer(test_img)
result_img = inverse_transform(test_result.cpu().squeeze())

plt.figure(figsize = (15,15))
plt.subplot(1,2,1)
plt.title('transfered image')
plt.imshow(out_img_hr)
plt.subplot(1,2,2)
plt.title('perceptual loss')
plt.imshow(result_img)
plt.show()

```

![png](/blog/eee4423/styletransfernetwork/GIwonShin_Lab10_files/GIwonShin_Lab10_48_0.png)

# Discussion

## Quantitative analysis
The ImageTransformNet followed the structure used in the notebook, including residual blocks defined inside the constructor. The main deviation from the paper was in the loss network: this version uses VGG-19 rather than VGG-16, so the style layers were chosen from the last convolution before pooling in the corresponding VGG-19 stages, namely layers 17 and 26. With that setup, the total loss dropped from roughly 32 to 12 over training.

## Qualitative analysis
The comparison above shows the effect of the transform network clearly. The stylized output keeps more of the original structure while applying the target texture more consistently than the direct baseline.

## Implementation differences
- "Image Style Transfer Using Convolutional Neural Networks"
 - conv layers of vgg-19
 - five reference layers of style loss
- "Perceptual Losses for Real-Time Style Transfer
and Super-Resolution"
 - conv layers of vgg-16
 - four reference layers of style loss

### *References*
[1] Neural Transfer Pytorch Tutorial (https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
