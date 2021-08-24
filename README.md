# Image-Captioning
Generating captions from images


## About this project

This poject is from Udacity's Computer Vision Nanodegree

## The dataset

I use the Microsoft Common Objects in Contect (MS COCO) in this project. Link: [COCO](https://cocodataset.org/#home)

### Data loader


### Model architecture

The model consists of an encoder an a decoder.

#### Encoder (pre-trained ResNet50)

I use a pre-trained ResNet-50 network to extract the features from an image. I removed the last fc layer, flattened the final output and pass through a dense layer to obtain a feature vector of size `embed_size`

```python3
import torch
import torch.nn as nn
import torchvision.models as models

class ResnetEncoder(nn.module):
  def __init__(self, embed_size):
    super(ResnetEncoder, self).__init__()
    
    resnet50 = models.resnet50(pretrained=True)
    last_fc_in_features = resnet50.fc.in_features
    # all layers excluding the last
    modules = list(resnet50.children())[:-1]
    self.resnet = nn.Sequential(*modules)
    self.embedding = nn.Linear(last_fc_in_features, embed_size)
    
  def forward(self, x):
    # pretrained resnet
    y = self.resnet(x)
    # flatten
    y = y.view(y.shape[0], -1)
    # embedding
    y = self.embedding(y)
    
    return y
```
#### Decoder (LSTM)

I use an LSTM netork as the decoder. I train the model from scratch.

```python3

```


### Hyperparameters

#### Vocaburary frequency threshold

The parameter `vocab_freq_threshold` set the minimum number that a vocabulary needs to appear in the training corpus to enter our vocabulary dictionary.

The larger the `vocab_freq_threshold`, the bigger the vocabulary dictionary.


