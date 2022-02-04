"""
COCO Image Captioning - Encoder and Decoder Models

EncoderCNN - ResNet encoder
DecoderRNN - LSTM decoder
"""

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """
    Encoder (ResNet-based)
    """
    def __init__(self, embed_size):
        """
        Args
            embed_size: (int) dimension of image semantics features to be encoded
        """
        super(EncoderCNN, self).__init__()
        # load the pre-trained ResNet
        resnet = models.resnet50(pretrained=True)
        # freeze the weights
        for param in resnet.parameters():
            param.requires_grad_(False)
        # grab all CNN layers except the last one
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # embedding layers
        self.embedding = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        """
        Args
            images: (tensor) processed image tensor. shape=(batch_size, 3, 224, 224)
        Returns
            feature: (tensor) extracted image semantic features. shape=(batch_size, self.embed_size)
        """
        # resnet stage
        features = self.resnet(images)
        # flatten to 1 dim
        features = features.view(features.size(0), -1)
        # embedding to final feature
        features = self.embedding(features)
        return features


class DecoderRNN(nn.Module):
    """
    Decoder (LSTM-based)
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Args
            embed_size: (int) dimension of extracted image semantics features
            hidden_size: (int) dimension of decoder hidden states
            vocab_size: (int) size of vocabulary
            num_layers: (int) number of decoder layers
        """
        super(DecoderRNN, self).__init__()
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # LSTM layer(s)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # dense layer from hidden states to vocab dimension
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        Args
            features: (tensor) encoder output. shape=(batch_size, embed_size)
            captions: (tensor) caption tokens (each element is an int). shape=(batch_size, seq_len)
        Returns
            fc_output: (tensor) final output. shape=(batch, vocab_size)
        """
        # batch size
        batch_size = features.shape[0]
        # embedding dimension
        embed_size = features.shape[1]
        # caption length
        seq_len = captions.shape[1]
        # remove the <end> token
        captions = captions[:, :-1]
        # pass the tokenized captions into the embedding layer
        embedded_captions = self.embedding(captions)  # (batch_size, seq_len-1, embed_size)
        # convert features as the very first tokens
        features = torch.unsqueeze(features, dim=1)  # (batch_size, 1, embed_size)
        # concatenate to obtain lstm_input
        lstm_input = torch.cat((features, embedded_captions), dim=1)  # (batch_size, seq_len, embed_size)
        # LSTM layer
        lstm_output, lstm_hidden = self.lstm(lstm_input)
        # dense layer
        fc_output = self.fc(lstm_output)
        return fc_output


    def sample(self, inputs, states=None, max_len=20):
        """
        Decode an image from the embedded feature tensor.
        Args
            inputs: (tensor) embedded image features. shape=(1, 1, embed_size)
            states: (tensor) hidden states of LSTM. shape=(1, hidden_size)
            max_len: (int) maximum length of predicted token list
        Returns
            tokens: (list) a list of tokens predicted by decoder
        """
        tokens = []
        x = inputs
        # output tokens one by one
        for _ in range(max_len):
            # lstm layer
            x, states = self.lstm(x, states)  # (batch_size=1, 1, hidden_size)
            # dense layer
            x = self.fc(x)  # (batch_size=1, 1, vocab_size)
            # token
            tok = torch.argmax(x, dim=-1)  # (batch_size=1, 1)
            # append to the output
            tokens.append(int(tok[0, 0]))
            # early stop (token == 1)
            if tok[0, 0] == 1:
                break
            # embedding
            x = self.embedding(tok)  # (batch_size, 1, embed_size)
        return tokens
