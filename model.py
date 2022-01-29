import torch
import torch.nn as nn
import torchvision.models as models


"""
Encoder (CNN-based architecture)
"""
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """
        Args
            embed_size - final dimension of the feature embedding of the input image
        """
        super(EncoderCNN, self).__init__()
        
        # load the pre-trained ResNet
        resnet = models.resnet50(pretrained=True)
        
        # freeze the weights
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # grab layers except the last one
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # embedding layers
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        # resnet
        features = self.resnet(images)
        # flatten
        features = features.view(features.size(0), -1)
        # embedding
        features = self.embed(features)
        
        return features
    
    
    
"""
Decoder (RNN-based architecture)
"""
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Args
            embed_size - the dimension of encoder embedding
            hidden_size - the dimension of the hidden states of decoder
            vocab_size - size of the vocabulary
            num_layers - number of decoder layers
        """
        super(DecoderRNN, self).__init__()
        
        # embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer (n=num_layers)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # dense layer from hidden states to vocab dimension
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        """
        Forward pass
        Args
            features - the output from the encoder. shape = (batch_size, embed_size)
            captions - tokenized sentences (each element is an int). shape = (batch_size, seq_len)
        """
        batch_size = features.shape[0]
        embed_size = features.shape[1]
        seq_len = captions.shape[1]
        
        # remove the end token
        captions = captions[:, :-1]
        
        # pass the tokenized captions into the embedding layer
        embedded_captions = self.embed(captions)  # (batch_size, seq_len-1, embed_size)
        
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
            inputs - <tensor> embedded image features. shape=(1, 1, embed_size)
            states - <tensor> hidden states of LSTM. Default=None
            max_len - <int> maximum length of the decoded tokens
            
        Returns
            tokens - <list> a list of tokens outputed by the decoder
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
            x = self.embed(tok)  # (batch_size, 1, embed_size)
        
        return tokens