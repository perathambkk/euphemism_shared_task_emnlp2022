import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_func

class KimCNNTwoStreamsFT(nn.Module):
    def __init__(self, word_model, word_model2, **config):
        """
        Creates a CNN sentence classification module as described by Yoon Kim.

        Args:
            word_model: the input layer model to use. Can be rand, static, non-static, or multi-channel, as represented by
                SingleChannelWordModel and MultiChannelWordModel.
            **config: A dictionary of configuration settings. If blank, it defaults to those recommended in Kim's paper.
        """
        # In PyTorch, we typically initialize all the trainable modules/layers in __init__ and then use them in forward()
        super().__init__()
        n_fmaps = config.get("n_feature_maps", 100)
        weight_lengths = config.get("weight_lengths", [3, 4, 5]) # the sizes of the convolutional kernel
        embedding_dim = config.get("hidden_size", 768)
        # n_fmaps2 = config2.get("n_feature_maps", 100)
        # weight_lengths2 = config2.get("weight_lengths", [3, 4, 5]) # the sizes of the convolutional kernel
        # embedding_dim2 = config2.get("hidden_size", 768)
        # config.output_hidden_states = True

        # By doing self.word_model = word_model, word_model is now a sub-module of KimCNN: all its parameters are now
        # part of KimCNN's parameters.
        self.word_model = word_model
        self.word_model.output_hidden_states=True
        self.word_model2 = word_model2
        self.word_model2.output_hidden_states=True
        n_c = 2*13 #word_model.n_channels


        # The convolutional layers, 3 of 3x300, 4x300, 5x300 by default. (300 is the embedding size)
        self.conv_layers = [nn.Conv2d(n_c, n_fmaps, (w, embedding_dim), padding=(w - 1, 0)) for w in weight_lengths]
        for i, conv in enumerate(self.conv_layers):
            self.add_module("conv{}".format(i), conv) # since conv_layers is a list, we need to add the modules manually
        self.dropout = nn.Dropout(config.get("dropout", 0.5)) # a dropout layer
        # Finally linearly combine all conv layers to form a logits output for softmax with cross entropy loss...
        # There are 5 sentiments in SST by default: very negative, negative, neutral, positive, very positive
        self.fc = nn.Linear(len(self.conv_layers) * n_fmaps, config.get("n_labels", 2)) 
        self.criterion = nn.CrossEntropyLoss()

    # def preprocess(self, sentences):
    #     # Preprocess the string sentences for input to the model. In other words, takes a list of string sentences and outputs
    #     # its embedding tensor representation
    #     return torch.from_numpy(np.array(self.word_model.lookup(sentences)))

    def forward(self, x):
        # Runs x through the current word input model, which is one of rand, static, non-static, and multi-channel
        if self.training:
            labels = x['labels'].detach().clone()
        x1 = self.word_model(**x, output_hidden_states=True).hidden_states # shape: (batch, channel, sent length, embed dim)
        x2 = self.word_model2(**x, output_hidden_states=True).hidden_states
        x1 = torch.stack(list(x1), dim = 1)
        x2 = torch.stack(list(x2), dim = 1)
        x = torch.cat([x1, x2], dim = 1)
        # Perform convolution with rectified linear units as recommended in most papers
        x = [nn_func.relu(conv(x)).squeeze(3) for conv in self.conv_layers] # squeeze(3) to get rid of the extraneous dimension
        # max-pool over time as mentioned in the paper
        x = [nn_func.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        # Concatenate along the second dimension:
        x = torch.cat(x, 1)
        # Apply dropout
        x = self.dropout(x)
        # Return logits
        logits = self.fc(x)
        if self.training:
            loss = self.criterion(nn_func.softmax(logits, dim=1), labels)
        else:
            loss = None
        return loss, logits