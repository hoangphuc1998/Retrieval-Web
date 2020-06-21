from torch.nn import Sequential
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch
from transformers import *
import os
import time

def normalize(X):
    '''
    Normalize input tensor to zero mean and unit variance
    Input: tensor of size (batch_size, _)
    Output: normalized tensor with same size of input
    '''
    mean = torch.mean(X, dim=1, keepdim=True)
    std = torch.std(X, dim=1, keepdim=True)
    return (X - mean) / std

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, hidden_activation='relu', output_activation='relu', use_dropout = False, use_batchnorm=False):
        super().__init__()
        self.network = nn.Sequential()
        hidden_units = [input_dim] + hidden_units
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        
        for i in range(len(hidden_units) - 1):
            self.network.add_module("dense_" + str(i), nn.Linear(hidden_units[i], hidden_units[i+1]))
            # Hidden activation
            if hidden_activation == 'relu':
              self.network.add_module("activation_" + str(i), nn.ReLU())
            elif hidden_activation == 'sigmoid':
              self.network.add_module("activation_" + str(i), nn.Sigmoid())
            elif hidden_activation == 'tanh':
              self.network.add_module("activation_" + str(i), nn.Tanh())
            elif hidden_activation == 'lrelu':
              self.network.add_module("activation_" + str(i), nn.LeakyReLU())
            elif hidden_activation == 'prelu':
              self.network.add_module("activation_" + str(i), nn.PReLU())
            # Batchnorm on hidden layers
            if self.use_batchnorm:
              self.network.add_module("batchnorm_" + str(i), nn.BatchNorm1d(hidden_units[i+1]))
        
        # Dropout with 20% probability
        if self.use_dropout:
          self.network.add_module("dropout", nn.Dropout(0.2))

        self.network.add_module("output", nn.Linear(hidden_units[-1], output_dim))
        # Output activation
        if output_activation == 'relu':
          self.network.add_module("activation_out", nn.ReLU())
        elif output_activation == 'sigmoid':
          self.network.add_module("activation_out", nn.Sigmoid())
        elif output_activation == 'tanh':
          self.network.add_module("activation_out", nn.Tanh())

    def forward(self, x):
        return self.network(x)

class BertFinetune(nn.Module):
  def __init__(self, bert_model, output_type='cls'):
    super().__init__()
    self.bert_model = bert_model
    self.output_type = output_type
    #self.dropout = nn.Dropout(0.2)
  def forward(self, input_ids, attention_mask):
    output = self.bert_model(input_ids, attention_mask = attention_mask)
    if self.output_type == 'mean':
      feature = (output[0] * attention_mask.unsqueeze(2)).sum(dim=1).div(attention_mask.sum(dim=1, keepdim=True))
    elif self.output_type == 'cls2':
      feature = torch.cat((output[2][-1][:,0,...], output[2][-2][:,0,...]), -1)
    elif self.output_type == 'cls4':
      feature = torch.cat((output[2][-1][:,0,...], output[2][-2][:,0,...], output[2][-3][:,0,...], output[2][-4][:,0,...]), -1)
    else:
      feature = output[2][-1][:,0,...]
    return feature

def load_transform_model(opt, text_encoder_path, device, image_encoder_path = ''):
    '''
    Load image and text encoder model
    Input:
        - opt: Hyperparameters dictionary
        - image_encoder_path: where image encoder pth state dict
        - text_encoder_path: where text encoder pth state dict
        - device: where models are moved to
    Output: image_encoder, text_encoder
    '''
    # Initialize models
    text_encoder = NeuralNetwork(input_dim=opt['text_dim'], 
                              output_dim=opt['common_dim'], 
                              hidden_units=opt['text_encoder_hidden'], 
                              hidden_activation=opt['text_encoder_hidden_activation'], 
                              output_activation=opt['text_encoder_output_activation'],
                              use_dropout=opt['use_dropout'],
                              use_batchnorm=opt['use_batchnorm']).to(device)
    
    # Load models
    text_encoder.load_state_dict(torch.load(text_encoder_path,map_location=device))
    # Change to eval mode
    text_encoder.eval()
    # Load image encoder model
    if len(image_encoder_path) > 0:
        image_encoder = NeuralNetwork(input_dim=opt['image_dim'], 
                              output_dim=opt['common_dim'], 
                              hidden_units=opt['image_encoder_hidden'], 
                              hidden_activation=opt['image_encoder_hidden_activation'], 
                              output_activation=opt['image_encoder_output_activation'],
                              use_dropout=opt['use_dropout'],
                              use_batchnorm=opt['use_batchnorm']).to(device)
        image_encoder.load_state_dict(torch.load(image_encoder_path, map_location=device))
        image_encoder.eval()
        return image_encoder, text_encoder
    return text_encoder

def load_text_model(model_type, pretrained, output_type, device, model_path=''):
    '''
    Load model to extract feature from text
    Input:
        - model_type (str): Type of models: bert
        - pretrained (str): Type of pretrained weights
        - device (torch.device): where model is saved when loaded
    Output:
        - model and tokenizer
    '''
    #TODO: Add RoBerta and others
    if model_type == 'bert':
        config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True)
        bert = BertModel(config).to(device)
        model = BertFinetune(bert, output_type)
        if len(model_path)>0:
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return model, tokenizer
    elif model_type == 'roberta':
        config = RobertaConfig.from_pretrained('roberta-base', output_hidden_states = True)
        bert = RobertaModel(config).to(device)
        model = BertFinetune(bert, output_type)
        if len(model_path)>0:
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        return model, tokenizer