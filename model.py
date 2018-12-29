import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # LSTM cell
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout=0, batch_first=True)
        
        # Dropout layer
        #self.dropout = nn.Dropout(0.5)
        
        # Fully-connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Initialize bias tensor to all zeros
        #self.fc.bias.data.fill_(0)
        
        # Initialize FC weights as random uniform
        #self.fc.weight.data.uniform_(-1, 1)
    
    def forward(self, features, captions):
        
        # Discard the <end> word to avoid predicting when <end> is the input of the RNN, 
        # which would produce an extra 'invalid' output
        captions = captions[:, :-1]  
        
        self.hidden = self.init_hidden(features.shape[0])
        
        # Create embedded word vectors for each word in the captions
        embeddings = self.word_embeddings(captions) 
        # shape: (batch_size, captions_length, embed_size)
        
        # Stack the features and captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) 
        # shape: (batch_size, captions_length, embed_size)
        
        # Get the output and hidden state by passing the lstm our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden) 
        # output shape: (batch_size, captions_length, hidden_size)
        
        outputs = self.fc(lstm_out)
        # outputs shape : (batch_size, captions_length, vocab_size)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        outputs = []
        
        for i in range(max_len):
            
            lstm_out, states = self.lstm(inputs, states)
            #shape : (1, 1, hidden_size)
            #print("lstm_out shape:", lstm_out.shape)
            #print("lstm_out type:", type(lstm_out))
            
            output = self.fc(lstm_out)
            # shape: (1, 1, vocab_size)
            #print(output.shape)
            
            output = output.squeeze(1)
            # shape: (1, vocab_size)
            #print(output.shape)
    
            # Retrieve highest probability vocabulary value
            _, max_index = torch.max(output, dim=1)
            #print("max_index shape:", max_index.shape)
            #print("max_index type:", type(max_index))
            
            #print(type(max_index.cpu().numpy()[0]))
            #print(max_index.cpu().numpy().shape)
            #print(type(max_index.cpu().numpy()[0].item()))
    
            # Use Tensor.cpu() to copy the tensor to host memory first (can't directly convert CUDA tensor to numpy)
            # Output should be a list of integers (int vs numpy.int64)
            # (add .item() - Copies an element of a numpy array to a standard Python scalar and returns it.)
            outputs.append(max_index.cpu().numpy()[0].item())
        
            if (max_index == 1):
                # If <end> is predicted, break
                break
    
            # Embed the last predicted word and input into the LSTM
            inputs = self.word_embeddings(max_index) 
            # shape: (1, embed_size)
            
            inputs = inputs.unsqueeze(1) 
            # shape: (1, 1, embed_size) - for our network, embed_size = hidden_size

        return outputs
    
    def init_hidden(self, batch_size):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros((1, batch_size, self.hidden_size), device="cuda"), 
                torch.zeros((1, batch_size, self.hidden_size), device="cuda"))