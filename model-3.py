import torch
import torch.nn as nn
import torchvision.models as models


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
        # this is only a test encoder.
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        pass
    ### Need to add a hidden state as pre note book example
    
    def forward(self, features, captions):
        
        #Test forward        
        embeddings = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_words = []
        
        
        for i in range(max_len):
            # step 1: input our image input & states into our trained LSTM(RNN)
                # returns: the output from the lstm and the updated states.
            lstm_output, states = self.lstm(inputs, states)
            
            # step 2: pass the output from our lstm to the linear network
                # we get a list of predictions for that out put and we need to find the 
                # prediction with the highest value which is highest probabilty 
            predicted_value = self.linear(lstm_output)
            
            #step 3: use the argmax function to select the max prediction value.
            # & add to our list of predicted words
            best_predict = predicted_value.argmax(1)

            #step 4: add our predicted word to the embedding input
            inputs = self.embeddings(best_predict)
            
            #setp 5: finally append word to the list of predicted word that form caption.
            predicted_words.append(best_predict)
        
        return predicted_words