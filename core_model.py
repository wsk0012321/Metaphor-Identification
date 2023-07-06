import transformers
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from copy import deepcopy
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# load pre-trained model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
base_model = RobertaModel.from_pretrained('roberta-base') 
train_data = pd.read_csv(r'./data/encodings_VUA20.csv')

# read train data
def read_train_data(data):
 
    weights = [eval(weights) for weights in data['weights'].tolist()]
    weight1 = [w[0] for w in weights]
    weight2 = [w[1] for w in weights]
    labels = data['labels'].tolist()
    probability = data['probability'].tolist()
    train = [eval(subdata) for subdata in data['encodings'].tolist()]
    inputs_ids,inputs_ids2, inputs_ids3, inputs_ids4, target_mask, target_mask2, target_mask3, target_mask4, target_mask5 = [],[],[],[],[],[],[],[],[]
    for sm, wm, wsi, se, di, dm, twe, wsim, sem, ce in train:
        inputs_ids.append(ce)
        inputs_ids2.append(se)
        inputs_ids3.append(wsi)
        inputs_ids4.append(di)
        target_mask.append(sm)
        target_mask2.append(wm)
        target_mask3.append(dm)
        target_mask4.append(wsim)
        target_mask5.append(sem)
       
    return list(zip(inputs_ids,inputs_ids2, inputs_ids3, inputs_ids4, target_mask, target_mask2, target_mask3, target_mask4, target_mask5, weight1, weight2, labels, probability)) 

data = read_train_data(encodings)
print('finished data loading')

def apply_apriori_probability(tensor1,tensor2):
    tensor2 = tensor2.to('cuda:0')
    #print(tensor1, tensor2)
    tensor1 = tensor1[:,0].to('cuda:0')
  
    tensor = tensor1 * tensor2
    output_tensor = torch.tensor([value if value < 1.0 else 0.9999 for value in tensor.tolist()], dtype=torch.float32, requires_grad=True)
    return output_tensor.unsqueeze(0).float()
    
class Model(nn.Module):
    def __init__(self, num_labels=2):
        super(Model, self).__init__()
        self.encoder =  RobertaModel.from_pretrained('roberta-base')
        self.num_labels = num_labels
        self.config = RobertaConfig.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout_ratio)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.classifier2 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.classifier3 = nn.Linear(self.config.hidden_size * 3, num_labels)
        self._init_weights(self.classifier)
        self._init_weights(self.classifier2)
        self._init_weights(self.classifier3)
        self.softmax = nn.Softmax(dim=1)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    # Define the model architecture

    def forward(
        self,
        input_ids,
        input_ids2,
        input_ids3,
        input_ids4,
        target_mask,
        target_mask2,
        target_mask3,
        target_mask4,
        target_mask5,
        weight1,
        weight2
    ):  
        
        # encode the sentences
        target_sent_output = self.encoder(input_ids, attention_mask=target_mask).last_hidden_state #[0]
        
        # get sentence embedding in context
        target_sent_output = self.dropout(target_sent_output)
        
        # get word embeddings in context
        target_word_output = self.dropout(self.encoder(input_ids,attention_mask=target_mask2).last_hidden_state)
        target_word_output = target_word_output.mean(1)

        # get individual sentence embeddings
        outputs2 = self.encoder(input_ids2,attention_mask=target_mask5).last_hidden_state
        target_sent_output2 = self.dropout(outputs2)
        
        # get individual word embeddings
        outputs3 = self.encoder(input_ids3,attention_mask=target_mask4).last_hidden_state
        target_word_output2 = self.dropout(outputs3)
        target_word_output2 = target_word_output2.mean(1)
        
        # MIP layers    
        MIP_hidden = torch.cat([target_word_output, target_word_output2], dim=1)
        MIP2_hidden = torch.cat([target_sent_output, target_sent_output2], dim=2)
        MIP2_hidden = MIP2_hidden.mean(1)
        MIP_hidden = self.classifier2(MIP_hidden)
        MIP2_hidden = self.classifier2(MIP2_hidden)
        
        # Domain layer
        outputs4 = self.encoder(input_ids4,attention_mask=target_mask3).last_hidden_state
        domain_outputs = self.dropout(outputs4)
        Domain_hidden = self.classifier(domain_outputs)
        Domain_hidden = Domain_hidden[:,:90,:].mean(1)
        
        # apply weights
        weight1 = weight1.view(-1, 1).float()
        weight2 = weight2.view(-1, 1).float()
        MIP_hidden = weight2 * MIP_hidden
        MIP2_hidden = weight1 * MIP2_hidden

        # Feed-forward layers
        concated_layer = torch.cat([MIP_hidden, MIP2_hidden, Domain_hidden], dim=1)
        logits = self.classifier3(self.dropout(concated_layer))
        logits = self.softmax(logits)
        
        return logits
     
# set hyperparameters
batch_size = 32
n_fold = 8
num_epoch = 3
learning_rate = 1e-5
dropout_ratio = 0.1

# set up gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare data and instantiate model
model = Model()
print('start training')
loss_fn = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
kf = KFold(n_splits=n_fold, shuffle=True)

# set default f1score
best_fscore = 0

for train_ids, val_ids in kf.split(data):
    train_data = torch.utils.data.Subset(dataset, train_ids)
    val_data = torch.utils.data.Subset(dataset, val_ids)
    
    model.to(device)
    torch.cuda.empty_cache()
    
    print('starting next iteration')

    for epoch in range(num_epoch):
        
        model.train()
        loss_value = []
        
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
        
        for input_ids, input_ids2, input_ids3, input_ids4,target_mask, target_mask2, target_mask3, target_mask4, target_mask5, weight1, weight2, labels, prob_value in tqdm(train_dataloader):
            input_ids = [tensor.to(device) for tensor in input_ids]
            input_ids2 = [tensor.to(device) for tensor in input_ids2]
            input_ids3 = [tensor.to(device) for tensor in input_ids3]
            input_ids4 = [tensor.to(device) for tensor in input_ids4]
            
            input_ids = torch.stack(input_ids, dim=1)
            input_ids2 = torch.stack(input_ids2, dim=1)
            input_ids3 = torch.stack(input_ids3, dim=1)
            input_ids4 = torch.stack(input_ids4, dim=1)
            
            target_mask = [tensor.to(device) for tensor in target_mask]
            target_mask2 = [tensor.to(device) for tensor in target_mask2]
            target_mask3 = [tensor.to(device) for tensor in target_mask3]
            target_mask4 = [tensor.to(device) for tensor in target_mask4]
            target_mask5 = [tensor.to(device) for tensor in target_mask5]
            
            target_mask = torch.stack(target_mask, dim=1)
            target_mask2 = torch.stack(target_mask2, dim=1)
            target_mask3 = torch.stack(target_mask3, dim=1)
            target_mask4 = torch.stack(target_mask4, dim=1)
            target_mask5 = torch.stack(target_mask5, dim=1)
        
            weight1 = weight1.unsqueeze(dim=0).to(device)
            weight2 = weight2.unsqueeze(dim=0).to(device)
            labels = labels.unsqueeze(dim=0).to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, input_ids2, input_ids3, input_ids4,target_mask, target_mask2, target_mask3, target_mask4, target_mask5, weight1, weight2)
            outputs2 = outputs[:,0].view(-1, 1).to('cuda:0')
            loss = loss_fn(outputs2.transpose(0,1), labels.float())
            loss.backward()
            loss_value.append(loss.item())
            optimizer.step()
            
        print(f'finished epoch:{epoch+1}, loss:{sum(loss_value)/len(loss_value)}')
        
        model.eval()
        print(f'evaluation in process')
        
        with torch.no_grad():
            predicted, gold_labels, val_out = [],[],[]
            for input_ids, input_ids2, input_ids3, input_ids4,target_mask, target_mask2, target_mask3, target_mask4, target_mask5, weight1, weight2, val_labels, prob_value in val_dataloader:
                input_ids = [tensor.to(device) for tensor in input_ids]
                input_ids2 = [tensor.to(device) for tensor in input_ids2]
                input_ids3 = [tensor.to(device) for tensor in input_ids3]
                input_ids4 = [tensor.to(device) for tensor in input_ids4]
            
                input_ids = torch.stack(input_ids, dim=1)
                input_ids2 = torch.stack(input_ids2, dim=1)
                input_ids3 = torch.stack(input_ids3, dim=1)
                input_ids4 = torch.stack(input_ids4, dim=1)
            
                target_mask = [tensor.to(device) for tensor in target_mask]
                target_mask2 = [tensor.to(device) for tensor in target_mask2]
                target_mask3 = [tensor.to(device) for tensor in target_mask3]
                target_mask4 = [tensor.to(device) for tensor in target_mask4]
                target_mask5 = [tensor.to(device) for tensor in target_mask5]
            
                target_mask = torch.stack(target_mask, dim=1)
                target_mask2 = torch.stack(target_mask2, dim=1)
                target_mask3 = torch.stack(target_mask3, dim=1)
                target_mask4 = torch.stack(target_mask4, dim=1)
                target_mask5 = torch.stack(target_mask5, dim=1)
        
                weight1 = weight1.unsqueeze(dim=0).to(device)
                weight2 = weight2.unsqueeze(dim=0).to(device)

                val_outputs = model(input_ids, input_ids2, input_ids3, input_ids4,target_mask, target_mask2, target_mask3, target_mask4, target_mask5, weight1, weight2)
                val_out.append(val_outputs)
                val_outputs = apply_apriori_probability(val_outputs, prob_value.float()).to('cuda:0')      
                val_predicted = [1 if tensor.item() > 0.5 else 0 for tensor in val_outputs[0]]
                predicted += val_predicted
                gold_labels += val_labels.tolist()
            
            # apply evaluation
            recall = recall_score(gold_labels, predicted)
            precision = precision_score(gold_labels, predicted)
            f1score = f1_score(gold_labels, predicted)
            # print info
            print(f'Epoch: {epoch + 1}, Recall: {recall}, Precision: {precision}, F1_score: {f1score}')              
                
        if f1score > best_fscore:
            best_fscore = f1score
            optimal_weights = deepcopy(model.state_dict())
        
        torch.cuda.empty_cache()

# save weights
torch.save(best_model_weights, './optimal_weights.pth')