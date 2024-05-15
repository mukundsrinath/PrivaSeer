import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
import os
import json
from sklearn import metrics
os.environ['CUDA_VISIBLE_DEVICES']='0,2,3'
print(torch.cuda.is_available())

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
MAX_SEQ_LEN = 512
BATCH_SIZE = 8
destination_folder = '/data/mus824/models/classification_models'

import pandas as pd
source_directory = '/data/mus824/data/privacy-practice/'
train = pd.read_csv(source_directory+'train_dataset.csv')
validation = pd.read_csv(source_directory+'validation_dataset.csv')
test = pd.read_csv(source_directory+'test_dataset.csv')

with open(source_directory+'practice_labels.json') as fp:
    labels = json.load(fp)

train_labels = np.zeros((len(train), len(labels)))
validation_labels = np.zeros((len(validation), len(labels)))
test_labels = np.zeros((len(test), len(labels)))

for i, line in train.iterrows():
    train_labels[i, labels[line.label]] = 1

for i, line in validation.iterrows():
    validation_labels[i, labels[line.label]] = 1

for i, line in test.iterrows():
    test_labels[i, labels[line.label]] = 1

train = [line for line in train.text if (len(line) > 0 and not line.isspace())]
validation = [line for line in validation.text if (len(line) > 0 and not line.isspace())]
test = [line for line in test.text if (len(line) > 0 and not line.isspace())]

train = tokenizer(train, add_special_tokens=True, truncation=True, max_length=None, padding=True)
validation = tokenizer(validation, add_special_tokens=True, truncation=True, max_length=None, padding=True)
test = tokenizer(test, add_special_tokens=True, truncation=True, max_length=None, padding=True)

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        #self.l1 = RobertaModel.from_pretrained('/data/mus824/nessie')
        #self.l1 = RobertaModel.from_pretrained('roberta-base')
        self.l1 = RobertaModel.from_pretrained('/data/mus824/python-files/privacy_bert2/checkpoint-48000/')
        self.l2 = torch.nn.Dropout(0.5)
        self.l3 = torch.nn.Linear(768, 12)

    def forward(self, ids, mask):
        _, output_1= self.l1(ids, attention_mask = mask, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

def batch(text_batch, label_batch, _batch=BATCH_SIZE):
    length = len(label_batch)
    for i in range(0, length, _batch):
        yield torch.tensor(label_batch[i:i+_batch]), torch.tensor(text_batch['input_ids'][i:i+_batch]), torch.tensor(text_batch['attention_mask'][i:i+_batch])
        	
def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')
	
def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=torch.device('cuda'))
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']
	
def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, torch.device('cuda'))
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']
	
def train_(model, optimizer, train=train, train_labels=train_labels, validation=validation, validation_labels=validation_labels, num_epochs = 1, eval_every = 100, file_path = destination_folder, best_valid_loss = float("Inf")):

    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []	
    model.train()
    for epoch in range(num_epochs):
        for labels, text_ids, text_masks in batch(train, train_labels):
            #labels = labels.type(torch.LongTensor)
            labels = labels.cuda()
            text_ids = text_ids.cuda()
            text_masks = text_masks.cuda()
            #text = text.to(device)
            logits = model(input_ids=text_ids, attention_mask=text_masks)[0]
            loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1
			
            if global_step % eval_every == 0:
                model.eval()
                val_iter = 1
                with torch.no_grad():  
                    for val_labels, val_text_ids, val_text_masks, in batch(validation, validation_labels):
                        #labels = labels.type(torch.LongTensor)           
                        val_labels = val_labels.cuda()
                        #text = text.type(torch.LongTensor)  
                        val_text_ids = val_text_ids.cuda()
                        val_text_masks = val_text_masks.cuda()
                        output = model(text_ids, text_masks)[0]
                        valid_loss_func = BCEWithLogitsLoss()
                        valid_loss = loss_func(output, val_labels)
                        valid_running_loss += valid_loss.item()
                        val_iter += 1				
                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / val_iter
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
				
                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()
				
                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch+1, num_epochs, global_step, int(num_epochs*len(train_labels)/4), average_train_loss, average_valid_loss))
							  
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
					
    #save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
	
#model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=12)
#model.cuda()
#optimizer = optim.Adam(model.parameters(), lr=2e-5)

#train(model=model, optimizer=optimizer)

#train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt')

# Evaluation Function

def evaluate(model, x, y):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for labels, text_ids, text_masks in batch(x, y):
            #labels = labels.type(torch.LongTensor)           
            labels = labels.cuda()
            text_ids = text_ids.cuda()
            text_masks = text_masks.cuda()
            output = model(text_ids, text_masks)
            #loss_func = BCEWithLogitsLoss()
            #loss = loss_func(output, labels)
            y_true.extend(labels.cpu().detach().numpy().tolist())
            y_pred.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())
    y_pred = np.array(y_pred) >= 0.5
    print('Classification Report:')
    f1_score_macro = metrics.f1_score(y_true, y_pred, average='macro')
    print(f1_score_macro)
    print(classification_report(y_true, y_pred, digits=4))
    
    #cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    #ax= plt.subplot()
    #sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    #ax.set_title('Confusion Matrix')

    #ax.set_xlabel('Predicted Labels')
    #ax.set_ylabel('True Labels')

    #ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    #ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
	
best_model = BERTClass()
best_model.cuda()
load_checkpoint(destination_folder + '/model.pt', best_model)
evaluate(best_model, test, test_labels)
