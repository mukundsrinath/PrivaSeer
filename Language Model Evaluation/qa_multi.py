import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import json
import pandas as pd
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
print(torch.cuda.is_available())
device = torch.device("cuda")

writer = SummaryWriter()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
MAX_SEQ_LEN = 512
BATCH_SIZE = 96
destination_folder = '/data/mus824/models/qa_models'

import pandas as pd
source_directory = '/data/mus824/data/'

train = pd.read_csv(source_directory+'privacyqa_train', '\t')
test = pd.read_csv(source_directory+'privacyqa_test', '\t')

_train = []
for i in range(149600):
    _train.append([train.question.values[i], train.answer.values[i]])

train_labels = train.label.values[:149600]

validation = []
for i in range(149600, len(train.question.values)):
    validation.append([train.question.values[i], train.answer.values[i]])

validation_labels = train.label.values[149600:]


_test = []
for i in range(len(test.question.values)):
    _test.append([test.question.values[i], test.answer.values[i]])

test_labels = test.majority.values

#_train = [line for line in _train if (len(line) > 0 and not line.isspace())]
#validation = [line for line in validation if (len(line) > 0 and not line.isspace())]
#test = [line for line in test if (len(line) > 0 and not line.isspace())]

#print(len())

_train = tokenizer(_train, add_special_tokens=True, truncation=True, max_length=None, padding=True)
validation = tokenizer(validation, add_special_tokens=True, truncation=True, max_length=None, padding=True)
test = tokenizer(_test, add_special_tokens=True, truncation=True, max_length=None, padding=True)

#print(len(_train['input_ids']))
#print(len(train_labels))

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        #self.l1 = RobertaModel.from_pretrained('/data/mus824/models/epoch_3/')
        #self.l1 = RobertaModel.from_pretrained('/data/mus824/python-files/privacy_bert2/checkpoint-48000/')
        self.l1 = RobertaModel.from_pretrained('roberta-base')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 2)
    
    def forward(self, ids, mask):
        _, output_1= self.l1(ids, attention_mask = mask, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

def batch(text, label, batch=BATCH_SIZE):
    length = len(text['input_ids'])
    for i in range(0, length, batch):
        yield torch.tensor(label[i:i+batch]), torch.tensor(text['input_ids'][i:i+batch]), torch.tensor(text['attention_mask'][i:i+batch])
        	
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
    
    state_dict = torch.load(load_path, map_location=device)
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
    
    state_dict = torch.load(load_path, device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']
	
def train(model, optimizer, _train=_train, train_labels=train_labels, validation=test, validation_labels=test_labels, num_epochs = 5, eval_every = 500, file_path = destination_folder, best_valid_loss = float("Inf")):

    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    val_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []	
    model.train()
    for epoch in range(num_epochs):
        for t_labels, text_ids, text_masks in batch(_train, train_labels):
            t_labels = t_labels.type(torch.LongTensor)
            t_labels = t_labels.to(device)
            text_ids = text_ids.to(device)
            text_masks = text_masks.to(device)
            logits = model(ids=text_ids, mask=text_masks)
            loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([1,8]).double().to(device))
            loss = loss_func(logits.double(), t_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_loss', loss.item(), global_step)
            running_loss += loss.item()
            global_step += 1
			
            if global_step % eval_every == 0:
                model.eval()
                val_iter = 1
                with torch.no_grad():  
                    for val_labels, val_text_ids, val_text_masks, in batch(validation, validation_labels):
                        val_labels = val_labels.type(torch.LongTensor)           
                        val_labels = val_labels.to(device)
                        #text = text.type(torch.LongTensor)  
                        val_text_ids = val_text_ids.to(device)
                        val_text_masks = val_text_masks.to(device)
                        output = model(val_text_ids, val_text_masks)
                        valid_loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([1,8]).double().to(device))
                        valid_loss = loss_func(output.double(), val_labels)
                        valid_loss = valid_loss.item()
                        valid_running_loss += valid_loss
                        val_iter += 1
                        val_step += 1
                        writer.add_scalar('valid_loss', valid_loss, val_step)			
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
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch+1, num_epochs, global_step, int(num_epochs*len(train_labels)/BATCH_SIZE), average_train_loss, average_valid_loss))
							  
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
					
    #save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
	
#model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=12)
model = BERTClass()
model = nn.DataParallel(model, device_ids=[0,1])
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-6)

train(model=model, optimizer=optimizer)

train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt')

# Evaluation Function

def evaluate(model, test, test_labels):
    y_pred = []
    y_true = []
    soft = nn.Softmax()
    probas = []

    model.eval()
    with torch.no_grad():
        for t_labels, text_ids, text_masks in batch(test, test_labels):
            #labels = labels.type(torch.LongTensor)           
            t_labels = t_labels.to(device)
            text_ids = text_ids.to(device)
            text_masks = text_masks.to(device)
            output = model(text_ids, text_masks)
            probas.extend(soft(output[:,1]).cpu().detach().numpy().tolist())
            y_pred.extend(torch.argmax(output, 1).cpu().detach().numpy().tolist())
            y_true.extend(t_labels.cpu().detach().numpy().tolist())
 
    print('Classification Report:')
    print(classification_report(y_true, y_pred, digits=4))
   
best_model = BERTClass()
best_model = nn.DataParallel(best_model, device_ids=[0,1])
best_model.to(device)
load_checkpoint(destination_folder + '/model.pt', best_model)
probas, y_pred, y_true = evaluate(best_model, test, test_labels)
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(2):
#    fpr[i], tpr[i], _ = roc_curve(y_true, probas)
#    roc_auc[i] = auc(fpr[i], tpr[i])

#plt.figure()
#lw = 2
#plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC (AUC = %0.2f)' % roc_auc[1])
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.savefig('figure.png')
