from transformers import RobertaTokenizer, RobertaModel
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import json
import pandas as pd
print(torch.cuda.is_available())
device = "cuda:1"
device_number = 1
writer = SummaryWriter()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
MAX_SEQ_LEN = 512
BATCH_SIZE = 16
destination_folder = ''

source_directory = ''
_train = pd.read_csv(source_directory+'train_dataset.csv')
validation = pd.read_csv(source_directory+'val_dataset.csv')
test = pd.read_csv(source_directory+'test_dataset.csv')

with open(source_directory+'practice_labels.json') as fp:
    labels = json.load(fp)

train_labels = _train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']].to_numpy()
validation_labels = validation[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']].to_numpy()
test_labels = np.zeros((len(test), len(labels)))

for i, line in test.iterrows():
    test_labels[i, labels[line.label]] = 1

_train = [line for line in _train.text if (len(line) > 0 and not line.isspace())]
validation = [line for line in validation.text if (len(line) > 0 and not line.isspace())]
test = [line for line in test.text if (len(line) > 0 and not line.isspace())]

_train = tokenizer(_train, add_special_tokens=True, truncation=True, max_length=None, padding=True)
validation = tokenizer(validation, add_special_tokens=True, truncation=True, max_length=None, padding=True)
test = tokenizer(test, add_special_tokens=True, truncation=True, max_length=None, padding=True)

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained('mukund/privbert')
        self.l2 = torch.nn.Dropout(0.15)
        self.l3 = torch.nn.Linear(768, 12)
    
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
    
    state_dict = torch.load(load_path, map_location=torch.device(device))
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
    
    state_dict = torch.load(load_path, torch.device(device))
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']
	
def train(model, optimizer, _train=_train, train_labels=train_labels, validation=validation, validation_labels=validation_labels, num_epochs = 5, eval_every = 137, file_path = destination_folder, best_valid_loss = float("Inf")):

    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    val_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []	
    model.train()
    for epoch in range(num_epochs):
        for labels, text_ids, text_masks in batch(_train, train_labels):
            #labels = labels.type(torch.LongTensor)
            labels = labels.cuda(device_number)
            text_ids = text_ids.cuda(device_number)
            text_masks = text_masks.cuda(device_number)
            #text = text.to(device)
            logits = model(ids=text_ids, mask=text_masks)
            loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits.double(), labels.double())
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
                        #labels = labels.type(torch.LongTensor)           
                        val_labels = val_labels.cuda(device_number)
                        #text = text.type(torch.LongTensor)  
                        val_text_ids = val_text_ids.cuda(device_number)
                        val_text_masks = val_text_masks.cuda(device_number)
                        output = model(val_text_ids, val_text_masks)
                        valid_loss_func = BCEWithLogitsLoss()
                        valid_loss = valid_loss_func(output.double(), val_labels.double())
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
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch+1, num_epochs, global_step, int(num_epochs*len(train_labels)/16), average_train_loss, average_valid_loss))
							  
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
					
    #save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
	
#model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=12)
model = BERTClass()
model.cuda(device_number)
optimizer = optim.Adam(model.parameters(), lr=2.5e-5)

train(model=model, optimizer=optimizer)

train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt')

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
	
best_model = BERTClass()
best_model.cuda(device_number)
load_checkpoint(destination_folder + '/model.pt', best_model)
evaluate(best_model, test, test_labels)
