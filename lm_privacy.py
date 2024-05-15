import os
os.environ['CUDA_VISIBLE_DEVICES']='0,2,3'
import torch
print(torch.cuda.is_available())
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')
print('Number of parameters available'+ str(model.num_parameters()))

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="privacy_bert",   
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=16,
    save_steps=10000,
    save_total_limit=10,
)

#tokenizer.enable_truncation(max_length=64)

from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

for i in range(0, 26):
    training_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="training-sets/"+str(i),
        block_size=128
    )

    print('done with loading dataset')

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=training_dataset,
        prediction_loss_only=True
    )

    trainer.train()
    trainer.save_model("privcy_bert")
