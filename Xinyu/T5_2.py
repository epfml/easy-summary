'''
sum_sim model
'''

from functools import lru_cache
from gc import callbacks
from lib2to3.pgen2 import token
from pathlib import Path
from weakref import ref
import math
from pytorch_lightning.loggers import TensorBoardLogger
from easse.sari import corpus_sari
from torch.nn import functional as F
from preprocessor import tokenize, yield_sentence_pair, yield_lines, load_preprocessor, read_lines, \
    count_line, OUTPUT_DIR, get_complexity_score, safe_division, get_word2rank, remove_stopwords, remove_punctuation
# import Levenshtein
import argparse
from argparse import ArgumentParser
import os
import logging
import random
import nltk
from preprocessor import  get_data_filepath
#from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer import seed_everything
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    BertTokenizer, BertForPreTraining,
    BartForConditionalGeneration, BartTokenizer,pipeline,BartTokenizerFast, BartModel,
    get_linear_schedule_with_warmup, AutoConfig, AutoModel,
    get_cosine_schedule_with_warmup,
)
from Ts_T5 import T5FineTuner
#from sentence_transformers import SentenceTransformer
# from keybert import KeyBERT
#kw_model = KeyBERT()
#BERT_Sum = Summarizer(model='distilbert-base-uncased')

class MetricsCallback(pl.Callback):
  def __init__(self):
    super().__init__()
    self.metrics = []
  
  def on_validation_end(self, trainer, pl_module):
      self.metrics.append(trainer.callback_metrics)



class SumSim(pl.LightningModule):
    def __init__(self,args):
        super(SumSim, self).__init__()
        self.args = args
        self.save_hyperparameters()

        self.summarizer = T5ForConditionalGeneration.from_pretrained(self.args.sum_model)
        self.summarizer_tokenizer = T5TokenizerFast.from_pretrained(self.args.sum_model)
        self.summarizer = self.summarizer.to(self.args.device)

        self.simplifier = T5FineTuner.load_from_checkpoint('experiments/exp_T5_FineTuned_WikiLarge/checkpoint-epoch=2.ckpt')
        self.simplifier = self.simplifier.model.to(self.args.device)
        self.simplifier_tokenizer = T5TokenizerFast.from_pretrained(self.args.sim_model)

        
        
        self.W = torch.randn((768, int(self.args.hidden_size)), requires_grad=True, device = self.args.device)

        self.CosSim = nn.CosineSimilarity(dim = 2, eps = 1e-6)
        self.relu = nn.ReLU()


    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(self, input_ids, 
    attention_mask = None,
    decoder_input_ids = None,
    decoder_attention_mask = None, labels = None):
        
        outputs = self.simplifier(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_input_ids = decoder_input_ids,
            decoder_attention_mask =  decoder_attention_mask,
            labels = labels
        )

        return outputs


    def training_step(self, batch, batch_idx):
        source = batch["source"]
        labels = batch['target_ids']
        targets = batch['target']
        labels[labels[:,:] == self.simplifier_tokenizer.pad_token_id] = -100
        # zero the gradient buffers of all parameters
        self.opt.zero_grad()
        #print(source, len(source))

        #### tokenize targets
        targets_encoding = self.simplifier_tokenizer(
            targets,
            max_length = 256,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt'
        )
        tgt_ids = targets_encoding['input_ids'].to(self.args.device)
        tgt_mask = targets_encoding['attention_mask'].to(self.args.device)
        
        tgt_output = self.simplifier(
            input_ids =  tgt_ids,
            attention_mask = tgt_mask,
            labels = labels,
            decoder_attention_mask = batch['target_mask']
        )
        H_sim = tgt_output.encoder_last_hidden_state


        ## summarizer stage
        inputs = self.summarizer_tokenizer(
            source,
            max_length = 512, # 1024
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt'
        )#.to(self.args.device)

        
        src_ids = inputs['input_ids'].to(self.args.device)
        src_mask = inputs['attention_mask'].to(self.args.device)


        # compute the loss between summarization and simplification target
        # sum_outputs.loss

        sum_outputs = self.summarizer(
            input_ids = src_ids,
            attention_mask  = src_mask,
            labels = labels,
            decoder_attention_mask = batch['target_mask']
        )
        
        #H1 = sum_outputs.encoder_last_hidden_state

        # generate summary
        summary_ids = self.summarizer.generate(
            inputs['input_ids'].to(self.args.device),
            do_sample = True,
            num_beams = 5,
            min_length = 10,
            max_length = 256, # 512
        ).to(self.args.device)
        
 
        
        # add 'simplify: ' ids
        # 18356, 10
        for i, summary_id in enumerate(summary_ids):
            added_tokens = torch.tensor([18356, 10]).to(self.args.device)
            summary_ids[i] = torch.cat((added_tokens, summary_id), dim=0)[:-2]
        
        ### modified loss
        padded_summary_ids = torch.zeros((summary_ids.shape[0], 256), dtype = torch.long).fill_(self.simplifier_tokenizer.pad_token_id).to(self.args.device)
        for i, summary_id in enumerate(summary_ids):
            padded_summary_ids[i,:summary_id.shape[0]]=summary_id


        summary_attention_mask = torch.ones(padded_summary_ids.shape).to(self.args.device)
        summary_attention_mask[padded_summary_ids[:,:]==self.summarizer_tokenizer.pad_token_id]=0
            

        
        # forward pass
        sim_outputs  = self(
            # summary_ids -> padded
            input_ids = padded_summary_ids,
            attention_mask = summary_attention_mask,
            labels = labels,
            decoder_attention_mask = batch['target_mask']
        )

        H2 = sim_outputs.encoder_last_hidden_state

        



        Rep1 = torch.matmul(H_sim, self.W)
        Rep2 = torch.matmul(H2, self.W)
        Rep1 = self.relu(Rep1)
        Rep2 = self.relu(Rep2)
        # Rep1 = H_sim
        # Rep2 = H2

        sim_score = self.CosSim(Rep1, Rep2)

        ###################


        if self.args.custom_loss:

            
            loss = sim_outputs.loss * self.args.w1
            #loss += sum_outputs.loss * self.args.w2

            ### CosSim ###
            loss += (-self.args.lambda_ * (sim_score.mean(dim=1).mean(dim=0)))

     
            self.log('train_loss', sim_outputs.loss, on_step=True, prog_bar=True, logger=True)
            # print(loss)
            return loss
        else:
            loss = sim_outputs.loss
            self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
            #print(loss)
            return loss


    def validation_step(self, batch, batch_idx):
        loss = self.sari_validation_step(batch)
        # loss = self._step(batch)
        print("Val_loss", loss)
        logs = {"val_loss": loss}

        self.log('val_loss', loss, batch_size = self.args.valid_batch_size)
        return torch.tensor(loss, dtype=float)

    def sari_validation_step(self, batch):
        def generate(sentence):
            #sentence = self.preprocessor.encode_sentence(sentence)

            text = "summarize: " + sentence
            # summarize the document
            inputs = self.summarizer_tokenizer(
            [text],
            max_length = 512, #1024
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt'
        )
            # generate summary
            summary_ids = self.summarizer.generate(
                inputs['input_ids'].to(self.args.device),
                num_beams = 15,
                #min_length = 30,
                max_length = 256, # 512
                top_k = 130, top_p = 0.95
            ).to(self.args.device)
            
            for i, summary_id in enumerate(summary_ids):
                add_tokens = torch.tensor([18356, 10]).to(self.args.device)
                summary_ids[i,:] = torch.cat((summary_id, add_tokens), dim=0)[:-2]

            summary_attention_mask = torch.ones(summary_ids.shape).to(self.args.device)
            summary_attention_mask[summary_ids[:,:]==self.summarizer_tokenizer.pad_token_id]=0

            


            # set top_k = 130 and set top_p = 0.95 and num_return_sequences = 1
            beam_outputs = self.simplifier.generate(
                input_ids=summary_ids,
                attention_mask=summary_attention_mask,
                do_sample=True,
                max_length=256,#512
                num_beams=2,
                top_k=80,
                top_p=0.90,
                early_stopping=True,
                num_return_sequences=1
            ).to(self.device)
            # final_outputs = []
            # for beam_output in beam_outputs:
            ## T5:
            sent  = self.simplifier_tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # if sent.lower() != sentence.lower() and sent not in final_outputs:
                # final_outputs.append(sent)
            
            return sent

        pred_sents = []
        for source in batch["source"]:
            pred_sent = generate(source)
            pred_sents.append(pred_sent)

        score = corpus_sari(batch["source"], pred_sents, [batch["targets"]])


        print("Sari score: ", score)

        return 1 - score / 100

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model1 = self.summarizer
        model2 = self.simplifier
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model2.named_parameters() if not any(nd in n for nd in no_decay)],
                                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model2.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model1.named_parameters() if not any(nd in n for nd in no_decay)],
                                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model1.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": self.W
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None,
                       on_tpu=None, using_native_amp=None, using_lbfgs=None):
        optimizer.step(closure=optimizer_closure)

        optimizer.zero_grad()
        self.lr_scheduler.step()
    
    def save_core_model(self):
      tmp = self.args.model_name + 'core'
      store_path = OUTPUT_DIR / tmp
      self.model.save_pretrained(store_path)
      self.simplifier_tokenizer.save_pretrained(store_path)



    def train_dataloader(self):
        train_dataset = TrainDataset(dataset=self.args.dataset,
                                     tokenizer=self.simplifier_tokenizer,
                                     max_len=self.args.max_seq_length,
                                     sample_size=self.args.train_sample_size)

        dataloader = DataLoader(train_dataset,
                                batch_size=self.args.train_batch_size,
                                drop_last=True,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=4)
        t_total = ((len(dataloader.dataset) // (self.args.train_batch_size * max(1, self.args.n_gpu)))
                   // self.args.gradient_accumulation_steps
                   * float(self.args.num_train_epochs)
                   )
        # scheduler = get_linear_schedule_with_warmup(
        #     self.opt, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        # )
        scheduler = get_cosine_schedule_with_warmup(
            self.opt, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = ValDataset(dataset=self.args.dataset,
                                 tokenizer=self.simplifier_tokenizer,
                                 max_len=self.args.max_seq_length,
                                 sample_size=self.args.valid_sample_size)
        return DataLoader(val_dataset,
                          batch_size=self.args.valid_batch_size,
                          num_workers=4)
    @staticmethod
    def add_model_specific_args(parent_parser):
      p = ArgumentParser(parents=[parent_parser],add_help = False)
      p.add_argument('-HiddenSize','--hidden_size',type=int, default = 768)
      p.add_argument('-SeqDim','--seq_dim', type=int, default = 1024)
      p.add_argument('-Weight1', '--w1', type = int, default = 1)
      p.add_argument('-Weight2', '--w2', type = int, default = 0.001)
      p.add_argument('-Lambda', '--lambda_', type = int, default = 0.5)
      p.add_argument('-Simplifier','--sim_model', default='t5-base')
      p.add_argument('-Summarizer','--sum_model', default='t5-base')
      p.add_argument('-TrainBS','--train_batch_size',type=int, default=6)
      p.add_argument('-ValidBS','--valid_batch_size',type=int, default=6)
      p.add_argument('-lr','--learning_rate',type=float, default=0.0003)
      p.add_argument('-MaxSeqLen','--max_seq_length',type=int, default=256)
      p.add_argument('-AdamEps','--adam_epsilon', default=1e-8)
      p.add_argument('-WeightDecay','--weight_decay', default = 0.0001)
      p.add_argument('-WarmupSteps','--warmup_steps',default=5)
      p.add_argument('-NumEpoch','--num_train_epochs',default=7)
      p.add_argument('-CosLoss','--custom_loss', default=True)
      p.add_argument('-GradAccuSteps','--gradient_accumulation_steps', default=1)
      p.add_argument('-GPUs','--n_gpu',default=1) #torch.cuda.device_count()
      p.add_argument('-nbSVS','--nb_sanity_val_steps',default = -1)
      p.add_argument('-TrainSampleSize','--train_sample_size', default=1)
      p.add_argument('-ValidSampleSize','--valid_sample_size', default=1)
      p.add_argument('-device','--device', default = 'cuda')
      #p.add_argument('-NumBeams','--num_beams', default=8)
      return p


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                print(key, metrics[key])
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.args.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


##### build dataset Loader #####
class TrainDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=512, sample_size=1):
        self.sample_size = sample_size
        print("init TrainDataset ...")
        self.source_filepath = get_data_filepath(dataset,'train','complex') 
        self.target_filepath = get_data_filepath(dataset,'train','simple')
        print("source_filepath: ", self.source_filepath)
        print("Initialized dataset done.....")

        self.max_len = max_len
        self.tokenizer = tokenizer

        self._load_data()

    def _load_data(self):
        self.inputs = read_lines(self.source_filepath)
        self.targets = read_lines(self.target_filepath)

    def __len__(self):
        return int(len(self.inputs) * self.sample_size)

    def __getitem__(self, index):
        source = self.inputs[index]


        source = "summarize: " + self.inputs[index]
        target = self.targets[index]

        tokenized_inputs = self.tokenizer(
            [source],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )
        tokenized_targets = self.tokenizer(
            [target],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )
        source_ids = tokenized_inputs["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()

        src_mask = tokenized_inputs["attention_mask"].squeeze()  # might need to squeeze
        target_mask = tokenized_targets["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,
                'sources': self.inputs[index], 'targets': [self.targets[index]],
                'source': source, 'target': target}


class ValDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=512, sample_size=1):
        self.sample_size = sample_size
        
        self.source_filepath = get_data_filepath(dataset, 'valid', 'complex')#
        self.target_filepaths = get_data_filepath(dataset, 'valid', 'simple')


        self.max_len = max_len
        self.tokenizer = tokenizer

        self._build()

    def __len__(self):
        return int(len(self.inputs) * self.sample_size)

    def __getitem__(self, index):
        return {"source": self.inputs[index], "targets": self.targets[index]}

    def _build(self):
        self.inputs = []
        self.targets = []

        for source in yield_lines(self.source_filepath):
            self.inputs.append(source)
        
        for target in yield_lines(self.target_filepaths):
            self.targets.append(target)




def train(args):
    seed_everything(args.seed)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename="checkpoint-{epoch}",
        monitor="val_loss",
        verbose=True,
        mode="min",
        save_top_k=1
    )
    bar_callback = pl.callbacks.TQDMProgressBar(refresh_rate=1)
    metrics_callback = MetricsCallback()
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        # early_stop_callback=False,
        # gradient_clip_val=args.max_grad_norm,
        # checkpoint_callback=checkpoint_callback,
        callbacks=[
            LoggingCallback(),
            #metrics_callback,
            checkpoint_callback, bar_callback],
        logger=TensorBoardLogger(f'{args.output_dir}/logs'),
        num_sanity_val_steps=0,  # skip sanity check to save time for debugging purpose
        # plugins='ddp_sharded',
        #progress_bar_refresh_rate=1,

    )

    print("Initialize model")
    #model = T5FineTuner(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.cuda.set_device(0)
    model = SumSim(args)
    model.args.dataset = args.dataset
    print(model.args.dataset)
    print(args.dataset)
    trainer = pl.Trainer(**train_params)
    # trainer = pl.Trainer.from_argparse_args(
    #     args,
    #     gpus = args.n_gpu,
    #     max_epochs = args.num_train_epochs,
    #     accumulate_grad_batches = args.gradient_accumulation_steps,
    #     callbacks = [LoggingCallback(), checkpoint_callback, bar_callback],
    #     num_sanity_val_steps = args.nb_sanity_val_steps,
    # )

    print(" Training model")
    trainer.fit(model)

    print("training finished")
