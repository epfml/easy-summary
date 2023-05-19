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
import Levenshtein
import argparse
from argparse import ArgumentParser
import os
import logging
import random
import nltk
from preprocessor import  get_data_filepath
from summarizer import Summarizer

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
    get_cosine_schedule_with_warmup
)
from Ts_T5 import T5FineTuner
from Ts_BART import BartFineTuner

#BERT_Sum = Summarizer(model='distilbert-base-uncased')

class MetricsCallback(pl.Callback):
  def __init__(self):
    super().__init__()
    self.metrics = []
  
  def on_validation_end(self, trainer, pl_module):
      self.metrics.append(trainer.callback_metrics)

### Special tokens
def char_ratio(complex_sentence, simple_sentence):
    return round(safe_division(len(simple_sentence), len(complex_sentence)))


def LevSim(complex_sentence, simple_sentence):
    return round(Levenshtein.ratio(complex_sentence, simple_sentence))

def word_rank_ratio(complex_sentence, simple_sentence):
    def get_rank(word):
        rank = get_word2rank().get(word, len(get_word2rank()))
        return np.log(1+rank)
    
    def get_lexical_complexity_score(sentence):
        words = tokenize(remove_stopwords(remove_punctuation(sentence)))
        words = [word for word in words if word in get_word2rank()]
        if len(words)==0:
            return np.log(1+len(get_word2rank()))
        return np.quantile([get_rank(word) for word in words], 0.75)
    
    return round(min(safe_division(
        get_lexical_complexity_score(simple_sentence),
        get_lexical_complexity_score(complex_sentence)
    ), 2))
### Speicial tokens end

class SumSim(pl.LightningModule):
    def __init__(self,args):
        super(SumSim, self).__init__()
        self.args = args
        self.save_hyperparameters()
        # Load pre-trained model and tokenizer
        #self.summarizer = BartModel.from_pretrained("facebook/bart-large-cnn")
        self.summarizer = BartForConditionalGeneration.from_pretrained(self.args.sum_model)
        self.summarizer_tokenizer = BartTokenizerFast.from_pretrained(self.args.sum_model)
        self.summarizer = self.summarizer.to(self.args.device)


        #self.simplifier = BartForConditionalGeneration.from_pretrained(self.args.sum_model)
        self.simplifier = BartFineTuner.load_from_checkpoint("experiments/exp_WikiLarge_BARTSingle/checkpoint-epoch=2.ckpt")
        self.simplifier = self.simplifier.model.to(self.args.device)
        self.simplifier_tokenizer = BartTokenizerFast.from_pretrained(self.args.sim_model)


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
        labels[labels[:,:] == self.simplifier_tokenizer.pad_token_id] = -100
        # zero the gradient buffers of all parameters
        self.opt.zero_grad()
        #print(source, len(source))
        ## summarizer stage
        inputs = self.summarizer_tokenizer(
            source,
            max_length = 512,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt'
        ).to(self.args.device)

        
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
            num_beams = 5,
            min_length = 10,
            max_length = 256,
            top_k=120,top_p=0.95,
        ).to(self.args.device)

        ### Original loss
        # summary_attention_mask = torch.ones(summary_ids.shape).to(self.args.device)
        # summary_attention_mask[summary_ids[:,:]==self.summarizer_tokenizer.pad_token_id]=0
        
        # sim_outputs  = self(
        #     input_ids = summary_ids,
        #     attention_mask = summary_attention_mask,
        #     labels = labels,
        #     decoder_attention_mask = batch['target_mask']
        # )

        ### modified loss
        padded_summary_ids = torch.zeros((summary_ids.shape[0], 256), dtype=torch.long).fill_(self.simplifier_tokenizer.pad_token_id).to(self.args.device)
        
        for i, summary_id in enumerate(summary_ids):
            padded_summary_ids[i, :summary_id.shape[0]] = summary_id

        summary_attention_mask = torch.ones(padded_summary_ids.shape).to(self.args.device)
        summary_attention_mask[padded_summary_ids[:,:]==self.summarizer_tokenizer.pad_token_id]=0

        
        
        # forward pass
        sim_outputs  = self(
            input_ids = padded_summary_ids,
            attention_mask = summary_attention_mask,
            labels = labels,
            decoder_attention_mask = batch['target_mask']
        )
        #H2 = sim_outputs.encoder_last_hidden_state
        
        ## CosSim
        # Rep1 = torch.matmul(H1, self.W)
        # Rep2 = torch.matmul(H2, self.W)
        # Rep1 = self.relu(Rep1)
        # Rep2 = self.relu(Rep2)
        # CosSim = nn.CosineSimilarity(dim=2, eps=1e-6)
        # sim_score = CosSim(Rep1, Rep2)

        ## KL loss
        # H1 = torch.transpose((torch.transpose(H1, 1,2)@self.Q), 1,2)
        # H2 = torch.transpose((torch.transpose(H2, 1,2)@self.Q), 1,2)
        # Rep1 = torch.matmul(H1, self.W)
        # Rep2 = torch.matmul(H2, self.W)
        # Rep1 = Rep1.squeeze(dim=2)
        # Rep2 = Rep2.squeeze(dim=2)
        # LogSoftMax = nn.LogSoftmax(dim=1)
        # Rep1 = LogSoftMax(Rep1)
        # Rep2 = LogSoftMax(Rep2)

        if self.args.custom_loss:
            '''
            Custom Loss:
            Loss = oiginal_loss + lambda**2 * complexity_score

            - ratio: control the ratio of sentences we want to compute complexity for training.
            - lambda: control the weight of the complexity loss.
            '''
            loss = sim_outputs.loss * self.args.w1
            #loss += sum_outputs.loss * self.args.w2
            ### KL ###
            #loss += (self.args.lambda_ * self.kl_loss(Rep1, Rep2))
            
            ### CosSim ###
            #loss += (-self.args.lambda_ * (sim_score.mean(dim=1).mean(dim=0)))




            
            # self.manual_backward(loss)
            # self.opt.step()
            
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
            
            text = sentence
            # summarize the document
            inputs = self.summarizer_tokenizer(
            [text],
            max_length = 512,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt'
        )
            # generate summary
            summary_ids = self.summarizer.generate(
                inputs['input_ids'].to(self.args.device),
                num_beams = 5,
                min_length = 10,
                max_length = 256,
                top_k = 120, top_p = 0.95,
            ).to(self.args.device)

            summary_attention_mask = torch.ones(summary_ids.shape).to(self.args.device)
            summary_attention_mask[summary_ids==self.summarizer_tokenizer.pad_token_id]=0


            

            # set top_k = 130 and set top_p = 0.95 and num_return_sequences = 1
            beam_outputs = self.simplifier.generate(
                input_ids=summary_ids,
                attention_mask=summary_attention_mask,
                do_sample=True,
                max_length=self.args.max_seq_length,
                num_beams=5,
                top_k=130,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=1
            ).to(self.device)
            # final_outputs = []
            # for beam_output in beam_outputs:
            
            ## Bart:
            sent = self.simplifier_tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
           
            # if sent.lower() != sentence.lower() and sent not in final_outputs:
                # final_outputs.append(sent)
            
            return sent

        pred_sents = []
        for source in batch["source"]:
            pred_sent = generate(source)
            pred_sents.append(pred_sent)

        ### WIKI-large ###
        score = corpus_sari(batch["source"], pred_sents, [batch["targets"]])

        ### turkcorpuse ###
        #score = corpus_sari(batch["source"], pred_sents, batch["targets"])

        print("Sari score: ", score)

        return 1 - score / 100

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model1 = self.summarizer
        model2 = self.simplifier
        #no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n,p in model1.named_parameters()]
            },
            {
                "params": [p for n,p in model2.named_parameters()]
            },
            # {
            #     "params": self.W
            # },
            # {
            #     "params": self.Q
            # },
            # {
            #     "params": self.W2
            # }
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
      # facebook/bart-base
      p.add_argument('-HiddenSize','--hidden_size',type=int, default = 1)
      p.add_argument('-SeqDim','--seq_dim', type=int, default = 512)
      p.add_argument('-Weight1', '--w1', type = int, default = 1)
      p.add_argument('-Weight2', '--w2', type = int, default = 1)
      p.add_argument('-Lambda', '--lambda_', type = int, default = 11)
      # BRIO: Yale-LILY/brio-cnndm-uncased ainize/bart-base-cnn
      p.add_argument('-Summarizer','--sum_model', default='ainize/bart-base-cnn')
      p.add_argument('-Simplifier','--sim_model', default='facebook/bart-base')
      p.add_argument('-TrainBS','--train_batch_size',type=int, default=6)
      p.add_argument('-ValidBS','--valid_batch_size',type=int, default=6)
      p.add_argument('-lr','--learning_rate',type=float, default=5e-5)
      p.add_argument('-MaxSeqLen','--max_seq_length',type=int, default=256)
      p.add_argument('-AdamEps','--adam_epsilon', default=1e-8)
      p.add_argument('-WeightDecay','--weight_decay', default = 0.0001)
      p.add_argument('-WarmupSteps','--warmup_steps',default=5)
      p.add_argument('-NumEpoch','--num_train_epochs',default=7)
      p.add_argument('-CosLoss','--custom_loss', default=False)
      p.add_argument('-GradAccuSteps','--gradient_accumulation_steps', default=1)
      p.add_argument('-GPUs','--n_gpu',default=torch.cuda.device_count())
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
    def __init__(self, dataset, tokenizer, max_len=256, sample_size=1):
        self.sample_size = sample_size
        print("init TrainDataset ...")
        self.source_filepath = get_data_filepath(dataset,'train','complex')
        self.target_filepath = get_data_filepath(dataset,'train','simple')
        print("Initialized dataset done.....")
        # preprocessor = load_preprocessor()
        # self.source_filepath = preprocessor.get_preprocessed_filepath(dataset, 'train', 'complex')
        # self.target_filepath = preprocessor.get_preprocessed_filepath(dataset, 'train', 'simple')

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
        # source = "summarize: " + self.inputs[index]
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
    def __init__(self, dataset, tokenizer, max_len=256, sample_size=1):
        self.sample_size = sample_size
        ### WIKI-large dataset ###
        self.source_filepath = get_data_filepath(dataset, 'valid', 'complex')
        self.target_filepaths = get_data_filepath(dataset, 'valid', 'simple')

        ### turkcorpus dataset ###
        # self.source_filepath = get_data_filepath(TURKCORPUS_DATASET, 'valid', 'complex')
        # self.target_filepaths = [get_data_filepath(TURKCORPUS_DATASET, 'valid', 'simple.turk',i)for i in range(8)]
        # if dataset == NEWSELA_DATASET:
        #     self.target_filepaths = [get_data_filepath(dataset, 'valid', 'simple')]

        # else:  # TURKCORPUS_DATASET as default
        #     self.target_filepaths = [get_data_filepath(TURKCORPUS_DATASET, 'valid', 'simple.turk', i) for i in range(8)]

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

        ### turkcorpus dataset ###
        # self.targets = [ [] for _ in range(count_line(self.target_filepaths[0]))]
        # for file_path in self.target_filepaths:
        #     for i, target in enumerate(yield_lines(file_path)):
        #         self.targets[i].append(target)

        # self.targets = [[] for _ in range(count_line(self.target_filepaths[0]))]
        # for filepath in self.target_filepaths:
        #     for idx, line in enumerate(yield_lines(filepath)):
        #         self.targets[idx].append(line)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SumSim(args)
    #model = SumSim.load_from_checkpoint("Xinyu/experiments/exp_DWikiMatch_BART/checkpoint-epoch=4.ckpt")
    model.args.dataset = args.dataset
    print(model.args.dataset)
    #model = T5FineTuner(**train_args)
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

    # print("Saving model")
    # model.model.save_pretrained(args.output_dir)

    # print("Saved model")