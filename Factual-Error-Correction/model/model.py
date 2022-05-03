#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from datasets import load_metric
import numpy as np

from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
# from apex import amp
from tqdm import tqdm


MODEL_CLASSES = {
    'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
    }

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BartConfig, BertConfig, XLNetConfig, XLMConfig, 
                                                                                # RobertaConfig, DistilBertConfig)), ())
ALL_MODELS = ''

class Model:
    """Enhanced Sequential Inference Model (ESIM) for natural language inference.
    """
    def __init__(self, args):
        """Model initialization.
        """
        self.args = args
        self.logger = args.logger

        self._build_model()
        self.model.to(args.device)

        self.optimizer = self._get_optimizer(self._group_parameters(self.model))
        self.scheduler = self._get_scheduler(self.optimizer)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            self.logger.info("- Let's use {} GPUs !".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        else:
            self.logger.info("- Train the model on single GPU :/")

        # tensorboard
        if args.write_summary:
            self.logger.info("- Let's use tensorboard on local rank {} device :)".format(args.local_rank))
            self.writer = SummaryWriter(self.args.summary_path)

    def _build_model(self):
        """Build model.
        """
        model_type = self.args.model_type.lower()
        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[model_type]
        if self.args.use_pretrained:
            self.load_weights(self.args.checkpoint)
        else:
            self._load_from_library(self.args)

    def _load_from_library(self, args):
        """Initialize ESIM model paramerters.
        """
        self.logger.info("- Downloading model...")
        config = self.config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                                   num_labels=args.num_labels,
                                                   finetuning_task=args.task_name)
        self.tokenizer = self.tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name
                                                              else args.model_name_or_path,
                                                              do_lower_case=args.do_lower_case)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      from_tf=bool('.ckpt' in args.model_name_or_path),
                                                      config=config)

    def _group_parameters(self, model):
        """Specify which parameters do weight decay and which not.
        """
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params':
                [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay},
            {'params':
                [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0}
            ]
        return optimizer_grouped_parameters

    def _get_optimizer(self, optimizer_grouped_parameters):
        """Get optimizer for model training.
        """
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon)
        return optimizer

    def _get_scheduler(self, optimizer):
        """Get scheduler for adjusting learning rate.
        """
        if self.args.scheduler == 'warmup':
            train_steps = int(287227 / (self.args.per_gpu_train_batch_size * self.args.n_gpu) * self.args.num_epochs)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                             num_warmup_steps=self.args.warmup_steps,
                                             num_training_steps=train_steps,
                                             last_epoch=self.args.num_epochs)
        elif self.args.scheduler == 'exponential':
            scheduler = ExponentialLR(optimizer, 0.95)
        return scheduler

    def load_weights(self, checkpoint):
        """Load pre-trained model weights.
        """
        self.logger.info("- Load pre-trained model from: {}".format(checkpoint))
        self.model = self.model_class.from_pretrained(checkpoint)
        self.tokenizer = self.tokenizer_class.from_pretrained(checkpoint,
                                                              do_lower_case=self.args.do_lower_case)
        self.model.to(self.args.device)
        return self.model, self.tokenizer

    def load_transformer(self, weights):
        """Load pre-trained model weights.
        """
        # Take care of distributed/parallel training
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_type = self.args.model_type.lower()
        if model_type == 'bart':
            model_to_save.bart.load_state_dict(weights.state_dict())
        elif model_type == 'bert':
            model_to_save.bert.load_state_dict(weights.state_dict())
        elif model_type == 'xlnet':
            model_to_save.transformer.load_state_dict(weights.state_dict())
        elif model_type == 'distilbert':
            model_to_save.distilbert.load_state_dict(weights.state_dict())
        else:
            raise Exception("Unknow model type!")

    def save_model(self, output_path, epoch=None):
        """Save model's weights.
        """
        output_path = os.path.join(output_path, 'epoch_'+str(epoch))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # Take care of distributed/parallel training
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        torch.save(self.args, os.path.join(output_path, 'training_args.bin'))
        self.logger.info("- model, tokenzier and args is saved at: {}".format(output_path))

    def loss_batch(self, inputs, optimizer=None, step=None):
        """Calculate loss on a single batch of data.
        """
        if optimizer:
            assert step is not None
        # outputs = self.model(**inputs)
        outputs = self.model(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'], labels=inputs['labels'])
        loss, logits = outputs[0], outputs[1]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if optimizer is not None:
            if self.args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if self.args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                                   self.args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.args.max_grad_norm)
                optimizer.step()  # update model parameters
                optimizer.zero_grad()  # clean all gradients

        return loss.item(), logits.detach()

    def train_epoch(self, train_dataloader, optimizer, epoch):
        """Train the model for one single epoch.
        """
        self.model.train()  # set the model to training mode
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        train_loss = 0.0
        for i, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(self.args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2]}

            batch_loss, _ = self.loss_batch(inputs,
                                            optimizer=optimizer,
                                            step=i)
            train_loss += batch_loss

            if self.writer:
                self.writer.add_scalar('batch_loss', batch_loss, epoch*len(train_dataloader) + i + 1)

        # compute the average loss (batch loss)
        epoch_loss = train_loss / len(train_dataloader)

        # update scheduler
        self.scheduler.step()

        return epoch_loss

    def evaluate(self, eval_dataloader, epoch=None):
        """Evaluate the model.
        """
        self.model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            eval_loss = 0.0
            for i, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.args.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[2]}

                batch_loss, _ = self.loss_batch(inputs, optimizer=None)
                eval_loss += batch_loss
                if i == 0:
                    print('Starting to generate summary...')
                    summary_ids = self.model.generate(inputs['input_ids'], 
                                                    max_length=128, 
                                                    repetition_penalty=2.5, 
                                                    no_repeat_ngram_size=3,
                                                    early_stopping=True,
                                                    num_beams=5,
                                                    )
                    decoded_summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    decoded_labels = self.tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    decoded_input = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    if epoch is not None:
                        name = 'epoch_' + str(epoch) + '_summaries.txt'
                    else:
                        name = 'summaries.txt'
                    summuries_path = os.path.join(self.args.output_dir, name)
                    with open(summuries_path, 'w') as f:
                        for ind in range(len(decoded_summaries)):
                            f.write('(' + str(ind) + ')\n')
                            f.write(str(ind) + ') Generated:\n' + decoded_summaries[ind] + '\n')
                            f.write(str(ind) + ') Ground true:\n' + decoded_labels[ind] + '\n')
                            f.write(str(ind) + ') Input summary:\n' + decoded_input[ind].split('</s>')[0] + '\n')

            avg_loss = eval_loss / len(eval_dataloader)
        return avg_loss

    def fit(self, train_dataloader, eval_dataloader):
        """Model training and evaluation.
        """
        num_epochs = self.args.num_epochs

        for epoch in range(num_epochs):
            self.logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

            # training
            train_loss = self.train_epoch(train_dataloader, self.optimizer, epoch)
            self.logger.info("Traing Loss: {}".format(train_loss))

            # evaluation, only on the master node
            eval_loss = self.evaluate(eval_dataloader, epoch)
            self.logger.info("Evaluation:")
            self.logger.info("- loss: {}".format(eval_loss))

            # monitor loss and accuracy
            if self.writer:
                self.writer.add_scalar('epoch_loss', train_loss, epoch)
                self.writer.add_scalar('eval_loss', eval_loss, epoch)
                self.writer.add_scalar('lr', self.scheduler.get_lr()[0], epoch)

            # save the model
            self.logger.info("Saving model")
            self.save_model(self.args.model_dir, epoch)

    def compute_metrics(self, res_dict, preds, labels):
        metric = load_metric("rouge")
        summary_ids = self.model.generate(preds, 
                                        max_length=128, 
                                        repetition_penalty=2.5, 
                                        no_repeat_ngram_size=3,
                                        early_stopping=True,
                                        num_beams=5,
                                        )
        decoded_summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Rouge expects a newline after each sentence
        # decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_summaries]
        # decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        decoded_preds = [".\n".join(pred.split('.')) for pred in decoded_summaries]
        decoded_labels = [".\n".join(label.split('.')) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        for k, v in result.items():
            res_dict[k].extend(round(v, 4))
        return res_dict
    
    def test(self, eval_dataloader, epoch=None):
        """Evaluate the model.
        """
        self.model.eval()  # set the model to evaluation mode
        if epoch is not None:
            name = 'test_epoch_' + str(epoch) + '_summaries.txt'
        else:
            name = 'test_summaries.txt'
        summuries_path = os.path.join(self.args.output_dir, name)
        res_dict = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}
        with open(summuries_path, 'w') as f:
            with torch.no_grad():
                for i, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
                    batch = tuple(t.to(self.args.device) for t in batch)
                    inputs = {'input_ids':    batch[0],
                            'attention_mask': batch[1],
                            'labels':         batch[2]}
                    # if i >= 5:
                    res_dict = self.compute_metrics(res_dict, inputs['input_ids'], inputs['labels'])
                    results = {k: np.mean(v) for k, v in res_dict.items()}
                    print(results)
                    break
                    # else:
                    #     print('Starting to generate summary...')
                    #     summary_ids = self.model.generate(inputs['input_ids'], 
                    #                                         max_length=128, 
                    #                                         repetition_penalty=2.5, 
                    #                                         no_repeat_ngram_size=3,
                    #                                         early_stopping=True,
                    #                                         num_beams=5,
                    #                                         )
                    #     decoded_summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    #     decoded_labels = self.tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    #     decoded_input = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    #     res_dict = self.compute_metrics(res_dict, inputs['input_ids'], inputs['labels'])
                    #     for ind in range(len(decoded_summaries)):
                    #         f.write('(' + str(ind) + ')\n')
                    #         f.write(str(ind) + ') Generated:\n' + decoded_summaries[ind] + '\n')
                    #         f.write(str(ind) + ') Ground true:\n' + decoded_labels[ind] + '\n')
                    #         f.write(str(ind) + ') Input summary:\n' + decoded_input[ind].split('</s>')[0] + '\n')
