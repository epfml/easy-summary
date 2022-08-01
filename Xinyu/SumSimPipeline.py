# -- fix path --
import optparse
from pathlib import Path
import sys
from evaluate import evaluate
sys.path.append(str(Path(__file__).resolve().parent))
# -- end fix path --

import torch
from summarizer import Summarizer
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from contextlib import contextmanager
import spacy
from preprocessor import Preprocessor, yield_lines, get_data_filepath,write_lines,count_line,read_lines,generate_hash, WIKI_DOC,EXP_DIR
import numpy as np
from easse.cli import evaluate_system_output
from Ts_T5 import T5FineTuner
from easse.sari import corpus_sari
import time


#### gennerate ratio
ps = [0.1, 0.3, 0.5, 0.7, 0.9]


@contextmanager
def log_stdout(filepath, mute_stdout=False):
    '''Context manager to write both to stdout and to a file'''

    class MultipleStreamsWriter:
        def __init__(self, streams):
            self.streams = streams

        def write(self, message):
            for stream in self.streams:
                stream.write(message)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    save_stdout = sys.stdout
    log_file = open(filepath, 'w')
    if mute_stdout:
        sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
    else:
        sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
    try:
        yield
    finally:
        sys.stdout = save_stdout
        log_file.close()


##### 
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
max_len = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### specify the model
model_dirname = 'exp_wikiparagh_10_epoch'
checkpoint_path = 'checkpoint-epoch=3.ckpt'

### load the model
Model = T5FineTuner.load_from_checkpoint(EXP_DIR / model_dirname / checkpoint_path).to(device)
model =  Model.model.to(device)
tokenizer = Model.tokenizer

def generate(sentence, perprocessor):
    sentence = perprocessor.encode_sentence(sentence)
    text = "simplify: " + sentence
    encoding = tokenizer(text, max_length = max_len,
                         padding='max_length', 
                         truncation=True, 
                         return_tensors='pt')
    inputs_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    print(inputs_ids)
    if inputs_ids == None:
        return None
    beam_outputs = model.generate(
        inputs_ids=inputs_ids,
        attention_mask=attention_mask,
        do_sample = False,
        max_length = max_len,
        num_beams = 10,
        top_k = 130,
        top_p = 0.97,
        early_stopping = True,
        num_return_sequences = 1,
    )

    pred_sent = tokenizer.decode(beam_outputs[0], skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
    return pred_sent

def simplify_file(complex_filepath, output_filepath, features_kwargs, model_dirname = None, post_processing=True):
    preprocessor = Preprocessor(features_kwargs)
    total_lines = count_line(complex_filepath)

    output_file = Path(output_filepath).open('w')

    for n_line, complex_sent in enumerate(yield_lines(complex_filepath), start=1):
        output_sents = generate(complex_sent, preprocessor)
        print(f'{n_line}/{total_lines}', ' : ', output_sents)
        if output_sents:
            output_file.write(output_sents + '\n')
        else:
            output_file.write("\n")
    
    output_file.close()

    if post_processing: 
        post_process(output_filepath)

def post_process(file_path):
    lines = []
    for line in yield_lines(file_path):
        lines.append(line.replace("''",'"'))
    write_lines(lines, file_path)

def evaluate_on_WIKIDOC(features_kwargs, ratio, phase, model_dirname = None):
    dataset = WIKI_DOC
    model_dir = EXP_DIR / model_dirname
    output_dir = model_dir / 'outputs'

    output_dir.mkdir(parents = True, exist_ok = True)

    features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir / f'score_{features_hash}_{dataset}_{phase}.log.txt'
    complex_filepath = get_data_filepath(dataset, phase, 'complex_summary_'+str(ratio))

    if not output_score_filepath.exists() or count_line(output_score_filepath)==0:
        start_time = time.time()
        complex_filepath = get_data_filepath(dataset, phase, 'complex_summary_'+str(ratio))
        pred_filepath = output_dir / f'{features_hash}_{complex_filepath.stem}.txt'
        ref_filepaths = get_data_filepath(dataset, phase, 'simple')

        if pred_filepath.exists() and count_line(pred_filepath)==count_line(complex_filepath):
            print("File is already processed.")
        else:
            simplify_file(complex_filepath, pred_filepath, features_kwargs, model_dirname)

        print("Evaluate: ", pred_filepath)

        with log_stdout(output_score_filepath):
            score  = evaluate(complex_filepath, pred_filepath, ref_filepaths)
            
            if "WordRatioFeature" in features_kwargs:
                print("W:", features_kwargs["WordRatioFeature"]["target_ratio"], "\t", end="")
            if "CharRatioFeature" in features_kwargs:
                print("C:", features_kwargs["CharRatioFeature"]["target_ratio"], "\t", end="")
            if "LevenshteinRatioFeature" in features_kwargs:
                print("L:", features_kwargs["LevenshteinRatioFeature"]["target_ratio"], "\t", end="")
            if "WordRankRatioFeature" in features_kwargs:
                print("WR:", features_kwargs["WordRankRatioFeature"]["target_ratio"], "\t", end="")
            if "DependencyTreeDepthRatioFeature" in features_kwargs:
                print("DTD:", features_kwargs["DependencyTreeDepthRatioFeature"]["target_ratio"], "\t", end="")
            print("{:.2f} \t ".format(score))

            print("Execution time: --- %s seconds ---" % (time.time() - start_time))
    else:
        print("Already exists: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))


# Specify the token features to use
features_kwargs = {
    # 'WordRatioFeature': {'target_ratio': 1.05},
    'CharRatioFeature': {'target_ratio': 0.96},
    'LevenshteinRatioFeature': {'target_ratio': 0.68},
    'WordRankRatioFeature': {'target_ratio': 0.80},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.75}
}

evaluate_on_WIKIDOC(features_kwargs, 0.1, 'valid', model_dirname)


