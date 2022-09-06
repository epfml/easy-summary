'''
Evaluate the SARI score and Other metric of TurkCorpus test dataset
'''

from pathlib import Path
import sys
# sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from easse.cli import evaluate_system_output
from Ts_T5 import T5FineTuner
#from easse.report import get_all_scores
from contextlib import contextmanager
import json
from preprocessor import Preprocessor
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from preprocessor import get_data_filepath, EXP_DIR, TURKCORPUS_DATASET, REPO_DIR, WIKI_DOC, D_WIKI,WIKI_DOC_CLEAN
from preprocessor import write_lines, yield_lines, count_line, read_lines, generate_hash
from easse.sari import corpus_sari
import time
from utils.D_SARI import D_SARIsent
from googletrans import Translator
from Bart2 import SumSim
#from T5_2 import SumSim
#from T5_baseline_finetuned import T5BaseLineFineTuned
#from Bart_baseline_finetuned import BartBaseLineFineTuned

from keybert import KeyBERT
from transformers.pipelines import pipeline

hf_model = pipeline("feature-extraction", model="distilbert-base-cased")

kw_model = KeyBERT(model = 'all-mpnet-base-v2')

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

# set random seed universal
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
model_dir = None
_model_dirname = None
max_len = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# specify the model_name and checkpoint_name
#model_dirname = 'exp_DWiki_T5'
model_dirname = 'exp_DWiki_BART'
#model_dirname = 'exp_WikiDocSmall_BART_CosSim+SumSimLoss'
checkpoint_path = 'checkpoint-epoch=2.ckpt'

# load the model
# Model = T5BaseLineFineTuned.load_from_checkpoint(EXP_DIR / model_dirname / checkpoint_path).to(device)
# model = Model.model.to(device)
# tokenizer = Model.tokenizer

Model = SumSim.load_from_checkpoint(EXP_DIR /  model_dirname / checkpoint_path).to(device)
summarizer = Model.summarizer.to(device)
simplifier = Model.simplifier.to(device)
summarizer_tokenizer = Model.summarizer_tokenizer
simplifier_tokenizer = Model.simplifier_tokenizer
# translator = Translator()

def generate_single(sentence, preprocessor = None):
    
    ### add keyword
    # key_words = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 1), stop_words=None)
    # for i in range(min(3, len(key_words))):
    #     sentence = key_words[i][0] + "_" + str(round(key_words[i][1],2)) + ' ' +sentence
    
    text = "simplify: " + sentence
    text = sentence
    encoding = tokenizer(text, max_length=256,
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=False,
        max_length=max_len,
        num_beams=10,
        top_k=130,
        top_p=0.97,
        early_stopping=True,
        num_return_sequences=1,
    )
    sent = tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return sent


def generate(sentence, preprocessor=None):
    '''
    Apply model to generate prediction
    '''
    # key_words = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 1), stop_words=None)
    # for i in range(min(3, len(key_words))):
    #     sentence = key_words[i][0] + "_" + str(round(key_words[i][1],2)) + ' ' +sentence
    
    # For T5
    #sentence = 'summarize ' + sentence

    encoding = summarizer_tokenizer(
        [sentence],
        max_length = 256,
        truncation = True,
        padding = 'max_length',
        return_tensors = 'pt',
    )
    
    summary_ids = summarizer.generate(
        encoding['input_ids'].to(device),
        num_beams = 10,
        min_length = 10,
        max_length = 256,
    ).to(device)
    
    # For T5
    for i, summary_id in enumerate(summary_ids):
        add_tokens = torch.tensor([18356, 10]).to(device)
        summary_ids[i,:] = torch.cat((summary_id, add_tokens), dim=0)[:-2]
    
    summary_atten_mask = torch.ones(summary_ids.shape).to(device)
    summary_atten_mask[summary_ids[:,:] == summarizer_tokenizer.pad_token_id] = 0
    
    beam_outputs = simplifier.generate(
        input_ids = summary_ids,
        attention_mask = summary_atten_mask,
        do_sample = True,
        max_length = 256,
        num_beams =16, #16
        top_k = 120,  #120
        top_p = 0.95, #0.95
        early_stopping = True,
        num_return_sequences = 1,
    )
    
    sent = simplifier_tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return sent
    
    '''
    sentence = preprocessor.encode_sentence(sentence)
    text = "simplify: " + sentence
    encoding = tokenizer(text, max_length=max_len,
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)

    # set top_k = 130 and set top_p = 0.97 and num_return_sequences = 1
    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=False,
        max_length=max_len,
        num_beams=10,
        top_k=130,
        top_p=0.97,
        early_stopping=True,
        num_return_sequences=1
    )
    pred_sent = tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return pred_sent
    # final_outputs = []
    # for beam_output in beam_outputs:
    #     sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     if sent.lower() != sentence.lower() and sent not in final_outputs:
    #         final_outputs.append(sent)

    # return final_outputs
    '''
    
    


    

def evaluate(orig_filepath, sys_filepath, ref_filepaths):
    orig_sents = read_lines(orig_filepath)
    # NOTE: change the refs_sents if several references are used
    refs_sents = [read_lines(ref_filepaths)]
    #refs_sents = [read_lines(filepath) for filepath in ref_filepaths]

    # print(sys_filepath.name, f"Sari score:: ({sari_score})", )
    # print(len(orig_sents), len(read_lines(sys_filepath)))
    return corpus_sari(orig_sents, read_lines(sys_filepath), refs_sents)


# def evaluate_all_metrics(orig_filepath, sys_filepath, ref_filepaths):
#     orig_sents = read_lines(orig_filepath)
#     refs_sents = [read_lines(filepath) for filepath in ref_filepaths]
#     # return get_all_scores(orig_sents, read_lines(sys_filepath), refs_sents, lowercase=True)
#     # return get_all_scores(orig_sents, read_lines(sys_filepath), refs_sents, lowercase=False)
#     return get_all_scores(orig_sents, read_lines(sys_filepath), refs_sents, lowercase=True)


def evaluate_on(dataset, features_kwargs, phase, model_dirname=None, checkpoint_path=None):

    global model, tokenizer, device, model_dir, _model_dirname, max_len
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = EXP_DIR / model_dirname / checkpoint_path
    Model = T5FineTuner.load_from_checkpoint(model_path).to(device)
    model = Model.model
    tokenizer = Model.tokenizer
    model_dir = EXP_DIR / model_dirname

    # load_model(model_dirname)
    preprocessor = Preprocessor(features_kwargs)
    output_dir = model_dir / "outputs"
    # output_dir = REPO_DIR / f"outputs/{_model_dirname}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir: ", output_dir)
    features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir / f"score_{features_hash}_{dataset}_{phase}_log.txt"
    if not output_score_filepath.exists() or count_line(output_score_filepath) == 0:
        # log_params(output_dir / f"{features_hash}_features_kwargs.json", features_kwargs)
        start_time = time.time()
        complex_filepath = get_data_filepath(dataset, phase, 'complex')
        pred_filepath = output_dir / f'{features_hash}_{complex_filepath.stem}.txt'
        # ref_filepaths = [get_data_filepath(dataset, phase, 'simple.turk', i) for i in range(8)]
        ref_filepath = get_data_filepath(dataset, phase, 'simple')
        print(pred_filepath)
        if pred_filepath.exists() and count_line(pred_filepath) == count_line(complex_filepath):
            print("File is already processed.")
        else:
            simplify_file(complex_filepath, pred_filepath, preprocessor)

        # print("Evaluate: ", pred_filepath)
        with log_stdout(output_score_filepath):
            # print("features_kwargs: ", features_kwargs)

            # refs = [[line] for line in yield_lines(ref_filepath)]
            # score = corpus_sari(read_lines(complex_filepath), read_lines(pred_filepath), refs)
            # print(len(read_lines(complex_filepath)), len(read_lines(pred_filepath)) )
            # print([len(s) for s in refs])

            # scores = get_all_scores(read_lines(complex_filepath), read_lines(pred_filepath), refs)
            score = evaluate(complex_filepath, pred_filepath, [ref_filepath])
            # scores = evaluate_all_metrics(complex_filepath, pred_filepath, ref_filepaths)
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
        print("Already exist: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))

def back_translation(text):
    X = translator.translate(text, dest = 'de')
    return translator.translate(X.text, dest = 'en').text


def simplify_file(complex_filepath, output_filepath, features_kwargs=None, model_dirname=None, post_processing=True):
    '''
    Obtain the simplified sentences (predictions) from the original complex sentences.
    '''
    # load_model(model_dirname)
    # global model, tokenizer, device, model_dir, _model_dirname, max_len, Model
    #preprocessor = Preprocessor(features_kwargs)
    
    total_lines = count_line(complex_filepath)
    print(complex_filepath)
    print(complex_filepath.stem)

    output_file = Path(output_filepath).open("w")

    for n_line, complex_sent in enumerate(yield_lines(complex_filepath), start=1):
        #output_sents = generate_single(complex_sent, preprocessor = None)
        output_sents = generate(complex_sent, preprocessor=None)
        
        # apply back translation
        #output_sents = back_translation(output_sents)

        print(f"{n_line+1}/{total_lines}", " : ", output_sents)
        if output_sents:
            # output_file.write(output_sents[0] + "\n")
            output_file.write(output_sents + "\n")
        else:
            output_file.write("\n")
    output_file.close()
    
    if post_processing: post_process(output_filepath)

def post_process(filepath):
    lines = []
    for line in yield_lines(filepath):
        lines.append(line.replace("''", '"'))
    write_lines(lines, filepath)
    
def evaluate_on_TurkCorpus(features_kwargs, phase, model_dirname=None):
    '''
    Evaluate on the TurkCorpus dataset (test stage)
    '''
    dataset = TURKCORPUS_DATASET
    # global model, tokenizer, device, model_dir, _model_dirname, max_len, Model
    # model_path = EXP_DIR / model_dirname / checkpoint_path
    # Model = T5FineTuner.load_from_checkpoint(model_path).to(device)
    model_dir = EXP_DIR / model_dirname
    output_dir = model_dir / "outputs"
    
    # output_dir = REPO_DIR / f"outputs/{_model_dirname}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir: ", output_dir)

    features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir / f"score_{features_hash}_{dataset}_{phase}_log.txt"
    complex_filepath = get_data_filepath(dataset, phase, 'complex')
    # pred_filepath = output_dir / f'{features_hash}_{complex_filepath.stem}.txt'
    # print(evaluate(complex_filepath, pred_filepath, [get_data_filepath(dataset, phase, 'simple.turk', i) for i in range(1)]))

    if not output_score_filepath.exists() or count_line(output_score_filepath) == 0:
        # log_params(output_dir / f"{features_hash}_features_kwargs.json", features_kwargs)
        start_time = time.time()
        complex_filepath = get_data_filepath(dataset, phase, 'complex')
        pred_filepath = output_dir / f'{features_hash}_{complex_filepath.stem}.txt'
        ref_filepaths = [get_data_filepath(dataset, phase, 'simple.turk', i) for i in range(8)]
        print("Predict file_path: ", pred_filepath)
        if pred_filepath.exists() and count_line(pred_filepath) == count_line(complex_filepath):
            print("File is already processed.")
        else:
            simplify_file(complex_filepath, pred_filepath, features_kwargs, model_dirname)
        
    #    print(evaluate(complex_filepath, pred_filepath, ref_filepaths))
        print("Evaluate: ", pred_filepath)
        with log_stdout(output_score_filepath):
            # print("features_kwargs: ", features_kwargs)
            # scores = evaluate_all_metrics(complex_filepath, pred_filepath, ref_filepaths)
            scores = evaluate_system_output(test_set="turkcorpus_test", 
                                            sys_sents_path=str(pred_filepath),
                                            lowercase=True)
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
            print("SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['bleu'], scores['fkgl']))
            # print("{:.2f} \t {:.2f} \t {:.2f} ".format(scores['SARI'], scores['BLEU'], scores['FKGL']))

            print("Execution time: --- %s seconds ---" % (time.time() - start_time))
            return scores['sari']
            
    else:
        print("Already exist: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))

def evaluate_on_asset(features_kwargs, phase, model_dirname=None):
    '''
    evaluate on the asset dataset (test stage)
    '''
    dataset = "asset"
    # output_dir = REPO_DIR / f"outputs/{_model_dirname}"
    
    model_dir = EXP_DIR / model_dirname
    output_dir = model_dir / "outputs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir: ", output_dir)
    
    features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir / f"score_{features_hash}_{dataset}_{phase}_log.txt"
    if not output_score_filepath.exists() or count_line(output_score_filepath) == 0:
        start_time = time.time()
        complex_filepath = get_data_filepath(dataset, phase, 'orig')
        pred_filepath = output_dir / f'{features_hash}_{complex_filepath.stem}.txt'
        ref_filepaths = [get_data_filepath(dataset, phase, 'simp', i) for i in range(10)]
        print(pred_filepath)
        if pred_filepath.exists() and count_line(pred_filepath) == count_line(complex_filepath):
            print("File is already processed.")
        else:
            simplify_file(complex_filepath, pred_filepath, features_kwargs, model_dirname)
            
        with log_stdout(output_score_filepath):
            # scores = evaluate_all_metrics(complex_filepath, pred_filepath, ref_filepaths)
            scores = evaluate_system_output(test_set="asset_test", sys_sents_path=str(pred_filepath), lowercase=True)
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
            print("SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['bleu'], scores['fkgl']))

            print("Execution time: --- %s seconds ---" % (time.time() - start_time))
    else:
        print("Already exist: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))


def evaluate_on_WIKIDOC(phase, features_kwargs=None,  model_dirname = None):
    dataset = WIKI_DOC_CLEAN
    model_dir = EXP_DIR / model_dirname
    output_dir = model_dir / 'outputs'

    output_dir.mkdir(parents = True, exist_ok = True)
    #features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir / f'score_{dataset}_{phase}.log.txt'
    complex_filepath =get_data_filepath(dataset, phase, 'complex')
    
    if not output_score_filepath.exists() or count_line(output_score_filepath)==0:
        start_time = time.time()
        complex_filepath =get_data_filepath(dataset, phase, 'complex')
        
        #complex_filepath = get_data_filepath(dataset, phase, 'complex_summary_'+str(ratio))
        pred_filepath = output_dir / f'{complex_filepath.stem}.txt'
        ref_filepaths = get_data_filepath(dataset, phase, 'simple')

        if pred_filepath.exists() and count_line(pred_filepath)==count_line(complex_filepath):
            print("File is already processed.")
        else:
            simplify_file(complex_filepath, pred_filepath, features_kwargs, model_dirname)

        print("Evaluate: ", pred_filepath)

        with log_stdout(output_score_filepath):
            scores  = evaluate_system_output(test_set='custom',
                                             sys_sents_path=str(pred_filepath),
                                             orig_sents_path=str(complex_filepath),
                                             refs_sents_paths=str(ref_filepaths))

            # if "WordRatioFeature" in features_kwargs:
            #     print("W:", features_kwargs["WordRatioFeature"]["target_ratio"], "\t", end="")
            # if "CharRatioFeature" in features_kwargs:
            #     print("C:", features_kwargs["CharRatioFeature"]["target_ratio"], "\t", end="")
            # if "LevenshteinRatioFeature" in features_kwargs:
            #     print("L:", features_kwargs["LevenshteinRatioFeature"]["target_ratio"], "\t", end="")
            # if "WordRankRatioFeature" in features_kwargs:
            #     print("WR:", features_kwargs["WordRankRatioFeature"]["target_ratio"], "\t", end="")
            # if "DependencyTreeDepthRatioFeature" in features_kwargs:
            #     print("DTD:", features_kwargs["DependencyTreeDepthRatioFeature"]["target_ratio"], "\t", end="")
            print("SARI: {:.2f}\t D-SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['D-sari'], scores['bleu'], scores['fkgl']))
            # print("{:.2f} \t {:.2f} \t {:.2f} ".format(scores['SARI'], scores['BLEU'], scores['FKGL']))

            print("Execution time: --- %s seconds ---" % (time.time() - start_time))
            return scores['sari']
    else:
        print("Already exists: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))

def evaluate_on_D_WIKI(phase, features_kwargs=None,  model_dirname = None):
    dataset = D_WIKI
    model_dir = EXP_DIR / model_dirname
    output_dir = model_dir / 'outputs'

    output_dir.mkdir(parents = True, exist_ok = True)
    #features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir / f'score_{dataset}_{phase}.log.txt'
    complex_filepath =get_data_filepath(dataset, phase, 'complex')
    
    if not output_score_filepath.exists() or count_line(output_score_filepath)==0:
        start_time = time.time()
        complex_filepath =get_data_filepath(dataset, phase, 'complex')
        
        #complex_filepath = get_data_filepath(dataset, phase, 'complex_summary_'+str(ratio))
        pred_filepath = output_dir / f'{complex_filepath.stem}.txt'
        ref_filepaths = get_data_filepath(dataset, phase, 'simple')

        if pred_filepath.exists() and count_line(pred_filepath)==count_line(complex_filepath):
            print("File is already processed.")
        else:
            simplify_file(complex_filepath, pred_filepath, features_kwargs, model_dirname)

        print("Evaluate: ", pred_filepath)

        with log_stdout(output_score_filepath):
            scores  = evaluate_system_output(test_set='custom',
                                             sys_sents_path=str(pred_filepath),
                                             orig_sents_path=str(complex_filepath),
                                             refs_sents_paths=str(ref_filepaths))

            # if "WordRatioFeature" in features_kwargs:
            #     print("W:", features_kwargs["WordRatioFeature"]["target_ratio"], "\t", end="")
            # if "CharRatioFeature" in features_kwargs:
            #     print("C:", features_kwargs["CharRatioFeature"]["target_ratio"], "\t", end="")
            # if "LevenshteinRatioFeature" in features_kwargs:
            #     print("L:", features_kwargs["LevenshteinRatioFeature"]["target_ratio"], "\t", end="")
            # if "WordRankRatioFeature" in features_kwargs:
            #     print("WR:", features_kwargs["WordRankRatioFeature"]["target_ratio"], "\t", end="")
            # if "DependencyTreeDepthRatioFeature" in features_kwargs:
            #     print("DTD:", features_kwargs["DependencyTreeDepthRatioFeature"]["target_ratio"], "\t", end="")
            print("SARI: {:.2f}\t D-SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['D-sari'], scores['bleu'], scores['fkgl']))
            # print("{:.2f} \t {:.2f} \t {:.2f} ".format(scores['SARI'], scores['BLEU'], scores['FKGL']))

            print("Execution time: --- %s seconds ---" % (time.time() - start_time))
            return scores['sari']
    else:
        print("Already exists: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))


# Specify the token features to use
# features_kwargs = {
#     # 'WordRatioFeature': {'target_ratio': 1.05},
#     'CharRatioFeature': {'target_ratio': 0.93},
#     'LevenshteinRatioFeature': {'target_ratio': 0.62},
#     'WordRankRatioFeature': {'target_ratio': 0.68},
#     'DependencyTreeDepthRatioFeature': {'target_ratio': 0.72}
# }

####### WIKI_DOC #######
#evaluate_on_WIKIDOC(phase='test', features_kwargs=None, model_dirname=model_dirname)

####### WIKI_DOC_CLEAN ########
# T5 Single (original loss): SARI: 48.96      D-SARI: 0.44    BLEU: 23.70     FKGL: 9.17 
# T5_2 (original loss): SARI: 48.89      D-SARI: 0.43    BLEU: 11.53     FKGL: 6.78 
# T5_2 10CosSim(ReLU)+20Sim+1Sum: SARI: 47.81      D-SARI: 0.43    BLEU: 18.65     FKGL: 9.08


### Original loss function ###
####### wiki-doc whole #######
# T5 single (original loss): SARI: 41.74      BLEU: 8.04      FKGL: 9.92 
# BART single (original loss): SARI: 40.84     D-SARI: 0.40    BLEU: 3.83      FKGL: 9.12 
# T5_2 (original loss): SARI: 40.21      D-SARI: 0.38    BLEU: 2.40      FKGL: 7.98 

####### wiki-doc-small #######
# Bart: SARI: 40.16      BLEU: 5.01      FKGL: 9.59 
# T5: SARI: 40.04      BLEU: 2.55      FKGL: 7.73 
# T5 single: SARI: 40.69      BLEU: 4.71      FKGL: 10.02
# T5 single keyword_num: SARI: 40.98      D-SARI: 0.37    BLEU: 6.62      FKGL: 9.80 

### SimLoss + SumLoss ###
# Bart: SARI: 40.46      BLEU: 2.41      FKGL: 8.38
# T5: SARI: 40.51      BLEU: 4.96      FKGL: 8.99 

####### wiki-doc-mid #######
# Bart (original loss) : SARI: 41.03      BLEU: 5.93      FKGL: 9.32


####### D_WIKI #######
evaluate_on_D_WIKI(phase='test', features_kwargs=None, model_dirname=model_dirname)
# T5 single: SARI: 44.72      D-SARI: 0.36    BLEU: 25.67     FKGL: 9.07
# T5_2: SARI: 48.65      D-SARI: 0.38    BLEU: 13.86     FKGL: 6.58
# T5_2 key_num original loss: SARI: 48.28      D-SARI: 0.37    BLEU: 16.41     FKGL: 7.01
# BART Single: SARI: 46.18      D-SARI: 0.37    BLEU: 20.65     FKGL: 8.35 

# T5_2 5Sim+1Sum: SARI: 44.65

####### trained on D_wiki_small #######
# T5 -8CosSim+20SimLoss+3SumLoss: SARI: 44.14      BLEU: 22.12     FKGL: 9.09 
# T5 -6CosSim+20SimLoss+1SumLoss: SARI: 45.80      D-SARI: 0.34    BLEU: 19.11     FKGL: 8.37 
# T5          50SimLoss+1SumLoss: SARI: 45.12      D-SARI: 0.34    BLEU: 20.96     FKGL: 8.61
# T5 -10CosSim+50SimLoss+1SumLoss: SARI: 45.14      D-SARI: 0.34    BLEU: 19.92     FKGL: 8.50 
# T5 -10CosSim(ReLU)+50SimLoss+1SumLoss: SARI: 45.28      D-SARI: 0.34    BLEU: 19.64     FKGL: 8.40
# T5 5KL + 20Sim + 1Sum: SARI: 46.53      D-SARI: 0.35    BLEU: 11.86     FKGL: 7.06 
# T5 15KL + 20Sim + 1Sum: SARI: 46.69      D-SARI: 0.35    BLEU: 11.16     FKGL: 7.04
# T5 key_num (from complex-sent): SARI: 47.16      D-SARI: 0.36    BLEU: 13.67     FKGL: 7.18 

### Original loss function ###
# Bart (trained on wiki-doc-small): SARI: 42.57      BLEU: 12.57     FKGL: 8.85 
# T5 (trained on wiki-doc-small): SARI: 43.35      BLEU: 11.32     FKGL: 8.73

### SimLoss + SumLoss ###
# Bart (trained on wiki-doc-small): SARI: 42.75      BLEU: 7.66      FKGL: 8.02 



####### Turkcorpus #######
#evaluate_on_TurkCorpus(features_kwargs, 'test', model_dirname = model_dirname)

####### without tokens #########
# C: 0.97         L: 0.78         WR: 0.8         DTD: 0.8        SARI: 38.00      BLEU: 75.52     FKGL: 6.56 


######## wikiparagh old loss 10 epoch ###############
# C: 0.98         L: 0.72         WR: 0.8         DTD: 0.72       SARI: 43.85      BLEU: 68.69     FKGL: 6.68 

######## wikilarge 20 0.5prob mean ########
# C: 0.96   L: 0.75     WR: 0.94    DTD: 0.77   SARI: 42.95      BLEU: 69.67     FKGL: 6.70

######## wikilarge 60 0.3prob mean ########
#  C: 0.97         L: 0.73         WR: 0.81        DTD: 0.72       SARI: 43.49      BLEU: 68.22     FKGL: 6.57

######## wikilarge 60 0.5prob mean ########
# C: 0.96   L: 0.77     WR: 0.74    DTD: 0.74   SARI: 43.00      BLEU: 70.08     FKGL: 7.10 

######## wikilargeF 0 0 ###############
# C: 0.96   L: 0.68     WR: 0.8     DTD: 0.75   SARI: 43.14      BLEU: 75.35     FKGL: 7.11 

######## 20 0.3*prob mean wikilargeF ###############
# C: 0.96   L: 0.78     WR: 0.93    DTD: 0.74   SARI: 42.89      BLEU: 72.12     FKGL: 7.45 

######## 60 0.3*prob mean wikilargeF ###############
# C: 0.98         L: 0.72         WR: 0.9         DTD: 0.74       SARI: 42.93      BLEU: 68.01     FKGL: 6.88 




###### ASSET #############
#evaluate_on_asset(features_kwargs, 'test', model_dirname = model_dirname)

##### wikiparagh old loss 10 epoch ###############
# C: 0.97         L: 0.67         WR: 0.71        DTD: 0.74       SARI: 45.72      BLEU: 66.47     FKGL: 6.46  

##### wikilargeF 0 0 ###############
# C: 0.94         L: 0.62         WR: 0.72        DTD: 0.71       SARI: 44.82      BLEU: 72.12     FKGL: 6.73 

##### wikilargeF 20 0.3prob mean #############
# C: 0.94         L: 0.63         WR: 0.8         DTD: 0.76       SARI: 45.15      BLEU: 58.14     FKGL: 6.69

##### wikilargeF 60 0.3prob mean #############
# C: 0.98         L: 0.65         WR: 0.7         DTD: 0.69       SARI: 45.60      BLEU: 64.83     FKGL: 6.37 

#### wikiparaghF old loss ############
# C: 0.98         L: 0.66         WR: 0.75        DTD: 0.72       SARI: 45.84      BLEU: 63.20     FKGL: 6.27 