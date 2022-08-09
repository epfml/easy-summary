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
from preprocessor import get_data_filepath, EXP_DIR, TURKCORPUS_DATASET, REPO_DIR, WIKI_DOC, D_WIKI
from preprocessor import write_lines, yield_lines, count_line, read_lines, generate_hash
from easse.sari import corpus_sari
import time
from googletrans import Translator
from Bart2 import SumSim
#from T5_2 import SumSim


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
model_dirname = 'exp_WikiDocSmall_BART'
checkpoint_path = 'checkpoint-epoch=3.ckpt'

# load the model
#Model = T5FineTuner.load_from_checkpoint(EXP_DIR / model_dirname / checkpoint_path).to(device)
# model = Model.model.to(device)
# tokenizer = Model.tokenizer
Model = SumSim.load_from_checkpoint(EXP_DIR /  model_dirname / checkpoint_path).to(device)
summarizer = Model.summarizer.to(device)
simplifier = Model.simplifier.to(device)
summarizer_tokenizer = Model.summarizer_tokenizer
simplifier_tokenizer = Model.simplifier_tokenizer
translator = Translator()




def generate(sentence, preprocessor=None):
    '''
    Apply model to generate prediction
    '''
    # if not torch.cuda.is_available():
    #     print("Simplifying: ", sentence)
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
    
    summary_atten_mask = torch.ones(summary_ids.shape).to(device)
    summary_atten_mask[summary_ids[:,:] == summarizer_tokenizer.pad_token_id] = 0
    
    beam_outputs = simplifier.generate(
        input_ids = summary_ids,
        attention_mask = summary_atten_mask,
        do_sample = True,
        max_length = 256,
        num_beams = 10, top_k = 130, top_p = 0.95,
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
    dataset = WIKI_DOC
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
            print("SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['bleu'], scores['fkgl']))
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
            print("SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['bleu'], scores['fkgl']))
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

# Bart: SARI: 40.16      BLEU: 5.01      FKGL: 9.59 
# T5: SARI: 39.15      BLEU: 2.19      FKGL: 6.15 

####### D_WIKI #######
evaluate_on_D_WIKI(phase='test', features_kwargs=None, model_dirname=model_dirname)

# evaluate_on_WIKIDOC(features_kwargs=features_kwargs, 
#                     phase='test', ratio = 0.7,
#                     model_dirname=model_dirname)
#### wikiparagh oldloss ####
# original doc
# C: 0.95         L: 0.68         WR: 0.82        DTD: 0.79       SARI: 39.24      BLEU: 8.90      FKGL: 10.03 

# doc 0.1 summarization
# C: 0.94         L: 0.61         WR: 0.79        DTD: 0.72       SARI: 36.48      BLEU: 6.88      FKGL: 8.88

# doc 0.3 summarization
# C: 0.94         L: 0.61         WR: 0.79        DTD: 0.72       SARI: 38.37      BLEU: 8.25      FKGL: 8.45 

# doc 0.5 summarization
# C: 0.95         L: 0.62         WR: 0.67        DTD: 0.72       SARI: 38.79      BLEU: 8.62      FKGL: 8.54

# doc 0.7 summarization
# C: 0.93         L: 0.62         WR: 0.68        DTD: 0.72       SARI: 39.55      BLEU: 9.60      FKGL: 8.38 

# doc 0.9 summarization
# C: 0.94         L: 0.67         WR: 0.74        DTD: 0.76       SARI: 39.74      BLEU: 10.63     FKGL: 8.60 


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

######## 60 0.5*prob mean wikilargeF ###############
# C: 0.97         L: 0.77         WR: 0.9         DTD: 0.74       SARI: 42.94      BLEU: 70.23     FKGL: 7.44 

######## paraghF oldloss ############
# C: 0.98   L: 0.73     WR: 0.8     DTD: 0.72   SARI: 43.76      BLEU: 67.86     FKGL: 6.85 

######## paraghF 20 0.3prob mean ############
# C: 0.98         L: 0.73         WR: 0.8         DTD: 0.72       SARI: 43.76      BLEU: 67.86     FKGL: 6.85 

######## paraghF 20 0.3prob max ############
# C: 0.98   L: 0.73     WR: 0.92    DTD: 0.72   SARI: 43.69      BLEU: 68.19     FKGL: 6.83 

####### paraghF 60 0.3prob mean #############
# C: 0.98         L: 0.73         WR: 0.8         DTD: 0.72       SARI: 43.76      BLEU: 67.86     FKGL: 6.85

####### paraghF 60 0.3prob max #############
# C: 0.96   L: 0.77     WR: 0.92    DTD: 0.74   SARI: 42.78      BLEU: 67.35     FKGL: 7.35

####### paraghF 60 0.5prob mean #############
# from wikiparaghF_0_0
# C: 0.97         L: 0.72         WR: 0.8         DTD: 0.72       SARI: 43.47      BLEU: 65.14     FKGL: 6.81 

# from wikilargeF_0_0
# C: 0.98         L: 0.73         WR: 0.8         DTD: 0.72       SARI: 43.76      BLEU: 67.86     FKGL: 6.85


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