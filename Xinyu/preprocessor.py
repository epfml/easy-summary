'''
Preprocess the original dataset and get special-tokened sentences, which not used in
our paper.
From https://github.com/KimChengSHEANG/TS_T5/blob/main/source/preprocessor.py
'''

# -- fix path --
import optparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
# -- end fix path --
from functools import lru_cache
from multiprocessing import Pool, Lock
from string import punctuation
import multiprocessing
import Levenshtein
import numpy as np
import spacy
import nltk
import shutil
import time
import pickle
import hashlib


nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sacremoses import MosesDetokenizer, MosesTokenizer


REPO_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = REPO_DIR / 'resources'
EXP_DIR = REPO_DIR / 'experiments'
DATASETS_DIR = REPO_DIR / 'data'

OUTPUT_DIR = REPO_DIR / 'output'
DUMPS_DIR = RESOURCES_DIR / "DUMPS"

WIKI_DOC = 'wiki_doc'
D_WIKI = 'D_wiki'


LANGUAGES = ['complex', 'simple']
#PHASES = ['train', 'valid','test']
PHASES = ['train','valid']
#PHASES = ['train', 'valid', 'test']

# from source.helper import tokenize, yield_lines, load_dump, dump, write_lines, count_line, \
#     print_execution_time, save_preprocessor, yield_sentence_pair

stopwords = set(stopwords.words('english'))

#######################
def get_tokenizer():
    return MosesTokenizer(lang='en')

def get_detokenizer():
    return MosesDetokenizer(lang='en')

def tokenize(sentence):
    return get_tokenizer().tokenize(sentence)

def write_lines(lines, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as fout:
        for line in lines:
            fout.write(line + '\n')


def read_lines(filepath):
    return [line.rstrip() for line in yield_lines(filepath)]


def yield_lines(filepath):
    filepath = Path(filepath)
    with filepath.open('r') as f:
        for line in f:
            yield line.rstrip()


def yield_sentence_pair_with_index(filepath1, filepath2):
    index = 0
    with Path(filepath1).open('r') as f1, Path(filepath2).open('r') as f2:
        for line1, line2 in zip(f1, f2):
            index += 1
            yield (line1.rstrip(), line2.rstrip(), index)
            

def yield_sentence_pair(filepath1, filepath2):
    with Path(filepath1).open('r') as f1, Path(filepath2).open('r') as f2:
        for line1, line2 in zip(f1, f2):
            yield line1.rstrip(), line2.rstrip()


def count_line(filepath):
    filepath = Path(filepath)
    line_count = 0
    with filepath.open("r") as f:
        for line in f:
            line_count += 1
    return line_count


def load_dump(filepath):
    return pickle.load(open(filepath, 'rb'))


def dump(obj, filepath):
    pickle.dump(obj, open(filepath, 'wb'))

def save_preprocessor(preprocessor):
    DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_DUMP_FILE = DUMPS_DIR / 'preprocessor.pickle'
    dump(preprocessor, PREPROCESSOR_DUMP_FILE)


def load_preprocessor():
    PREPROCESSOR_DUMP_FILE = DUMPS_DIR / 'preprocessor.pickle'
    if PREPROCESSOR_DUMP_FILE.exists():
        return load_dump(PREPROCESSOR_DUMP_FILE)
    else:
        return None

def generate_hash(data):
    h = hashlib.new('md5')
    h.update(str(data).encode())
    return h.hexdigest()

def get_data_filepath(dataset, phase, type, i=None):
    suffix = ''
    if i is not None:
        suffix = f'.{i}'
    filename = f'{dataset}.{phase}.{type}{suffix}'
    return DATASETS_DIR / dataset / filename

#################################

def round(val):
    return '%.2f' % val


def safe_division(a, b):
    return a / b if b else 0


# def tokenize(sentence):
#     return sentence.split()

def is_punctuation(word):
    return ''.join([char for char in word if char not in punctuation]) == ''


def remove_punctuation(text):
    return ' '.join([word for word in tokenize(text) if not is_punctuation(word)])


def remove_stopwords(text):
    return ' '.join([w for w in tokenize(text) if w.lower() not in stopwords])


def get_dependency_tree_depth(sentence):
    def tree_height(node):
        if len(list(node.children)) == 0:
            return 0
        return 1 + max(tree_height(child) for child in node.children)

    tree_depths = [tree_height(spacy_sentence.root) for spacy_sentence in spacy_process(sentence).sents]
    if len(tree_depths) == 0:
        return 0
    return max(tree_depths)

def get_spacy_model():
    model = 'en_core_web_sm'
    if not spacy.util.is_package(model):
        spacy.cli.download(model)
        spacy.cli.link(model, model, force=True, model_path=spacy.util.get_package_path(model))
    return spacy.load(model)


def spacy_process(text):
    return get_spacy_model()(str(text))


def get_word2rank(vocab_size=np.inf):
    model_filepath = DUMPS_DIR / f"{WORD_EMBEDDINGS_NAME}.pk"
    if model_filepath.exists():
        return load_dump(model_filepath)


def get_normalized_rank(word):
    max = len(get_word2rank())
    rank = get_word2rank().get(word, max)
    return np.log(1 + rank) / np.log(1 + max)

    

def get_complexity_score2(sentence):
    words = tokenize(remove_stopwords(remove_punctuation(sentence)))
    words = [word for word in words if word in get_word2rank()]  # remove unknown words
    if len(words) == 0:
        return 1.0
    return np.array([get_normalized_rank(word) for word in words]).mean()

def get_word_frequency():
    model_filepath = DUMPS_DIR / f'{WORD_FREQUENCY_FILEPATH.stem}.pk'
    if model_filepath.exists():
        return load_dump(model_filepath)
    else:
        DUMPS_DIR.mkdir(parents=True, exist_ok=True) 
        word_freq = {}
        for line in yield_lines(WORD_FREQUENCY_FILEPATH):
            chunks = line.split(' ')
            word = chunks[0]
            freq = int(chunks[1])
            word_freq[word] = freq
        dump(word_freq, model_filepath)
        return word_freq


def get_normalized_inverse_frequency(word):
    max = 153141437 # the 153141437, the max frequency
    freq = get_word_frequency().get(word, 0)
    return 1.0 - np.log(1 + freq) / np.log(1 + max)


def get_complexity_score(sentence, operation_type = None):
    words = tokenize(remove_stopwords(remove_punctuation(sentence)))
    #words = tokenize(remove_punctuation(sentence))
    words = [word for word in words if word in get_word2rank()]  # remove unknown words
    if len(words) == 0:
        return 1.0
    if operation_type == 'mean':
        return np.array([get_normalized_inverse_frequency(word.lower()) for word in words]).mean()
    else:
        return np.array([get_normalized_inverse_frequency(word.lower()) for word in words]).max()




class RatioFeature:
    def __init__(self, feature_extractor, target_ratio=0.8):
        self.feature_extractor = feature_extractor
        self.target_ratio = target_ratio

    def encode_sentence(self, sentence):
        return f'{self.name}_{self.target_ratio}'

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        return f'{self.name}_{self.feature_extractor(complex_sentence, simple_sentence)}', simple_sentence

    def decode_sentence(self, encoded_sentence):
        return encoded_sentence

    @property
    def name(self):
        class_name = self.__class__.__name__.replace('RatioFeature', '')
        name = ""
        for word in re.findall('[A-Z][^A-Z]*', class_name):
            if word: name += word[0]
        if not name: name = class_name
        return name

### tokens features ###
class WordRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_word_length_ratio, *args, **kwargs)

    def get_word_length_ratio(self, complex_sentence, simple_sentence):
        return round(safe_division(len(tokenize(simple_sentence)), len(tokenize(complex_sentence))))


class CharRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_char_length_ratio, *args, **kwargs)

    def get_char_length_ratio(self, complex_sentence, simple_sentence):
        return round(safe_division(len(simple_sentence), len(complex_sentence)))


class LevenshteinRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_levenshtein_ratio, *args, **kwargs)

    def get_levenshtein_ratio(self, complex_sentence, simple_sentence):
        return round(Levenshtein.ratio(complex_sentence, simple_sentence))


class WordRankRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_word_rank_ratio, *args, **kwargs)

    def get_word_rank_ratio(self, complex_sentence, simple_sentence):
        return round(min(safe_division(self.get_lexical_complexity_score(simple_sentence),
                                       self.get_lexical_complexity_score(complex_sentence)), 2))

    def get_lexical_complexity_score(self, sentence):
        words = tokenize(remove_stopwords(remove_punctuation(sentence)))
        words = [word for word in words if word in get_word2rank()]
        if len(words) == 0:
            return np.log(1 + len(get_word2rank()))
        return np.quantile([self.get_rank(word) for word in words], 0.75)

    
    def get_rank(self, word):
        # return get_normalized_inverse_frequency(word)
        rank = get_word2rank().get(word, len(get_word2rank()))
        return np.log(1 + rank)


class DependencyTreeDepthRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_dependency_tree_depth_ratio, *args, **kwargs)

    def get_dependency_tree_depth_ratio(self, complex_sentence, simple_sentence):
        return round(
            safe_division(self.get_dependency_tree_depth(simple_sentence),
                          self.get_dependency_tree_depth(complex_sentence)))

    
    def get_dependency_tree_depth(self, sentence):
        def get_subtree_depth(node):
            if len(list(node.children)) == 0:
                return 0
            return 1 + max([get_subtree_depth(child) for child in node.children])

        tree_depths = [get_subtree_depth(spacy_sentence.root) for spacy_sentence in self.spacy_process(sentence).sents]
        if len(tree_depths) == 0:
            return 0
        return max(tree_depths)

    def spacy_process(self, text):
        return get_spacy_model()(text)

class Preprocessor:
    def __init__(self, features_kwargs=None):
        super().__init__()

        self.features = self.get_features(features_kwargs)
        if features_kwargs:
            self.hash = generate_hash(str(features_kwargs).encode())
        else:
            self.hash = "no_feature"

    def get_class(self, class_name, *args, **kwargs):
        return globals()[class_name](*args, **kwargs)

    def get_features(self, feature_kwargs):
        features = []
        for feature_name, kwargs in feature_kwargs.items():
            features.append(self.get_class(feature_name, **kwargs))
        return features

    def encode_sentence(self, sentence):
        if self.features:
            line = ''
            for feature in self.features:
                line += feature.encode_sentence(sentence) + ' '
            line += ' ' + sentence
            return line.rstrip()
        else:
            return sentence

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        # print(complex_sentence)
        if self.features:
            line = ''
            for feature in self.features:
                # startTime = timeit.default_timer()
                # print(feature)
                processed_complex, _ = feature.encode_sentence_pair(complex_sentence, simple_sentence)
                line += processed_complex + ' '
                # print(feature, timeit.default_timer() - startTime)
            line += ' ' + complex_sentence
            return line.rstrip()

        else:
            return complex_sentence

    def decode_sentence(self, encoded_sentence):
        for feature in self.features:
            decoded_sentence = feature.decode_sentence(encoded_sentence)
        return decoded_sentence

    def encode_file(self, input_filepath, output_filepath):
        with open(output_filepath, 'w') as f:
            for line in yield_lines(input_filepath):
                f.write(self.encode_sentence(line) + '\n')

    def decode_file(self, input_filepath, output_filepath):
        with open(output_filepath, 'w') as f:
            for line in yield_lines(input_filepath):
                f.write(self.decode_sentence(line) + '\n')

    def process_encode_sentence_pair(self, sentences):
        print(f"{sentences[2]}/{self.line_count}", sentences[0])  # sentence[0] index
        return (self.encode_sentence_pair(sentences[0], sentences[1]))

    def pool_encode_sentence_pair(self, args):
        # print(f"{processed_line_count}/{self.line_count}")
        complex_sent, simple_sent, queue = args
        queue.put(1)
        return self.encode_sentence_pair(complex_sent, simple_sent)

    
    def encode_file_pair(self, complex_filepath, simple_filepath):
        # print(f"Preprocessing file: {complex_filepath}")
        processed_complex_sentences = []
        self.line_count = count_line(simple_filepath)

        # nb_cores = multiprocessing.cpu_count()
        # manager = multiprocessing.Manager()
        # queue = manager.Queue()

        # pool = Pool(processes=nb_cores)
        # args = [(complex_sent, simple_sent, queue) for complex_sent, simple_sent in
        #         yield_sentence_pair(complex_filepath, simple_filepath)]
        # res = pool.map_async(self.pool_encode_sentence_pair, args)
        # cnt=0
        # while not res.ready():
        #     # remaining = res._number_left * res._chunksize
        #     size = queue.qsize()
        #     print(f"Preprocessing: {size} / {self.line_count}")
        #     cnt+=1
        #     if cnt>=30:
        #         break
        #     #time.sleep(0.5)
        # encoded_sentences = res.get()
        # pool.close()
        # pool.join()
        # pool.terminate()
        i = 0
        for complex_sentence, simple_sentence in yield_sentence_pair(complex_filepath, simple_filepath):
        # print(complex_sentence)
            processed_complex_sentence = self.encode_sentence_pair(complex_sentence, simple_sentence)
            i +=1
            print(f"{i}/{self.line_count}", processed_complex_sentence)
            processed_complex_sentences.append(processed_complex_sentence)

        return processed_complex_sentences

    def get_preprocessed_filepath(self, dataset, phase, type):
        filename = f'{dataset}.{phase}.{type}'
        return self.preprocessed_data_dir / filename

    def preprocess_dataset(self, dataset):
        # download_requirements()
        self.preprocessed_data_dir = PROCESSED_DATA_DIR / self.hash / dataset
        self.preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
        save_preprocessor(self)
        print(f'Preprocessing dataset: {dataset}')

        for phase in PHASES:
            # for phase in ["valid", "test"]:
            complex_filepath = get_data_filepath(dataset, phase, 'complex')
            simple_filepath = get_data_filepath(dataset, phase, 'simple')

            complex_output_filepath = self.preprocessed_data_dir / complex_filepath.name
            simple_output_filepath = self.preprocessed_data_dir / simple_filepath.name
            if complex_output_filepath.exists() and simple_output_filepath.exists():
                continue

            print(f'Prepocessing files: {complex_filepath.name} {simple_filepath.name}')
            processed_complex_sentences = self.encode_file_pair(complex_filepath, simple_filepath)

            write_lines(processed_complex_sentences, complex_output_filepath)
            shutil.copy(simple_filepath, simple_output_filepath)

        print(f'Preprocessing dataset "{dataset}" is finished.')
        return self.preprocessed_data_dir

if __name__ == '__main__':
    features_kwargs = {
        # 'WordRatioFeature': {'target_ratio': 0.8},
        'CharRatioFeature': {'target_ratio': 0.8},
        'LevenshteinRatioFeature': {'target_ratio': 0.8},
        'WordRankRatioFeature': {'target_ratio': 0.8},
        'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
    }

    preprocessor = load_preprocessor()
    # preprocessor.preprocess_dataset(WIKI_DATASET)
