# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
from preprocessor import DATASETS_DIR,PROCESSED_DATA_DIR,D_WIKI, D_WIKI_MATCH, WIKI_DOC,WIKI_DOC_FILTER,WIKI_DOC_CLEAN,WIKI_DOC_MATCH, WIKI_DOC_FINAL, WIKI_PARAGH_FILTER_DATASET, WIKILARGE_DATASET,yield_sentence_pair,yield_lines, WIKI_PARA_DATASET, get_data_filepath, tokenize, write_lines
import numpy as np
from preprocessor import WIKILARGE_DATASET
from rouge import Rouge
import matplotlib.pyplot as plt
import time
import nltk
import numpy as np
#from keybert import KeyBERT
from transformers.pipelines import pipeline
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# hf_model = pipeline("feature-extraction", model="distilbert-base-cased")

# kw_model = KeyBERT(model = 'all-mpnet-base-v2')

dataset = WIKI_DOC_FINAL

TYPES = ['complex', 'simple']
PHASES = ['train','valid','test']

'''
## wikilarge train set path
WikiLarge_train_complex = get_data_filepath(WIKILARGE_DATASET, 'train', 'complex')
WikiLarge_train_simple = get_data_filepath(WIKILARGE_DATASET, 'train', 'simple')

### combine wiki-doc train, valid and test set into one list
WikiDoc_complex_total = []
WikiDoc_simple_total = []

for ps in PHASES:
    ### path
    WikiDoc_complex = get_data_filepath(dataset, ps, 'complex')
    WikiDoc_simple = get_data_filepath(dataset, ps, 'simple')
    for complex_sentence, simple_sentence in yield_sentence_pair(WikiDoc_complex, WikiDoc_simple):
        WikiDoc_complex_total.append(complex_sentence)
        WikiDoc_simple_total.append(simple_sentence)

cnt = 0
WikiDocFinal_train_complex = []
WikiDocFinal_train_simple = []
WikiDocFinal_valid_complex = []
WikiDocFinal_valid_simple = []
WikiDocFinal_test_complex = []
WikiDocFinal_test_simple = []



### wikilarge path
WikiLarge_complex = get_data_filepath(WIKILARGE_DATASET, 'train', 'complex')
WikiLarge_simple = get_data_filepath(WIKILARGE_DATASET, 'train', 'simple')
for i in range(len(WikiDoc_complex_total)):
    complex_doc = WikiDoc_complex_total[i]
    simple_doc = WikiDoc_simple_total[i]

    for complex_sentence, simple_sentence in yield_sentence_pair(WikiLarge_complex, WikiLarge_simple):
        if simple_doc == simple_sentence:
            ### take the overlapping sample to the train set
            cnt+=1
            print(cnt)
            WikiDocFinal_train_complex.append(complex_doc)
            WikiDocFinal_train_simple.append(simple_doc)
            break

print(len(WikiDocFinal_train_complex)) #4685
## remove the overlapping samples from the wikidoc dataset
for i in range(len(WikiDocFinal_train_complex)):
    WikiDoc_complex_total.remove(WikiDocFinal_train_complex[i])
    WikiDoc_simple_total.remove(WikiDocFinal_train_simple[i])

### shuffle the wikidoctotal dataset
WikiDoc_complex_total = np.array(WikiDoc_complex_total)
WikiDoc_simple_total = np.array(WikiDoc_simple_total)
np.random.seed(42)
np.random.shuffle(WikiDoc_complex_total)
np.random.seed(42)
np.random.shuffle(WikiDoc_simple_total)
print(WikiDoc_complex_total[5])
print(WikiDoc_simple_total[5])


train_sz = 13973 - len(WikiDocFinal_train_complex)
valid_sz = 1768
test_sz = 1704

WikiDocFinal_train_complex = WikiDocFinal_train_complex + WikiDoc_complex_total[:train_sz].tolist()
WikiDocFinal_train_simple = WikiDocFinal_train_simple + WikiDoc_simple_total[:train_sz].tolist()
print(len(WikiDocFinal_train_complex))

WikiDocFinal_valid_complex = WikiDoc_complex_total[train_sz:train_sz+valid_sz].tolist()
WikiDocFinal_valid_simple = WikiDoc_simple_total[train_sz:train_sz+valid_sz].tolist()

WikiDocFinal_test_complex = WikiDoc_complex_total[train_sz+valid_sz:].tolist()
WikiDocFinal_test_simple = WikiDoc_simple_total[train_sz+valid_sz:].tolist()
print(len(WikiDocFinal_test_complex))


### write to file
for ps in PHASES:
    save_complex = get_data_filepath(WIKI_DOC_FINAL, ps, 'complex')
    save_simple = get_data_filepath(WIKI_DOC_FINAL, ps, 'simple')

    if ps == 'train':
        write_file_obj = open(save_complex, 'w', encoding='utf-8')
        for var in WikiDocFinal_train_complex:
            write_file_obj.write(var)
            write_file_obj.write('\n')
        write_file_obj.close()

        write_file_obj = open(save_simple, 'w', encoding='utf-8')
        for var in WikiDocFinal_train_simple:
            write_file_obj.write(var)
            write_file_obj.write('\n')
        write_file_obj.close()
    
    elif ps == 'valid':
        write_file_obj = open(save_complex, 'w', encoding='utf-8')
        for var in WikiDocFinal_valid_complex:
            write_file_obj.write(var)
            write_file_obj.write('\n')
        write_file_obj.close()

        write_file_obj = open(save_simple, 'w', encoding='utf-8')
        for var in WikiDocFinal_valid_simple:
            write_file_obj.write(var)
            write_file_obj.write('\n')
        write_file_obj.close()
    
    elif ps == 'test':
        write_file_obj = open(save_complex, 'w', encoding='utf-8')
        for var in WikiDocFinal_test_complex:
            write_file_obj.write(var)
            write_file_obj.write('\n')
        write_file_obj.close()

        write_file_obj = open(save_simple, 'w', encoding='utf-8')
        for var in WikiDocFinal_test_simple:
            write_file_obj.write(var)
            write_file_obj.write('\n')
        write_file_obj.close()

'''





# cnt = 0

# for ps in ['test']:
#     ### enumerate the samples in WIKILAEGE_DATASET
#     wikilarge_file_path_complex = get_data_filepath(WIKILARGE_DATASET, 'train', 'complex')
#     wikilarge_file_path_simple = get_data_filepath(WIKILARGE_DATASET, 'train', 'simple')

#     test_file_path_complex = get_data_filepath(WIKI_DOC_FINAL, ps, 'complex')
#     test_file_path_simple = get_data_filepath(WIKI_DOC_FINAL, ps, 'simple')

#     ### enumerate the samples in test dataset
#     for complex_sentence, simple_sentence in yield_sentence_pair(test_file_path_complex, test_file_path_simple):
#         ### check if the simple sentence is in the wikilarge dataset
#         for complex_sentence_wiki, simple_sentence_wiki in yield_sentence_pair(wikilarge_file_path_complex, wikilarge_file_path_simple):
#             if (simple_sentence == simple_sentence_wiki) :
#                 cnt+=1
#                 print(cnt)
#                 print("simple sentence in the test set: ", simple_sentence)
#                 print("simple sentence in the wikilarge set: ", simple_sentence_wiki)
#                 time.sleep(2)
#                 break


# print(cnt)
 
for ps in PHASES:
    save_complex_path = get_data_filepath(WIKI_DOC_FINAL, ps, 'complex')
    #save_simple_path = get_data_filepath(WIKI_DOC_FINAL, ps, 'simple')
    simple_file_path = get_data_filepath(dataset,ps, 'simple')
    complex_file_path = get_data_filepath(dataset,ps, 'complex')
    cnt=0
    tot=0
    complex_lens = []
    simple_lens = []
    L_comp = 0
    L_sim = 0

    for complex_sentence, simple_sentence in yield_sentence_pair(complex_file_path, simple_file_path):
        # L1 = len(tokenize(complex_sentence))
        # L2 = len(tokenize(simple_sentence))

        # if L1>1000 or L2>256:
        #     continue

        # if L1<L2:
        #     cnt+=1
        #     L_comp+=L1
        #     L_sim+=L2
            #print(cnt)
            # print(complex_sentence)
            # print(simple_sentence)
#             print("----------------------------")
#             time.sleep(6.0)

    #     simlist = []
    #     sim_kw = kw_model.extract_keywords(simple_sentence, keyphrase_ngram_range=(1, 1), stop_words=None,top_n = 10, use_mmr = True,diversity = 0.5)
    #     com_kw = kw_model.extract_keywords(complex_sentence, keyphrase_ngram_range=(1, 1), stop_words=None, top_n = 10, use_mmr = True,diversity = 0.5)

    #     # fg = True

    #     for i in range(min(5, len(sim_kw))):
    #         simlist.append(sim_kw[i][0])

    #     simlist = set(simlist)
    #     comlist = []

    #     for i in range(min(5, len(com_kw))):
    #         comlist.append(com_kw[i][0])
    #         if com_kw[i][0] in simlist:
    #             #cnt+=1
    #             tot+=1
    #             print(tot)
    #             complex_lens.append(complex_sentence)
    #             simple_lens.append(simple_sentence)
    #             # fg = False
    #             break
    #     # if fg:
    #     #     cnt+=1
    #     #     print(cnt)
    #     #     print(comlist)
    #     #     print(simlist)
    #     #     complex_lens.append(complex_sentence)
    #     #     simple_lens.append(simple_sentence)
    
    # file_write_obj = open(save_complex_path, 'w', encoding='utf-8')
    # for var in complex_lens:
    #     file_write_obj.write(var)
    #     file_write_obj.write('\n')
    # file_write_obj.close()

    # file_write_obj = open(save_simple_path, 'w', encoding='utf-8')
    # for var in simple_lens:
    #     file_write_obj.write(var)
    #     file_write_obj.write('\n')
    # file_write_obj.close()
    print('avg len of complex sentence: ', L_comp/cnt, 'Phase: ', ps)
    print('avg len of simple sentence: ', L_sim/cnt, 'Phase: ', ps)
    


    #print(f'{ps}: {cnt}/{tot}')
# train: 23889/34727
# valid: 5892/8540
# test: 118/166

'''
complex_lens = []
simple_lens = []
orig_complex_lens = []
orig_simple_lens = []

sent_num_comp = 0
sent_num_sim = 0
word_num_comp = 0
word_num_sim = 0
tot=0
for phase in PHASES:

    complex_file = get_data_filepath(D_WIKI, phase, 'complex')
    simple_file = get_data_filepath(D_WIKI, phase, 'simple')
    
    # save_complex_path = get_data_filepath(WIKI_DOC_FILTER, phase, 'complex')
    # save_simple_path = get_data_filepath(WIKI_DOC_FILTER, phase, 'simple')
    tmp = []
    for complex_sent, simple_sent in (yield_sentence_pair(complex_file, simple_file)):
        #tmp.append(len(tokenize(complex_sent))/len(tokenize(simple_sent)))
        word_num_comp += len(tokenize(complex_sent))
        word_num_sim += len(tokenize(simple_sent))
        a = len(sent_tokenize(complex_sent))
        b = len(sent_tokenize(simple_sent))
        sent_num_comp += a
        sent_num_sim += b
        tot+=1
        #print(tot)

        # sent_num_comp += a 
        # sent_num_sim += b
        # orig_complex_lens.append(a)
        # orig_simple_lens.append(b)
    
#     file_write_obj = open(save_complex_path, 'w', encoding='utf-8')
#     for var in complex_lens:
#         file_write_obj.write(var)
#         file_write_obj.write('\n')
#     file_write_obj.close()

#     file_write_obj = open(save_simple_path, 'w', encoding='utf-8')
#     for var in simple_lens:
#         file_write_obj.write(var)
#         file_write_obj.write('\n')
#     file_write_obj.close()

#     print("done")
print('D-wiki words per complex sentence: ', word_num_comp/sent_num_comp, 'D-wiki words per simple sentence: ', word_num_sim/sent_num_sim)
# print("Wiki-doc-match: complex sent: ", word_num_comp/sent_num_comp)
# print("Wiki-doc-match: simple sent: ", word_num_sim/sent_num_sim)

sent_num_comp = 0
sent_num_sim = 0
word_num_comp = 0
word_num_sim = 0
tot=0
for phase in PHASES:

    complex_file = get_data_filepath(WIKI_DOC, phase, 'complex')
    simple_file = get_data_filepath(WIKI_DOC, phase, 'simple')
    #tot=0
    # save_complex_path = get_data_filepath(WIKI_DOC_FILTER, phase, 'complex')
    # save_simple_path = get_data_filepath(WIKI_DOC_FILTER, phase, 'simple')
    tmp = []
    for complex_sent, simple_sent in (yield_sentence_pair(complex_file, simple_file)):
        #tmp.append(len(tokenize(complex_sent))/len(tokenize(simple_sent)))
        word_num_comp += len(tokenize(complex_sent))
        word_num_sim += len(tokenize(simple_sent))
        a = len(sent_tokenize(complex_sent))
        b = len(sent_tokenize(simple_sent))
        sent_num_comp += a
        sent_num_sim += b
        tot+=1
        # complex_lens.append(a)
        # simple_lens.append(b)
        # print(tot)
        # sent_num_comp += a
        # sent_num_sim += b 
# print("D-wiki-match: complex: ", word_num_comp)
# print("simple: ", word_num_sim)
print('Wiki-doc words per complex sentence: ', word_num_comp/sent_num_comp, 'Wiki-doc words per simple sentence: ', word_num_sim/sent_num_sim)
'''


'''
complex_lens = np.array(complex_lens)
simple_lens = np.array(simple_lens)
orig_complex_lens = np.array(orig_complex_lens)
orig_simple_lens = np.array(orig_simple_lens)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 9))
ax1.hist(orig_complex_lens, density=True, label='Complex Source', bins =40, color = 'lightskyblue')
ax1.hist(orig_simple_lens, density=True, label='Simple Reference', bins =40, color = 'darkorange')
ax1.legend(prop={'size': 31})
ax1.set_yscale('log')
ax1.set_title('Distribution of Wiki-Doc', fontsize=40)
ax1.set_xlabel("Article length", fontsize = 37)
plt.setp(ax1.get_xticklabels(), fontsize=20)
plt.setp(ax1.get_yticklabels(), fontsize=20)
ax1.grid(linestyle='-.')
ax2.hist(complex_lens, density=True, label='Complex Source', bins =40, color = 'lightskyblue')
ax2.hist(simple_lens, density=True, label='Simple Reference', bins =40, color = 'darkorange')
ax2.legend(prop={'size': 31})
ax2.set_yscale('log')
ax2.set_title('Distribution of Wiki-Doc-Match', fontsize=40)
ax2.set_xlabel("Article length", fontsize = 37)
ax2.grid(linestyle='-.')
plt.setp(ax2.get_xticklabels(), fontsize=20)
plt.setp(ax2.get_yticklabels(), fontsize=20)
plt.show()
plt.savefig('compare_dis.pdf', bbox_inches='tight')

### save to PDF
plt.savefig("LensDistribute.pdf", bbox_inches='tight')

print(f'complex mean: {np.mean(complex_lens)}, simple mean: {np.mean(simple_lens)}')
print(f'complex std: {np.std(complex_lens)}, simple std: {np.std(simple_lens)}')
print(f'Len: {len(simple_lens)}')

plt.hist(complex_lens, bins = 30)
plt.hist(simple_lens, bins = 30)
plt.title('distribution')
plt.savefig('dis.png')
'''




