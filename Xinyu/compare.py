from easse.sari import corpus_sari
from preprocessor import WIKILARGE_FILTER_DATASET,TURKCORPUS_DATASET, get_data_filepath, yield_lines,yield_sentence_pair
from rouge_score import rouge_scorer
from tqdm import tqdm
from Ts_T5 import T5FineTuner

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

print("******** LOAD MODEL ********")

model_sent2sent_turk_3loss_3_20 = T5FineTuner.load_from_checkpoint('Xinyu/experiments/exp_turk_3loss_3_20/checkpoint-epoch=1.ckpt')
turk_complex_file = get_data_filepath(TURKCORPUS_DATASET,'valid','complex')
avg_rouge1 = 0
avg_rouge2 = 0
avg_rougeL = 0
avg_sari = 0

for i in range(8):
    turk_simple_file = get_data_filepath(TURKCORPUS_DATASET,'valid','simple.turk',i)
    cnt = 0
    rouge1_score = 0
    rouge2_score = 0
    rougeL_score = 0
    sari_score = 0
    print("******** EVALUATE MODEL ********")
    for (complex_sent, simple_sent) in yield_sentence_pair(complex_file, simple_file):
        
        generation = model_sent2sent_turk_3loss_3_20.generate(complex_sent)
        scores = scorer.score(simple_sent,generation)
        # f-measure
        rouge1_score += scores['rouge1'][2]
        rouge2_score += scores['rouge2'][2]
        rougeL_score += scores['rougeL'][2]
        cnt+=1
        print(cnt)
        if cnt%100==0:
            print("ROUGE-1: ",rouge1_score/cnt)
            print("ROUGE-2: ",rouge2_score/cnt)
            print("ROUGE-L: ",rougeL_score/cnt)
            print("********")

        sari_score += corpus_sari([complex_sent],[generation],[[simple_sent]])

        if cnt%100==0:
            print("SARI: ",sari_score/cnt)

    avg_rouge1 += rouge1_score/cnt
    avg_rouge2 += rouge2_score/cnt
    avg_rougeL += rougeL_score/cnt
    avg_sari += sari_score/cnt

print("SARI score: ",avg_sari/8)
print("ROUGE-1: ",avg_rouge1/8)
print("ROUGE-2: ",avg_rouge2/8)
print("ROUGE-L: ",avg_rougeL/8)

# -------------------------- valid on wikilargeF --------------------------
# complex_file = 'Xinyu/resources/datasets/wikilargeF/wikilargeF.valid.complex'
# simple_file = 'Xinyu/resources/datasets/wikilargeF/wikilargeF.valid.simple'

#SARI score:  33.393714573741406 ROUGE-1:  0.32412501691399914 ROUGE-2:  0.17891711541522262 ROUGE-L:  0.28902915494861536  
#model_sent2sent_new3loss_10epoch_1.2 = T5FineTuner.load_from_checkpoint("experiments/exp_wikilarge_3loss_1.2/checkpoint-epoch=9.ckpt")

# SARI score:  33.14912030694694 ROUGE-1:  0.3269696732463188 ROUGE-2:  0.180208832383295 ROUGE-L:  0.290500700652506
#model_sent2sent_new3loss_20_epoch_100_50 = T5FineTuner.load_from_checkpoint("Xinyu/experiments/exp_wikilarge_3loss_100_50/checkpoint-epoch=5.ckpt")

# SARI score:  33.393714573741406 ROUGE-1:  0.32412501691399914 ROUGE-2:  0.17891711541522262 ROUGE-L:  0.28902915494861536
#model_sent2sent_new3loss_20_epoch_20_50 = T5FineTuner.load_from_checkpoint('Xinyu/experiments/exp_wikilarge_3loss_20_50/checkpoint-epoch=9.ckpt')

# SARI score:  32.82684132212545 ROUGE-1:  0.32899754533374587 ROUGE-2:  0.1831019970771164 ROUGE-L:  0.29030068957074107
#model_sent2sent_new3loss_10_epoch_10_10 = T5FineTuner.load_from_checkpoint('Xinyu/experiments/exp_wikilarge_3loss_10_10/checkpoint-epoch=3.ckpt')

# SARI score:  33.400080744617945 ROUGE-1:  0.3265695395737187 ROUGE-2:  0.17872800976203787 ROUGE-L:  0.2898019255349985
#model_sent2sent_new3loss_10_epoch_5_20 = T5FineTuner.load_from_checkpoint('Xinyu/experiments/exp_wikilarge_3loss_5_20/checkpoint-epoch=9.ckpt')

# SARI score:  33.185781387321484 ROUGE-1:  0.32701327689997073 ROUGE-2:  0.17924288209194994 ROUGE-L:  0.29176576435152896
#model_sent2sent_new3loss_10_epoch_20_20 = T5FineTuner.load_from_checkpoint('Xinyu/experiments/exp_wikilarge_3loss_20_20/checkpoint-epoch=5.ckpt')

# SARI score:  33.52355260672639 ROUGE-1:  0.3291568609764943 ROUGE-2:  0.18073871412525339 ROUGE-L:  0.2914004556052686
#model_sent2sent_new3loss_10_epoch_3_20 = T5FineTuner.load_from_checkpoint('Xinyu/experiments/exp_wikilarge_3loss_3_20/checkpoint-epoch=9.ckpt')

#SARI score:  33.14912030694694 ROUGE-1:  0.3269696732463188 ROUGE-2:  0.180208832383295 ROUGE-L:  0.290500700652506
#model_sent2sent_old_loss_10epoch = T5FineTuner.load_from_checkpoint("experiments/exp_wikilarge_oldloss/checkpoint-epoch=5.ckpt")

#SARI score:  33.06357309962498 ROUGE-1:  0.3278116620424481 ROUGE-2:  0.18016610803805574 ROUGE-L:  0.28854116363325794
#model_sent2sent = T5FineTuner.load_from_checkpoint("experiments/exp_wikilarge_bestfinetune/checkpoint-epoch=1.ckpt")

# SARI score:  26.6340300407876 ROUGE-1:  0.346540717020261 ROUGE-2:  0.19832989053819106 ROUGE-L:  0.3120344541293833
#model_para2para = T5FineTuner.load_from_checkpoint("experiments/exp_wiki_paragh/checkpoint-epoch=0.ckpt")

# SARI score:  30.76722753179715 ROUGE-1:  0.3328172684344919 ROUGE-2:  0.18441470507928764 ROUGE-L:  0.29402484999225503
#model_sent2sent_1epoch = T5FineTuner.load_from_checkpoint("experiments/exp_wikilarge_epflnewsFineTune_1epoch/checkpoint-epoch=0.ckpt")

# SARI score:  31.41731370776929 ROUGE-1:  0.32952106220905697 ROUGE-2:  0.18350543565795302 ROUGE-L:  0.29093801356257165
#model_sent2sent_5epoch = T5FineTuner.load_from_checkpoint("experiments/exp_wikilarge_epflnewsFineTune_5epoch/checkpoint-epoch=4.ckpt")

# SARI score:  27.246785099805155 ROUGE-1:  0.3438854231546133 ROUGE-2:  0.20036068315600603 ROUGE-L:  0.3096993371350689
#model_para2para_1epoch = T5FineTuner.load_from_checkpoint("experiments/exp_wikiparagh_epflnewsFineTune_1epoch/checkpoint-epoch=0.ckpt")

# SARI score:  27.956291273196598 ROUGE-1:  0.3444279112760297 ROUGE-2:  0.19830617791146754 ROUGE-L:  0.3078360074086852
#model_para2para_5epoch = T5FineTuner.load_from_checkpoint("experiments/exp_wikiparagh_epflnewsFineTune_5epoch/checkpoint-epoch=4.ckpt")

# SARI score:  32.88383368490323 ROUGE-1:  0.34797298616756933 ROUGE-2:  0.21591191019642364 ROUGE-L:  0.32356145626398003
#model_sent2sent_no_tokens_10epoch = T5FineTuner.load_from_checkpoint("experiments/exp_wikilarge_no_tokens_10epoch/checkpoint-epoch=8.ckpt")
# print("******** LOAD MODEL DONE ********")
# cnt=0
# rouge1_score = 0
# rouge2_score = 0
# rougeL_score = 0
# score = 0

# for complex_sent, simple_sent in (yield_sentence_pair(complex_file, simple_file)):
#     generation = model_sent2sent_new3loss_10_epoch_3_20.generate(complex_sent)
#     #generation = model_sen2sent.generate(complex_sent)
#     scores = scorer.score(simple_sent,generation)
#     # f-measure
#     rouge1_score += scores['rouge1'][2]
#     rouge2_score += scores['rouge2'][2]
#     rougeL_score += scores['rougeL'][2]
#     cnt+=1
#     print(cnt)
#     if cnt%100==0:
#         print("ROUGE-1: ",rouge1_score/cnt)
#         print("ROUGE-2: ",rouge2_score/cnt)
#         print("ROUGE-L: ",rougeL_score/cnt)
#         print("********")

#     score += corpus_sari([complex_sent],[generation],[[simple_sent]])
#     # cnt+=1
#     # print(cnt)
#     if cnt%100==0:
#         print("SARI: ",score/cnt)
# avg = score/cnt
# print("SARI score: ",avg)
# print("ROUGE-1: ",rouge1_score/cnt)
# print("ROUGE-2: ",rouge2_score/cnt)
# print("ROUGE-L: ",rougeL_score/cnt)

