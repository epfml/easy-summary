# -- fix path --
import optparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
# -- end fix path --
import numpy as np 
import pandas as pd
from preprocessor import EXP_DIR, DATASETS_DIR, D_WIKI_MATCH, WIKI_DOC_MATCH


dataset = "wiki_doc_match"

STORE_PATH = DATASETS_DIR / 'samples2'

complex_path = DATASETS_DIR / WIKI_DOC_MATCH / "wiki_doc_match.test.complex"
df_src = pd.read_fwf(complex_path, header = None)
src_store_path = STORE_PATH / 'src.txt'

T5_Single_Path = EXP_DIR / 'exp_WikiDocMatch_T5Single'
T5_Single_Res_Path = T5_Single_Path / 'outputs' / 'wiki_doc_match.test.txt'
df_T5Single = pd.read_fwf(T5_Single_Res_Path, header = None)
T5_Single_store_path = STORE_PATH / 'T5Single.txt'

BART_Single_Path = EXP_DIR / 'exp_WikiDocMatch_BARTSingle'
BART_Single_Res_Path = BART_Single_Path / 'outputs/wiki_doc_match.test.txt'
df_BARTSingle = pd.read_fwf(BART_Single_Res_Path, header = None)
BART_Single_store_path = STORE_PATH / 'BARTSingle.txt'

BRIO_Path = EXP_DIR / 'exp_WikiDocMatch_BRIO'
BRIO_Res_Path = BRIO_Path / 'outputs/wiki_doc_match.test.txt'
df_BRIO = pd.read_fwf(BRIO_Res_Path, header = None)
BRIO_store_path = STORE_PATH / 'BRIO.txt'

MUSS_Path = EXP_DIR /'exp_WikiDocMatch_MUSS'
MUSS_Res_Path = MUSS_Path / 'outputs/wiki_doc_match.test.txt'
df_MUSS = pd.read_fwf(MUSS_Res_Path, header = None)
MUSS_store_path = STORE_PATH / 'MUSS.txt'

T5_joint_Path = EXP_DIR / 'exp_WikiDocMatch_T5'
T5_joint_Res_Path = T5_joint_Path / 'outputs/wiki_doc_match.test.txt'
df_T5_joint = pd.read_fwf(T5_joint_Res_Path, header = None)
T5Joint_store_path = STORE_PATH / 'T5Joint.txt'

BART_joint_Path = EXP_DIR / 'exp_WikiDocMatch_BART'
BART_joint_Res_Path = BART_joint_Path / 'outputs/wiki_doc_match.test.txt'
df_BART_joint = pd.read_fwf(BART_joint_Res_Path, header = None)
BARTJoint_store_path = STORE_PATH / 'BARTJoint.txt'

T5_joint_kw_num4_div9_Path = EXP_DIR / "exp_WikiDocMatch_T5_kw_num4_div0.9"
T5_joint_kw_num4_div9_Res_Path = T5_joint_kw_num4_div9_Path / 'outputs/wiki_doc_match.test.complex_kw_num4_div0.txt'
df_T5_joint_kw_num4_div9 = pd.read_fwf(T5_joint_kw_num4_div9_Res_Path, header = None)
T5Joint_kw_num4_div9_store_path = STORE_PATH / 'T5Joint_kw_num4_div9.txt'

TOTAL_NUM = 50

ids = np.random.choice(range(len(df_T5_joint)), size = TOTAL_NUM, replace = False)
print(type(list(ids)))

select_T5_single = df_T5Single.iloc[list(ids)]
select_BART_single = df_BARTSingle.iloc[list(ids)]
select_BRIO = df_BRIO.iloc[list(ids)]
select_MUSS = df_MUSS.iloc[list(ids)]
select_T5Joint = df_T5_joint.iloc[list(ids)]
select_BARTJoint = df_BART_joint.iloc[list(ids)]
select_T5Joint_kw_score = df_T5_joint_kw_num4_div9.iloc[list(ids)]
select_src = df_src.iloc[list(ids)]

select_T5_single[0].to_csv(T5_Single_store_path)
select_BART_single[0].to_csv(BART_Single_store_path)
select_BRIO[0].to_csv(BRIO_store_path)
select_MUSS[0].to_csv(MUSS_store_path)
select_T5Joint[0].to_csv(T5Joint_store_path)
select_BARTJoint[0].to_csv(BARTJoint_store_path)
select_T5Joint_kw_score[0].to_csv(T5Joint_kw_num4_div9_store_path)
select_src[0].to_csv(src_store_path)






