# SIMSUM: Document-level Text Simplification via Simultaneous Summarization
This repo is the codes for Paper **SIMSUM: Document-level Text Simplification via Simultaneous Summarization** (ACL2023)

![](/fig/WechatIMG14.png)

## Installation
The required packages can be installed by

```
pip install -r requirements.txt
```

## Processed Dataset
The datasets used for document-level simplification are listed in `/SimSum/data` , named `D-Wiki` for D-Wikipedia and `wiki_doc` for WikiDoc.
![](/fig/WechatIMG10.png)

## Training
![](/fig/WechatIMG9.png)
To train the model:
```
python main.py
```

`Bart2.py` and `T5_2.py` means our SimSum model with BART and T5 as the backbone. For the single model, use the `Bart_baseline_finetuned.py` and `T5_baseline_finetuned.py`.


## Automatic Evaluation
To evaluate the model,
```
python evaluate.py
```

which will compute the SARI, D-SARI, BLEU and FKGL score.
![](/fig/WechatIMG11.png)

## Human Evaluation
![](/fig/WechatIMG13.png)