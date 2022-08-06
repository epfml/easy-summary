## Text Simplification using T5 with special tokens

### 1. Training
run `main.py` file to fine-tune the T5 model. You can also change the dataset for fine-tuning stage

```python
python main.py
```
### 2. Evaluation
You can evaluate the model's performance on Turkcorpus and ASSET test dataset by running the following command:
```python
python evaluate.py
```
It will output the performance on SARI, BLEU, and FKGL scores.

### 3. Checkpoint files

fine-tuned model checkpoint on WikiParagraph dataset is saved in `https://drive.google.com/file/d/13dQFvC72pG7F_4A_BD6e7ZCNKUb8rfIK/view?usp=sharing`



