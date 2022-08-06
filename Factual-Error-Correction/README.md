# Factual Error Correction for Abstractive Summarization Models
This directory contains code necessary to replicate the training for this paper ["Factual Error Correction for Abstractive Summarization Models"](https://arxiv.org/abs/2010.08712).

## Directory Structure

The code is organized into three subdirectories:

* `build_dataset`: code for building the aritificial trianing & validation dataset.
* `data_epfl_news`: raw and preprocessed EPFL news dataset.
* `model`: contains everything for BART model for training and testing.

And the enviroment file `fec_env.yml`.

## Build Dataset
EPFL news dataset is already built and could be used without any additional preprocessing.

To build CNN dailymail dataset first run:

```
cd build_dataset
sh create_data.sh
```

Then, go to `build_data/dataset.ipynb` and run the part which is about mixing corrupted and clean data in right proportion.

## Model Training
To train the BART model use:

```
cd model
python main.py --model_type=bart --model_name_or_path=facebook/bart-large-cnn --data_dir=../data_epfl_news
```

You can also check other parameters and modify them.

## Model Testing
To test the BART model use:

```
cd model
python test.py --model_type=bart --model_name_or_path=facebook/bart-large-cnn --data_dir=../data_epfl_news --use_pretrained=True --checkpoint=../results/your_data/model/epoch_2/
```

## Future work
* The main problem of this model is that it generates some noise after the summary. Also it make factual errors, even if it corrects the ones that were added intentionally.
* There is no specific metric to measure how accurate model corrects the errors
