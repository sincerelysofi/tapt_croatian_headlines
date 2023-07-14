# TakeLab Retriever - TAPT and Sentiment analysis

This set of scripts performs task-adaptive pre-training using a headlines dataset, and performs some statistical analysis on the dataset. 

## Preparation

If you plan on fine-tuning models with TAPT, please run pre-training scripts first.

All fine-tuning code requires an `articles_titles_all.csv` file in the `./pretrain` directory. This is a csv file containing TakeLab Retriever headlines.

Additionally, please include SToNe data set files as needed in `./finetune/data/raw`.

Finally, ensure you have the proper modules installed in your environment by running:

`$ pip install -r requirements.txt`

## Usage

There are a number of different components to this code.

### Pre-training

All pre-training code can be run at once using:
`$ ./pretrain/pretrain.sh'

However, you can also request specific models using:

`$ torchrun ./pretrain/pretrain.py -model MODEL_NAME_HERE`

with `MODEL_NAME_HERE` being the model's name on Huggingface.

This will produce models that will be placed in the `./models` directory.

#### BERTić

BERTić has some particular quirks which require separate scripts. BERTić pre-training can be run via:

`$ python ./pretrain/bertic_pretrain.py`

The token distribution for BERTić can also be acquired by running:

`$ python ./pretrain/get_token_distribution`

### Fine-tuning and evaluation

Fine-tuning and evaluation can be performed by running the `train.sh` script as follows:

`$ ./finetune/scripts/train.sh`

This script will automatically cycle through all the models, training statuses, and seeds. Please note that there should be models in the `./models` directory if you plan on using pre-trained models.

