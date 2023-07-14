# https://simpletransformers.ai/docs/lm-minimal-start/#fine-tuning-an-electra-model
# 0.61.5 version
# Provided 

# %%
import logging
import os
from pretrain import get_dataset, split_dataset



from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)


# %%
BEST_MODEL_DIR = "./electra_full/best_model/"
CACHE_DIR = "./cache_dir_electra_full"
OUTPUTS_DIR = "./electra_full/"
TENSORBOARD_DIR = "./electra_full/tb_logs/"

TRAIN_FILE = "./train.txt"
VAL_FILE = "./eval.txt"


# pick a GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# %%
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# %%
model_args = LanguageModelingArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = False
model_args.num_train_epochs = 3 # set to match other models
model_args.dataset_type = "simple"

model_args.evaluate_during_training = False
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_verbose = True
model_args.manual_seed = 4242

model_args.best_model_dir = BEST_MODEL_DIR
model_args.output_dir = OUTPUTS_DIR
model_args.cache_dir = CACHE_DIR
model_args.tensorboard_dir = TENSORBOARD_DIR

# %%
model = LanguageModelingModel(
    model_type="electra",
    model_name="electra",
    discriminator_name="classla/bcms-bertic",
    generator_name="classla/bcms-bertic-generator",
    args=model_args,
)

# %%
model.train_model(TRAIN_FILE, eval_file=VAL_FILE)

# %%
result = model.eval_model(VAL_FILE)
