# Mistral-finetune

<a target="_blank" href="https://colab.research.google.com/github/mistralai/mistral-finetune/blob/main/tutorials/mistral_finetune_7b.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


`mistral-finetune` is a light-weight codebase that enables memory-efficient and performant finetuning of Mistral's models.
It is based on [LoRA](https://arxiv.org/abs/2106.09685), a training paradigm where most weights are frozen and only 1-2% of additional weights in the form of low-rank matrix perturbations are trained. 

For maximum efficiency it is recommended to use an A100 or H100 GPU. The codebase is optimized 
for multi-GPU-single-node training setups, but for smaller models, such as the 7B a single GPU suffices.

> **Note**
> 
> - The goal of this repository is to provide a simple, guided entrypoint to finetune Mistral models.
> As such, it is fairly opinionated (especially around data formatting) and does not aim at being exhaustive
> across multiple model architectures or hardware types.
> For more generic approaches, you can check out some other great projects like 
> [torchtune](https://pytorch.org/torchtune/stable/overview.html).


## News

- `mistral-finetune` is now compatible with Mistral Nemo! 
  - 1. Download the new checkpoints [here](##model-download) and set `model_id_or_path` to the new checkpoint
  - 2. Fine-tuning Mistral-Nemo requires currently much more memory due to a larger vocabulary size which spikes the peak memory requirement of the CE loss (we'll soon add an improved CE loss here). For now set `seq_len` to 16384 or 8192
  - 3. It is recommended to use the same hyperparameters as for the 7B v3.

## Installation

To get started with Mistral LoRA fine-tuning, follow these steps:

1. Clone this repository:
```
cd $HOME && git clone https://github.com/mistralai/mistral-finetune.git
```

2. Install all required dependencies:
```
cd mistral-finetune
pip install -r requirements.txt
```

## Model download

We recommend fine-tuning one of the official Mistral models which you can download here:

| Model          | Link                                                                                                    | Checksum                          |
|----------------|---------------------------------------------------------------------------------------------------------|-----------------------------------|
| 7B Base V3       | [7B Base](https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-v0.3.tar)                            | `0663b293810d7571dad25dae2f2a5806`|
| 7B Instruct v3 | [7B Instruct v3](https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-Instruct-v0.3.tar)             | `80b71fcb6416085bcb4efad86dfb4d52`|
| 8x7B Base V1   | [8x7B Base](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)                                                                        | (HF link)                                |
| 8x7B Instruct V1 | [8x7B Instruct](https://models.mistralcdn.com/mixtral-8x7b-v0-1/Mixtral-8x7B-v0.1-Instruct.tar) | `8e2d3930145dc43d3084396f49d38a3f` |
| 8x22 Instruct V3 | [8x22 Instruct](https://models.mistralcdn.com/mixtral-8x22b-v0-3/mixtral-8x22B-Instruct-v0.3.tar)        | `471a02a6902706a2f1e44a693813855b`|
| 8x22B Base V3  | [8x22B Base](https://models.mistralcdn.com/mixtral-8x22b-v0-3/mixtral-8x22B-v0.3.tar)                        | `a2fa75117174f87d1197e3a4eb50371a`|
| 12B Instruct | [Mistral-Nemo Instruct](https://models.mistralcdn.com/mistral-nemo-2407/mistral-nemo-instruct-2407.tar) | `296fbdf911cb88e6f0be74cd04827fe7` |
| 12B Base | [Mistral-Nemo Base](https://models.mistralcdn.com/mistral-nemo-2407/mistral-nemo-base-2407.tar) | `c5d079ac4b55fc1ae35f51f0a3c0eb83` |

**Important Notice**: For 8x7B Base V1 and 8x7B Instruct V1, it is necessary to use our v3 tokenizer and extend the vocabulary size to 32768 prior to fine-tuning. For detailed instructions on this process, please refer to the ["Model extension"](https://github.com/mistralai/mistral-finetune?tab=readme-ov-file#model-extension) section. 

E.g., to download the 7B-base model you can run the following command:
```sh
mkdir -p ~/${HOME}/mistral_models
cd ${HOME} && wget https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-v0.3.tar
tar -xf mistral-7B-v0.3.tar -C mistral_models
```

Make sure to modify your training script and add the path to the downloaded 
folder as `model_id_or_path`.

E.g., modify [example/7B.yaml](https://github.com/mistralai/mistral-finetune/blob/main/example/7B.yaml) to include the absolute path to `$HOME/mistral_models/7B`:

```
model_id_or_path: "/Users/johndoe/mistral_models/7B"
```

## Prepare dataset 

To ensure effective training, `mistral-finetune` has strict 
requirements for how the training data has to be formatted.

All data files must be stored in jsonl format files.

You can build two types of data files:

### _Pretrain_:

Pretrain data corresponds to plain text data stored in the `"text"` key. E.g:

```jsonl
{"text": "Text contained in document n°1"}
{"text": "Text contained in document n°2"}
```

### _Instruct_:

Currently two different types of instruction following data are supported:

- _Instruct_: conversational data stored in the `"messages"` key in the form of a list. Each list item is a dictionary containing the `"content"` and `"role"` keys. `"role"` is a string being one of "user", "assistant" or "system". The loss will only be computed if "role" == "assistant". E.g.:

```jsonl
{
  "messages": [
    {
      "role": "user",
      "content": "User interaction n°1 contained in document n°1"
    },
    {
      "role": "assistant",
      "content": "Bot interaction n°1 contained in document n°1"
    },
    {
      "role": "user",
      "content": "User interaction n°2 contained in document n°1"
    },
    {
      "role": "assistant",
      "content": "Bot interaction n°2 contained in document n°1"
    }
  ]
}
{
  "messages": [
    {
      "role": "user",
      "content": "User interaction n°1 contained in document n°2"
    },
    {
      "role": "assistant",
      "content": "Bot interaction n°1 contained in document n°2"
    },
    {
      "role": "user",
      "content": "User interaction n°2 contained in document n°2"
    },
    {
      "role": "assistant",
      "content": "Bot interaction n°2 contained in document n°2",
      "weight": 0,  # don't train on n°2
    },
    {
      "role": "user",
      "content": "User interaction n°3 contained in document n°2"
    },
    {
      "role": "assistant",
      "content": "Bot interaction n°3 contained in document n°2"
    }
  ]
}
```

- _Function calling_: conversational data stored in the `"messages"` key in the form of a list. Each list item is a dictionary containing the `"role"` and `"content"` or `"tool_calls"` keys. `"role"` is a string being one of "user", "assistant", "system", or "tool". The loss will only be computed if "role" == "assistant".

**Note**: In function calling the `"id"` of `"tool_calls"` and the `"tool_call_id"` are randomly generated strings of exactly 9 chars. We recommend to generate this automatically 
in a data preparation script as is done [here](https://github.com/mistralai/mistral-finetune/blob/208b25c0f7299bb78d06cea25b82adee03834319/utils/reformat_data_glaive.py#L74).

E.g.:

```jsonl
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant who has access to the following functions to help the user, you can use the functions if needed"
    },
    {
      "role": "user",
      "content": "Can you help me generate an anagram of the word \"listen\"?"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "TX92Jm8Zi",
          "type": "function",
          "function": {
            "name": "generate_anagram",
            "arguments": "{\"word\": \"listen\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "{\"anagram\": \"silent\"}",
      "tool_call_id": "TX92Jm8Zi"
    },
    {
      "role": "assistant",
      "content": "The anagram of the word \"listen\" is \"silent\"."
    },
    {
      "role": "user",
      "content": "That's amazing! Can you generate an anagram for the word \"race\"?"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "3XhQnxLsT",
          "type": "function",
          "function": {
            "name": "generate_anagram",
            "arguments": "{\"word\": \"race\"}"
          }
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "generate_anagram",
        "description": "Generate an anagram of a given word",
        "parameters": {
          "type": "object",
          "properties": {
            "word": {
              "type": "string",
              "description": "The word to generate an anagram of"
            }
          },
          "required": [
            "word"
          ]
        }
      }
    }
  ]
}
```

## Verify dataset

Before starting a training run you should verify that your dataset is correctly formatted and get an 
estimation of the training time. You can do so by using the [./utils/validate_data](https://github.com/mistralai/mistral-finetune/blob/main/utils/validate_data.py) script.

Note that this step is crucial to ensure that the data is correctly formatted.

### Instruction following

Let's go over a simple example to train a model in instruction following:

- 1. **Load a chunk of [Ultachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)**

Create the data folder and navigate to the folder.
```sh
cd $HOME && mkdir -p data && cd $HOME/data
```

Load the data into a Pandas Dataframe. 

**Note**: Make sure to have pandas and pyarrow installed (`pip install pandas pyarrow`).

```py
import pandas as pd

df = pd.read_parquet('https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/main/data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet')
```
- 2. Split into train and eval

```py
df_train=df.sample(frac=0.95,random_state=200)
df_eval=df.drop(df_train.index)
```

- 3. Save data to jsonl

```py
df_train.to_json("ultrachat_chunk_train.jsonl", orient="records", lines=True)
df_eval.to_json("ultrachat_chunk_eval.jsonl", orient="records", lines=True)
```

- 4. Modify your training yaml to include the ultrachat dataset and verify the yaml

Modify [example/7B.yaml](https://github.com/mistralai/mistral-finetune/blob/main/example/7B.yaml) to include the absolute path to `$HOME/data/ultrachat_chunk_train.jsonl` as well as a dataset mixing weight for training and `$HOME/data/ultrachat_chunk_eval.jsonl` for eval, *e.g.*

```
data:
  instruct_data: "/Users/johndoe/data/ultrachat_chunk_train.jsonl"
  eval_instruct_data: "/Users/johndoe/data/ultrachat_chunk_eval.jsonl"
```

Now you can verify your training yaml to make sure the data is correctly formatted and to get an estimate of your training time.

```
cd $HOME/mistral-finetune
python -m utils.validate_data --train_yaml example/7B.yaml
```

Upon completion you should see an error report with many of the following errors:

```
The data in line 1412 of dataset /Users/johndoe/data/ultrachat_chunk_eval.jsonl is incorrectly formatted. Expected last role to be one of: [assistant] but got user
The data in line 1413 of dataset /Users/johndoe/data/ultrachat_chunk_eval.jsonl is incorrectly formatted. Expected last role to be one of: [assistant] but got user
The data in line 1414 of dataset /Users/johndoe/data/ultrachat_chunk_eval.jsonl is incorrectly formatted. Expected last role to be one of: [assistant] but got user
The data in line 1415 of dataset /Users/johndoe/data/ultrachat_chunk_eval.jsonl is incorrectly formatted. Expected last role to be one of: [assistant] but got user
```

Many conversations seem to end with the 'user' role which is unnecessary as we only train on 'assistant' messages and thus would unnecessarily process data.

You can make use of [./utils/reformat_data.py](https://github.com/mistralai/mistral-finetune/blob/main/utils/reformat_data.py) to correct the data:

```
cd $HOME/mistral-finetune
python -m utils.reformat_data $HOME/data/ultrachat_chunk_train.jsonl
python -m utils.reformat_data $HOME/data/ultrachat_chunk_eval.jsonl
```

You should see that a couple of samples will be skipped.

- 5. Potentially change number of training steps

Upon correction of the dataset, run the script again

```
cd $HOME/mistral-finetune
python -m utils.validate_data --train_yaml example/7B.yaml
```

You should get a summary of the data input and training parameters:

```
Train States
 --------------------
{
   "expected": {
       "eta": "00:52:44",
       "data_tokens": 25169147,
       "train_tokens": 131072000,
       "epochs": "5.21",
       "max_steps": 500,
       "data_tokens_per_dataset": {
           "/Users/johndoe/data/ultrachat_chunk_train.jsonl": "25169147.0"
       },
       "train_tokens_per_dataset": {
           "/Users/johndoe/data/ultrachat_chunk_train.jsonl": "131072000.0"
       },
       "epochs_per_dataset": {
           "/Users/johndoe/data/ultrachat_chunk_train.jsonl": "5.2"
       }
   },
}
```

Having `max_steps` set to 500 would lead to iterating through the dataset roughly 5 times which is reasonable, but might 
be a bit too much. A recommended setting is shown below which would only take 30min on a 8xH100 cluster.

### Function calling

Next let's go over a more advanced use case to fine-tune a model on function calling.
Function calling requires the data to be in the format as [explained above](#instruct). Let's go over an example.

- 1. **Load a chat-formatted version of the [Glaive function calling dataset](https://huggingface.co/datasets/Locutusque/function-calling-chatml)**

Create the data folder and navigate to the folder.
```sh
cd $HOME && mkdir -p data && cd $HOME/data
```

Load the data into a Pandas Dataframe.

**Note**: Make sure to have pandas and pyarrow installed (`pip install pandas pyarrow`).

```py
import pandas as pd

df = pd.read_parquet('https://huggingface.co/datasets/Locutusque/function-calling-chatml/resolve/main/data/train-00000-of-00001-f0b56c6983b4a78f.parquet')
```
- 2. Split into train and eval

```py
df_train=df.sample(frac=0.95,random_state=200)
df_eval=df.drop(df_train.index)
```

- 3. Save data to jsonl

```py
df_train.to_json("glaive_train.jsonl", orient="records", lines=True)
df_eval.to_json("glaive_eval.jsonl", orient="records", lines=True)
```

- 4. Reformat dataset

As one can see the dataset does not follow the required function calling format, so it will need to be reformatted. Among other things `"from"` should be renamed to `"user"` and superfluous `"\n"` characters should be removed.
For this dataset you can make use of [`./utils/reformat_data_glaive.py`](https://github.com/mistralai/mistral-finetune/blob/main/utils/reformat_data_glaive.py):

```
cd $HOME/mistral-finetune
python -m utils.reformat_data_glaive $HOME/data/glaive_train.jsonl
python -m utils.reformat_data_glaive $HOME/data/glaive_eval.jsonl
```

Running this command will make sure that most samples are in the correct format.

**Note**: It is impossible to write reformatting scripts that work for all kinds of datasets. 
If you have datasets that don't yet follow the required format above, you will most probably have to 
create a reformatting script yourself (mistral-chat or chat-gpt is your best friend here!).

- 5. Validate dataset

You can now validate the dataset by setting `data.instruct_data` and `data.eval_instruct_data` to
`$HOME/data/glaive_train.jsonl` and `$HOME/data/glaive_eval.jsonl` in `example/7B.yaml` respectively.

The reformatted datasets still have some errors which can be removed with `--create_corrected`. For this, make sure to add
`--create_corrected` as follows:

```
cd $HOME/mistral-finetune
python -m utils.validate_data --train_yaml example/7B.yaml --create_corrected
```

Running this command will show a couple of errors and save two new datasets `$HOME/data/glaive_train.jsonl.corrected` and `$HOME/data/glaive_eval.jsonl.corrected`. Make sure to use these two dataset in `example/7B.yaml` and run the command again. Now the dataset should be correctly formatted!


## Start training

Having followed the [dataset verification section](#verify-dataset), we can now start training.
For faster training, we recommend setting max_steps to only 300. Make sure to define `run_dir` to your experiment folder and optionally set `wandb_project` to a Weights & Biases project for logging`, *e.g.*:
```
max_steps: 300
run_dir: "/Users/johndoe/ultra_chat_test"
wandb.project: ultra_chat
```

Optionally you can also set `wandb`

Save the training configuration and start training! Make sure to set `--nproc-per-node` to the number of available GPUs.

```
cd $HOME/mistral-finetune
torchrun --nproc-per-node 8 --master_port $RANDOM -m train example/7B.yaml
```

Training on ultra-chat should take around 30min on a 8xH100 node and the resulting weights should give an MT Bench score around 6.3.

Training on glaive should take around 1h on a 8xH100 node and the resulting weights should work nicely for function calling.

## Customizing training configuration

The example `mistral-finetune/examples/7B` defines reasonable parameters for learning rate, weight decay, etc... but you are advised to 
customize these settings for your use case.

Generally, a training configuration should fill the following parameters:

- `model_id_or_path` defines the model to start training from. This can be a path to a pre-trained model or a local model directory.
- `run_dir` defines the directory where training checkpoints and metrics are stored.
- `seq_len` defines the sequence length for training. This is the maximum length of input sequences the model will process. Samples are packed to reach a length of `seq_len` for maximum training efficiency.
- `batch_size` defines the number of training examples used per GPU. **Note**: The overall effective batch_size (in tokens) across all GPUs equals `num_gpus` x `batch_size` x `seq_len`.
- `max_steps` defines the maximum number of training steps. This is the total number of iterations the training process will run. It can be adjusted based on the specific needs of your training scenario. Total number of tokens seen during training is `max_steps` x `num_gpus` x `batch_size` x `seq_len`.
- `optim.lr` defines the learning rate. This is the initial learning rate for the optimizer.
- `optim.weight_decay` defines weight decay. Weight decay is a regularization technique used to prevent overfitting by penalizing large weights. We recommend leaving it at 0.1.
- `optim.pct_start` defines the percentage of the total training steps used for the learning rate warm-up phase before it starts to decrease. It corresponds to pct_start of PyTorch's OneCycleLR.
- `lora.rank` defines the size of the LoRA (Low-Rank Adaptation) adapters. We recommend 64 or less, which adjusts the rank of the low-rank decomposition used in LoRA.
- `seed` defines the random seed for initialization and data shuffling/sampling. Setting a seed ensures reproducibility of results.
- `log_freq` defines the logging frequency. This specifies how often (in steps) to log training metrics.
- `data.instruct_data` is the path to the instruction data used for training. This field has to be filled with one or multiple data sources in the format as explained above. Each data source should either be a path to a jsonl file or a path to a directory containing jsonl files followed by a weighting to define the importance of this dataset: `<path/to/data_source>:<weight>`. E.g.: `data.instruct_data: "/path/to/data1.jsonl:5.,/path/to/data2.jsonl:1.,/path/to/dir_of_jsonls:1."`
- `data.data` is an optional path to additional pretraining data in the format as explained above. Note that this field can be left blank.
- `data.eval_instruct_data` is an optional path to evaluation instruction data to run cross-validation at every `eval_freq` steps. Cross-validation metrics are displayed as `loss` and `perplexity`.
- `eval_freq` defines how often (in steps) to evaluate the model. This specifies the interval at which the model is evaluated on the validation set.
- `no_eval` is a flag to enable or disable intermediate evaluation. Setting it to False enables periodic evaluation during training.
- `ckpt_freq` defines how often (in steps) to save checkpoints. This specifies the interval at which the model's state is saved.
- `save_adapters` defines whether to only save the trained LoRA checkpoints or whether the trained LoRA should directly be merged into the base model and saved. **Note**: When setting `save_adapters=False` make sure that you have enough CPU and GPU memory to save the full model on a single process (this is usually only possible for the 7B model).
- `wandb.key` is used to pass your Weights & Biases (wandb) API key for logging. This allows you to log training metrics to the wandb dashboard.
- `wandb.project` defines the wandb project name. This is where the training run will be logged in the wandb interface.

## Inference

Once your model is trained, you should try it out in inference. We recommend using [mistral-inference](https://github.com/mistralai/mistral-inference). 

Make sure to have `mistral_inference` correctly installed:
```
pip install mistral_inference
```

Assuming your `lora.safetensors` is saved under `$HOME/ultra_chat_test/checkpoints/checkpoint_000300/consolidated/lora.safetensors`, you can chat with the model using `mistral_inference`, *e.g.*:

```sh
mistral-chat /mnt/slow/runs/patrick/mistral-finetune/7B/ --max_tokens 256 --temperature 1.0 --instruct --lora_path $HOME/ultra_chat_test/checkpoints/checkpoint_000300/consolidated/lora.safetensors
```

## Adding Weights and Biases (wandb) Support

We have added explicit support for [Weights and Biases](https://www.wandb.com/) to help you monitor and visualize your training runs. This integration allows you to log various metrics and track experiments easily.

### Setting Up Weights and Biases

To use Weights and Biases with `mistral-finetune`, follow these steps:

1. **Install Weights and Biases:**

   Make sure you have the `wandb` library installed. You can install it using pip:

```sh
   pip install wandb
```
### Viewing Your Logs

Once the training starts, you can monitor the progress in real-time by visiting your wandb project dashboard. All metrics, including training loss, evaluation loss, learning rate, etc., will be logged and visualized.

For more details on how to use wandb, visit the [Weights and Biases documentation](https://docs.wandb.ai/).

## Model extension

**Important**: Note that one can only fine-tune mistral models that are compatible with the v3 tokenizer which entails that the models have a vocabulary size of 32768 - not 32000. One can however easily extend older version of vocabulary size 32000 to have a vocabulary size of 32768 by using:
```
python -m utils.extend_model_vocab --original_model_ckpt /folder/to/old/model --extended_model_ckpt /folder/to/extended/model
```

Once the extension has worked, one can fine-tune using the newly created model checkpoint in `/folder/to/extended/model`.

## FAQ:

> - What's the best practice of fine-tuning MoEs?

We see a higher degree of performance variance in when fine-tuning MoE models. It's not unusual to find that fine-tuning MoE models with different seeds can lead to a high variance in performance. We did not observe such a high variance with dense models. Therefore, we suggest running multiple instances of the same fine-tuning process on MoEs models and selecting the one that performs best.

> - How can I determine the number of tokens used during the model training process?
  
You can use the following script to find out: https://github.com/mistralai/mistral-finetune/blob/main/utils/validate_data.py. This script accepts a .yaml training file as input and returns the number of tokens the model is being trained on.

> - What should I do if I encounter a CUDA out-of-memory error?
  
One possible solution is to reduce the batch size per GPU. The batch size is equal to `seq_len` x `batch_size`. Try setting `batch_size` to 1 and reduce `seq_len`. You can define the `batch_size` and `seq_len` in the .yaml file.
