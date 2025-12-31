from unsloth import FastLanguageModel
import torch
from utils.rewards import *
max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 256 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)

# dataset
from datasets import load_dataset
import pandas as pd
import numpy as np

dataset = load_dataset("Dionysianspirit/reddit-movie-rec-sft", split = "train")
# dataset = load_dataset("Dionysianspirit/movie-sensitivity-warnings", split = "train")
# dataset = load_dataset("Dionysianspirit/reddit-movie-entity2id", split = "train")


# def format_dataset(x):
#     expected_answer = x["expected_answer"]
#     problem = x["problem"]

#     # Remove generated <think> and </think>
#     thoughts = x["generated_solution"]
#     thoughts = thoughts.replace("<think>", "").replace("</think>", "")

#     # Strip newlines on left and right
#     thoughts = thoughts.strip()
#     # Add our custom formatting
#     final_prompt = \
#         reasoning_start + thoughts + reasoning_end + \
#         solution_start + expected_answer + solution_end
#     return [
#         {"role" : "system",    "content" : system_prompt},
#         {"role" : "user",      "content" : problem},
#         {"role" : "assistant", "content" : final_prompt},
#     ]

# dataset["Messages"] = dataset.apply(format_dataset, axis = 1)

# import re

# # Add optional EOS token matching
# solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
#     "(?:" + re.escape(tokenizer.eos_token) + ")?"

# match_format = re.compile(
#     rf"{reasoning_end}.*?"\
#     rf"{solution_start}(.+?){solution_end_regex}"\
#     rf"[\s]{{0,}}$",
#     flags = re.MULTILINE | re.DOTALL
# )



dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["prompt"]},
    ],
    "answer": extract_hash_answer(x["solution"]),
    "avoid_content": x["avoid_content"],
    "ground_truth_imdb_ids": x["ground_truth_imdb_ids"],
})

import re

# Add optional EOS token matching
solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
    "(?:" + re.escape(tokenizer.eos_token) + ")?"

match_format = re.compile(
    rf"{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end_regex}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)
match_numbers = re.compile(
    solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)
tokenized = dataset.map(
    lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    batched = True,
)
print(tokenizer.decode(tokenized[0]["tokens"]))


tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})

import numpy as np
maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length = ", maximum_length)

# Filter only samples smaller than 90% max length
dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized

# Split dataset: 300 samples for testing, rest for training
dataset = dataset.train_test_split(test_size=300, seed=3407)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

max_prompt_length = maximum_length + 1 # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = 3407,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    temperature = 1.0,
    learning_rate = 5e-6,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 100,
    save_steps = 100,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",

    # Enable training + evaluation
    fp16_full_eval = True,
    per_device_eval_batch_size = 1,
    eval_accumulation_steps = 1,
    eval_strategy = "steps",
    eval_steps = 50,  # Evaluate every 50 steps
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        calculate_final_reward,  # Main reward: 0.7*Safety + 0.2*Accuracy + 0.1*Coverage
        log_reward_details,      # Logging function for debugging
    ],
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
)
trainer.train()


# text = "What is the sqrt of 101?"

# from vllm import SamplingParams
# sampling_params = SamplingParams(
#     temperature = 1.0,
#     top_k = 50,
#     max_tokens = 1024,
# )
# output = model.fast_generate(
#     [text],
#     sampling_params = sampling_params,
#     lora_request = None,
# )[0].outputs[0].text

model.save_lora("safe_rec_lora")
# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")
