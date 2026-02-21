import shutil, os
import sys
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unsloth import FastVisionModel
import torch
from transformers import TextStreamer, EarlyStoppingCallback
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from dataset import get_data_loaders
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

"""cache_dir = os.path.join(os.path.dirname(__file__), "unsloth_compiled_cache")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)"""

# 1. Load the model
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = True,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

# 2. Load the dataset
cleaned_dataset_dir = os.environ.get("DATASET_DIR", "data/cleaned_dataset")
output_excel_file = os.environ.get("METADATA_FILE", "data/cleaned_dataset_metadata.xlsx")
train_loader, val_loader, test_loader = get_data_loaders(output_excel_file, cleaned_dataset_dir, batch_size=4, drop_last=True)
train_loader, val_loader, test_loader = accelerator.prepare(train_loader, val_loader, test_loader)

instruction = "You are an expert radiographer. Describe accurately what you see in this chest x-ray image. Write it as you write the findings section of the report."

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["PA_path"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["Findings"]} ]
        },
    ]
    return { "messages" : conversation }
pass

# Cache converted datasets to avoid re-converting every run
import pickle

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

train_cache = os.path.join(CACHE_DIR, "converted_train.pkl")
val_cache = os.path.join(CACHE_DIR, "converted_val.pkl")
test_cache = os.path.join(CACHE_DIR, "converted_test.pkl")

if os.path.exists(train_cache) and os.path.exists(val_cache):
    print("Loading cached datasets...")
    with open(train_cache, "rb") as f:
        converted_train_dataset = pickle.load(f)
    with open(val_cache, "rb") as f:
        converted_val_dataset = pickle.load(f)
    if os.path.exists(test_cache):
        with open(test_cache, "rb") as f:
            converted_test_dataset = pickle.load(f)
    print(f"Loaded {len(converted_train_dataset)} train, {len(converted_val_dataset)} val samples from cache")
else:
    print("Converting datasets (will be cached for future runs)...")
    converted_train_dataset = [convert_to_conversation(sample) for sample in tqdm(train_loader.dataset, desc="Converting train")]
    converted_val_dataset = [convert_to_conversation(sample) for sample in tqdm(val_loader.dataset, desc="Converting val")]
    converted_test_dataset = [convert_to_conversation(sample) for sample in tqdm(test_loader.dataset, desc="Converting test")]

    # Save cache
    with open(train_cache, "wb") as f:
        pickle.dump(converted_train_dataset, f)
    with open(val_cache, "wb") as f:
        pickle.dump(converted_val_dataset, f)
    with open(test_cache, "wb") as f:
        pickle.dump(converted_test_dataset, f)
    print("Datasets cached for future runs")
# 3. Before training

FastVisionModel.for_inference(model)

first_batch = next(iter(train_loader))
image_batch = first_batch["PA"]
finding_batch = first_batch["Findings"]

pil_image = to_pil_image(image_batch[0])

instruction = "You are an expert radiographer. Describe accurately what you see in this chest x-ray image. Write it as you write the findings section of the report."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    pil_image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

if accelerator.is_main_process:
    print("\nBefore training:\n")

    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 1.5, min_p = 0.1)

    print("\nActual Findings: ", finding_batch)


# 4. Training
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_train_dataset,
    eval_dataset=converted_val_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 4,
        num_train_epochs = 10,
        eval_strategy = "epoch",
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
        save_strategy = "epoch",
        save_total_limit = 2,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
    ),
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
if accelerator.is_main_process:
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()
# Print training metrics (including loss)
if accelerator.is_main_process:
    print("Training metrics:", trainer_stats.metrics)

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
if accelerator.is_main_process:
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# Evaluate on the validation set after training (if not already done)
eval_results = trainer.evaluate()
if accelerator.is_main_process:
    print("Evaluation results:", eval_results)

# Save the final model and tokenizer only on the main process
if accelerator.is_main_process:
    print("Saving the final model and tokenizer...")
    trainer.save_model("outputs/final_model")
    tokenizer.save_pretrained("outputs/final_model")

# 5. After training

FastVisionModel.for_inference(model)
first_batch = next(iter(train_loader))
image_batch = first_batch["PA"]
finding_batch = first_batch["Findings"]
pil_image = to_pil_image(image_batch[0])

first_batch2 = next(iter(test_loader))
image_batch2 = first_batch2["PA"]
finding_batch2= first_batch2["Findings"]
pil_image2 = to_pil_image(image_batch2[0])

instruction = "You are an expert radiographer. Describe accurately what you see in this chest x-ray image. Write it as you write the findings section of the report."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    pil_image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")
if accelerator.is_main_process:
    print("\nAfter training:\n")
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 1.5, min_p = 0.1)

    print("\nActual Findings: ", finding_batch)
    print("\n------------------------------------")

inputs = tokenizer(
    pil_image2,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

if accelerator.is_main_process:
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 1.5, min_p = 0.1)

    print("\nActual Findings: ", finding_batch2)


# Access the log history (a list of dictionaries, each corresponding to a logging event)
log_history = trainer.state.log_history

train_loss_by_epoch = {}
val_loss_by_epoch = {}

for log in log_history:
    if "epoch" in log:
        epoch_int = int(log["epoch"])  # Convert fractional epoch to an integer epoch index
        # For training loss logs
        if "loss" in log:
            train_loss_by_epoch[epoch_int] = log["loss"]  # This will overwrite previous values; the last one remains
        # For evaluation loss logs
        if "eval_loss" in log:
            val_loss_by_epoch[epoch_int] = log["eval_loss"]


# Extract sorted epoch values and corresponding losses
train_epochs = sorted(train_loss_by_epoch.keys())
train_losses = [train_loss_by_epoch[epoch] for epoch in train_epochs]

val_epochs = sorted(val_loss_by_epoch.keys())
val_losses = [val_loss_by_epoch[epoch] for epoch in val_epochs]

if accelerator.is_main_process:
    print("Train Losses per epoch:", list(zip(train_epochs, train_losses)))
    print("Validation Losses per epoch:", list(zip(val_epochs, val_losses)))

plt.figure(figsize=(8, 6))
plt.plot(train_epochs, train_losses, label="Train Loss", marker="o")
plt.plot(val_epochs, val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
plt.close()  # Close the figure to free up memory