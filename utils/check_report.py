from unsloth import FastVisionModel
import os
import sys
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
from tqdm import tqdm
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoTokenizer  # or use your tokenizer from FastVisionModel
from unsloth import is_bf16_supported
from dataset import get_data_loaders
from transformers import TextStreamer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Load the dataset (including the test loader)
cleaned_dataset_dir = os.environ.get("DATASET_DIR", "data/cleaned_dataset")
output_excel_file = os.environ.get("METADATA_FILE", "data/cleaned_dataset_metadata.xlsx")
train_loader, val_loader, test_loader = get_data_loaders(output_excel_file, cleaned_dataset_dir, drop_last=True)

# 2. Load the trained model and tokenizer from the saved checkpoint.
#    Note: We assume the checkpoint already includes LoRA adapters, so we don't reapply them.
model, tokenizer = FastVisionModel.from_pretrained(
    "outputs/checkpoint-2516",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)

# Switch to inference mode
FastVisionModel.for_inference(model)

# Define your instruction and conversation template (same for all samples in this example)
instruction = ("You are an expert radiographer. Describe accurately what you see in this chest x-ray image. Write it as you write the findings section of the report.")
conversation_template = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]

# We'll generate reports for the first 100 samples from the test loader.
num_samples = 100
sample_count = 0

print("\nGenerating reports for {} samples:\n".format(num_samples))

# Loop over the test_loader
for batch in test_loader:
    # If your batch contains multiple samples, you can loop over them.
    batch_size = len(batch["PA"])
    for i in range(batch_size):
        if sample_count >= num_samples:
            break

        # Get the image and actual finding
        image = batch["PA"][i]
        actual_finding = batch["Findings"][i]

        # Convert image tensor to a PIL image
        pil_image = to_pil_image(image)

        # Create the chat prompt using the tokenizer's chat template method
        input_text = tokenizer.apply_chat_template(conversation_template, add_generation_prompt=True)

        # Tokenize the inputs; ensure they are sent to the same device as the model.
        inputs = tokenizer(
            pil_image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        print(f"Sample {sample_count + 1}:")
        print("Generated Report:")
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            use_cache=True,
            temperature=1.5,
            min_p=0.1
        )

        print("\nActual Findings:")
        print(actual_finding)
        print("-" * 80)

        sample_count += 1

    if sample_count >= num_samples:
        break
