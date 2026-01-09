import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import os

# -------------------- GPU Check --------------------
if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("CUDA is NOT available! Training will be on CPU and very slow.")

# -------------------- Model & Quantization --------------------
model_name = "google/gemma-3-4b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# Optional: pad token handling if needed
# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = tokenizer.eos_token_id

# -------------------- LoRA Configuration --------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------- Load & Split Dataset --------------------
data = load_dataset("junaidiqbalsyed/insurance_policy_qa")

# Original splits
split_1 = data["train"].train_test_split(test_size=0.2, seed=42)
split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict({
    "train": split_1["train"],
    "validation": split_2["train"],
    "test": split_2["test"]
})

print(
    f"Original Train: {dataset['train'].num_rows}, "
    f"Validation: {dataset['validation'].num_rows}, "
    f"Test: {dataset['test'].num_rows}"
)

# -------------------- Reduce Training Set to ~1/3 --------------------
reduced_train = dataset["train"].train_test_split(test_size=2/3, seed=42)["train"]
dataset["train"] = reduced_train

print(
    f"Reduced Train: {dataset['train'].num_rows}, "
    f"Validation: {dataset['validation'].num_rows}, "
    f"Test: {dataset['test'].num_rows}"
)

# -------------------- Tokenization --------------------
def tokenize(batch):
    """Tokenize without returning PyTorch tensors, keep as lists for Arrow dataset"""
    texts = [
        f"### Instruction:\n{q}\n\n### Response:\n{a}"
        for q, a in zip(batch["question"], batch["actual_answer"])
    ]
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128
    )
    # Labels are same as input_ids
    tokens["labels"] = [x.copy() for x in tokens["input_ids"]]
    return tokens

# Path to cache tokenized dataset
tokenized_path = "./tokenized_dataset"

if os.path.exists(tokenized_path):
    print("Loading tokenized dataset from disk...")
    tokenized_dataset = DatasetDict.load_from_disk(tokenized_path)
else:
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4  # adjust based on CPU cores
    )
    print("Saving tokenized dataset to disk...")
    tokenized_dataset.save_to_disk(tokenized_path)

# -------------------- Training Arguments --------------------
training_args = TrainingArguments(
    output_dir="./fine-tuned-model/gemma-insurance-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=10,
    fp16=True,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    remove_unused_columns=False
)

# -------------------- Trainer --------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer  # Trainer will convert lists to tensors automatically
)

# -------------------- Train --------------------
trainer.train()

# -------------------- Save Fine-Tuned LoRA Adapter --------------------
adapter_path = "./trained_adapters/gemma-insurance-lora-tuned-adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

# -------------------- Final Test Evaluation --------------------
test_metrics = trainer.evaluate(tokenized_dataset["test"])
print("Test metrics:", test_metrics)
