import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
import os

# -------------------- GPU Check --------------------
assert torch.cuda.is_available(), "CUDA is required for QLoRA"
print(f"GPU: {torch.cuda.get_device_name(0)}")

# -------------------- Model & Quantization --------------------
model_name = "google/gemma-3-4b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
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

# 🔴 REQUIRED FOR GEMMA
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# -------------------- LoRA Configuration --------------------
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------- Load Dataset --------------------
data = load_dataset("junaidiqbalsyed/insurance_policy_qa")

split_1 = data["train"].train_test_split(test_size=0.2, seed=42)
split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict({
    "train": split_1["train"],
    "validation": split_2["train"],
    "test": split_2["test"]
})

# -------------------- Reduce Dataset Sizes --------------------
TRAIN_SIZE = 1000
VAL_SIZE = 500
TEST_SIZE = 500

dataset["train"] = dataset["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(VAL_SIZE))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(TEST_SIZE))

print("Train:", dataset["train"].num_rows)
print("Val:", dataset["validation"].num_rows)
print("Test:", dataset["test"].num_rows)

# -------------------- Tokenization (FAST & SAFE) --------------------
def tokenize(batch):
    texts = [
        f"### Instruction:\n{q}\n\n### Response:\n{a}"
        for q, a in zip(batch["question"], batch["actual_answer"])
    ]

    return tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding=False   # 🔥 dynamic padding
    )

tokenized_path = "./tokenized_dataset_1k_fixed"

if os.path.exists(tokenized_path):
    tokenized_dataset = DatasetDict.load_from_disk(tokenized_path)
else:
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )
    tokenized_dataset.save_to_disk(tokenized_path)

# -------------------- Data Collator (CORRECT) --------------------
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt"
)

# -------------------- Training Arguments --------------------
training_args = TrainingArguments(
    output_dir="./fine-tuned-model/gemma_checkpoint",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-3,
    bf16=True,
    fp16=False,
    logging_steps=10,
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
    tokenizer=tokenizer,
    data_collator=data_collator
)

# -------------------- Train --------------------
trainer.train()

# -------------------- Save Adapter --------------------
adapter_path = "./trained_adapters/gemma-insurance-lora-1k-fixed"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

# -------------------- Final Evaluation --------------------
test_metrics = trainer.evaluate(tokenized_dataset["test"])
print("Test metrics:", test_metrics)
