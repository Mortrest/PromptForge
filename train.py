from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset
import numpy as np
import evaluate
import torch

# Load datasets and metric
dataset = load_dataset("squad")
metric = evaluate.load("squad")

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Preprocessing function
def preprocess(example):
    inputs = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    offset_mapping = inputs.pop("offset_mapping")
    sample_mapping = inputs.pop("overflow_to_sample_mapping")

    start_positions, end_positions = [], []
    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sample_idx = sample_mapping[i]
        answers = example["answers"][sample_idx]
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            start_positions.append(token_start_index - 1)
            end_positions.append(token_end_index + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Prepare data
train_data = dataset["train"].select(range(5000)).map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
val_data = dataset["validation"].select(range(500)).map(preprocess, batched=True, remove_columns=dataset["validation"].column_names)

# Evaluation function
def evaluate_model(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    start_preds, end_preds = [], []
    start_labels, end_labels = [], []

    for batch in torch.utils.data.DataLoader(dataset, batch_size=8):
        # Move everything to the model's device
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            start_preds.extend(torch.argmax(outputs.start_logits, dim=-1).cpu().tolist())
            end_preds.extend(torch.argmax(outputs.end_logits, dim=-1).cpu().tolist())
            start_labels.extend(batch["start_positions"].cpu().tolist())
            end_labels.extend(batch["end_positions"].cpu().tolist())

    start_acc = np.mean(np.array(start_preds) == np.array(start_labels))
    end_acc = np.mean(np.array(end_preds) == np.array(end_labels))
    return {"start_acc": start_acc, "end_acc": end_acc}

# Evaluate before fine-tuning
print("Evaluating before fine-tuning...")

# val_data.set_format("torch")
# metrics_before = evaluate_model(model, val_data)
# print("Before fine-tuning:", metrics_before)

# Training setup
training_args = TrainingArguments(
    output_dir="./distilbert-qa-small",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# Fine-tune on 1000 samples
print("Fine-tuning on 1000 samples...")
trainer.train()

# Evaluate after fine-tuning
print("Evaluating after fine-tuning...")
val_data.set_format("torch")
metrics_after = evaluate_model(model, val_data)
print("After fine-tuning:", metrics_after)