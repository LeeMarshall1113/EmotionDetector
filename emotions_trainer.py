import torch
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ---------------------------
# 1. Load the GoEmotions Dataset
# ---------------------------
dataset = load_dataset("go_emotions")

# ---------------------------
# 1.1 Convert Multi-labels to Single Label
# ---------------------------
# GoEmotions provides multi-label annotations (a list of labels).
# For a simple single-label classification, we'll select the first label.
def convert_to_single_label(example):
    # If the list is empty (unlikely), default to a special value (-1).
    example["label"] = example["labels"][0] if len(example["labels"]) > 0 else -1
    return example

# Apply the conversion on all splits
dataset = dataset.map(convert_to_single_label)

# Optionally, remove the original "labels" column to avoid confusion.
dataset = dataset.remove_columns("labels")

# ---------------------------
# 2. Load Pre-trained Tokenizer and Model
# ---------------------------
model_name = "distilbert-base-uncased"
num_labels = 28  # Adjust according to your label space; GoEmotions has 28 possible labels.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# ---------------------------
# 3. Tokenize the Dataset
# ---------------------------
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ---------------------------
# 4. Define Training Arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
)

# ---------------------------
# 5. Define the Evaluation Metric using 'evaluate'
# ---------------------------
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# ---------------------------
# 6. Initialize the Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# ---------------------------
# 7. Fine-Tune the Model
# ---------------------------
print("Starting training...")
trainer.train()

# Evaluate the model after training
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Save the fine-tuned model and tokenizer
model.save_pretrained("./go_emotions_model")
tokenizer.save_pretrained("./go_emotions_model")

# ---------------------------
# 8. Inference: Detecting Emotion from User Input
# ---------------------------
# Map label IDs to human-readable emotion names (adjust as needed)
id2label = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral"
}

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict_emotion(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    # Move input tensors to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Get model predictions without calculating gradients
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)
    # Get the predicted label id
    predicted_idx = torch.argmax(probabilities, dim=-1).item()
    predicted_emotion = id2label.get(predicted_idx, "Unknown")
    
    return predicted_emotion, probabilities.cpu().numpy()

# ---------------------------
# 9. Example: Detecting Emotion from a Sample Input
# ---------------------------
sample_text = "I just got a promotion at work and I'm incredibly excited!"
emotion, probs = predict_emotion(sample_text)
print(f"\nInput Text: {sample_text}")
print(f"Predicted Emotion: {emotion}")
