import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define id2label mapping
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

# Load the saved model and tokenizer
model_name_or_path = "./go_emotions_model"  # adjust path if necessary
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

# Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict_emotion_percentages(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    # Move tensors to device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Inference without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).squeeze()  # Remove batch dimension
    percentages = (probabilities * 100).cpu().numpy()
    
    # Map percentages to corresponding emotions
    emotion_percentages = {id2label[i]: float(percentages[i]) for i in range(len(percentages))}
    # Sort the dictionary in descending order
    sorted_emotion_percentages = dict(sorted(emotion_percentages.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_emotion_percentages

def main():
    print("Enter a message to analyze its emotions. Type 'exit' to quit.\n")
    while True:
        user_input = input("Your message: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        
        emotion_percentages = predict_emotion_percentages(user_input)
        print("\nEmotion percentages (from greatest to least):")
        for emotion, percent in emotion_percentages.items():
            print(f"{emotion}: {percent:.2f}%")
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
