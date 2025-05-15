import openai
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to dataset files
train_path = "./Dataset/training.csv"
val_path = "./Dataset/validation.csv"
test_path = "./Dataset/test.csv"


#model = "gpt-3.5-turbo"

# Define a mapping for numerical labels to emotion names
label_mapping = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "Surprise",
    # Add more as needed based on your dataset
}

# Read the CSV files into Pandas DataFrames
try:
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
except Exception as e:
    print(f"Error reading dataset files: {e}")
    exit()

# Map numerical labels to emotion names and ensure all labels are strings
def preprocess_data(df):
    if 'label' not in df.columns:
        print("Error: 'label' column not found in dataset.")
        exit()

    try:
        # Map numerical labels to emotion names
        df['label'] = df['label'].map(label_mapping)

        # Convert all labels to strings
        df['label'] = df['label'].astype(str)

        # Check for any unmapped labels
        if df['label'].isnull().any():
            print("Warning: Some labels couldn't be mapped. Check your label_mapping.")
        return df
    except Exception as e:
        print(f"Error processing labels: {e}")
        exit()

# Preprocess datasets
train_data = preprocess_data(train_data)
val_data = preprocess_data(val_data)
test_data = preprocess_data(test_data)

def classify_emotion_gpt(text, model="gpt-4"):
    try:
        # Define the valid emotion set
        valid_emotions = {"sadness", "joy", "love", "anger", "fear", "surprise"}

        # GPT prompt with constraints
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI that detects emotions from text. "
                        "The only valid emotions are: sadness, joy, love, anger, fear, and surprise. "
                        "Classify the emotion of the text into one of these categories."
                    ),
                },
                {"role": "user", "content": f"Classify the emotion of the following text: '{text}'."}
            ],
            max_tokens=10,  # Keep the output short
            temperature=0,  # Deterministic output
        )

        # Extract and normalize the response
        predicted_label = response['choices'][0]['message']['content'].strip().lower()

        # Validate the prediction
        if predicted_label in valid_emotions:
            return predicted_label
        else:
            print(f"Invalid prediction: {predicted_label} for text: {text}")
            return None  # Or handle invalid predictions differently if needed
    except Exception as e:
        print(f"Error processing text: {text}\nError: {e}")
        return None


def evaluate_dataset(dataset, model="gpt-4"):
    predictions = []
    true_labels = []

    for _, row in dataset.iterrows():
        text = row['text']
        true_label = row['label'].lower()  # Ensure label is lowercase

        predicted_label = classify_emotion_gpt(text, model=model)

        if predicted_label is not None:  # Only include valid predictions
            predictions.append(predicted_label)
            true_labels.append(true_label)
        else:
            print(f"Skipped invalid prediction for text: {text}")

        print(f"Text: {text}\nTrue Label: {true_label}\nPredicted Label: {predicted_label}\n")

    # Generate and print classification metrics only if we have valid predictions
    if predictions:
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, zero_division=0))

        conf_matrix = confusion_matrix(true_labels, predictions)
        print("\nConfusion Matrix:\n", conf_matrix)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    else:
        print("No valid predictions were made.")


# Main block to run evaluation
if __name__ == "__main__":
    # Ensure test_data is loaded correctly before using
    if 'test_data' in locals():
        print("Evaluating on Test Dataset...")
        evaluate_dataset(test_data)
    else:
        print("Test dataset not loaded properly.")
