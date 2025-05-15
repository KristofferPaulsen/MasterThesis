from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import torch  
import time
import seaborn as sns
import matplotlib.pyplot as plt

# Start timer to measure total runtime 
start_time = time.time()


model_name = "bert-base-uncased"  # Using pre-trained BERT
num_labels = 6  #Six emotion classes

# Load dataset splits from CSV files
train = pd.read_csv('/work/Kristoffer/Dataset/training.csv')
validation = pd.read_csv('/work/Kristoffer/Dataset/validation.csv')
test = pd.read_csv('/work/Kristoffer/Dataset/test.csv')

# Define label mapping from emotion text to integers
labelMapping = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "suprise": 5}
labelReverseMapping = {v: k for k, v in labelMapping.items()}  # For reversing predictions

# If labels in CSVs are strings, convert them to integers using labelMapping
if 'label' in train.columns and isinstance(train['label'].iloc[0], str):
    train['label'] = train['label'].map(labelMapping)
    validation['label'] = validation['label'].map(labelMapping)
    test['label'] = test['label'].map(labelMapping)

# Convert the pandas DataFrames into Hugging Face Dataset format
trainDataset = Dataset.from_pandas(train)
valDataset = Dataset.from_pandas(validation)
testDataset = Dataset.from_pandas(test)

# Rename label column to "labels" because that is the required column name for Hugging Face Trainer
trainDataset = trainDataset.rename_column("label", "labels")
valDataset = valDataset.rename_column("label", "labels")
testDataset = testDataset.rename_column("label", "labels")

# Load tokenizer and model from Hugging Face model hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Check if a GPU is available and use it if possible, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function for tokenizing text examples
def tokenizeFunction(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Apply tokenization to all dataset splits
trainDataset = trainDataset.map(tokenizeFunction, batched=True)
valDataset = valDataset.map(tokenizeFunction, batched=True)
testDataset = testDataset.map(tokenizeFunction, batched=True)

# Convert datasets to PyTorch format so we can train with Trainer
trainDataset = trainDataset.with_format("torch")
valDataset = valDataset.with_format("torch")
testDataset = testDataset.with_format("torch")

# Function to calculate and print evaluation metrics
def computeMetrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)  # Get the predicted label
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    # Print confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    print("\nConfusion Matrix:\n", conf_matrix)

    # Print precision, recall, f1 for each class
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=labelMapping.keys()))   

    # Show confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labelMapping.keys(), yticklabels=labelMapping.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    return {"accuracy": acc, "f1": f1}

# Set training arguments for model fine-tuning
training_args = TrainingArguments(
    output_dir='./results',  # Where to save results
    evaluation_strategy="epoch",  # Evaluate after each epoch
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=4.9e-5,  # Learning rate found from tuning
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_steps=500,
    logging_dir='./logs',
    load_best_model_at_end=True,  # Restore the best checkpoint at the end
    report_to="none",  # Disable reporting to third-party tools
)

# Create Trainer instance with early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trainDataset,
    eval_dataset=valDataset,
    compute_metrics=computeMetrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Start the training process
trainer.train()

# Save the trained model and tokenizer for future use
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

# Evaluate model performance on the test dataset
results = trainer.evaluate(testDataset)
accuracy = results.get("eval_accuracy")
print(f"Test Accuracy: {accuracy}")

# Function for classifying a single piece of text using the trained model
def predictEmotion(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return labelReverseMapping[predicted_class]

# Run the model on custom examples to see how it behaves
examples = [
    "I am so excited to see you!",
    "This is the worst day of my life.",
    "I'm scared to go outside at night.",
    "Wow, I didn't expect this surprise!",
    "I feel indifferent about the situation.",
    "I did not like when the man approached me", 
    "I was at the playground", 
    "The man was trying to touch my private parts"
]

print("\nTesting with Custom Text Inputs:")
for example in examples:
    emotion = predictEmotion(example, model, tokenizer)
    print(f"Text: {example}\nPredicted Emotion: {emotion}\n")

# Print how long the script took to run
print(f"Total runtime: {time.time() - start_time:.2f} seconds")
