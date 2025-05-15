import optuna
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

start_time = time.time()

# Model configuration
model_name = "bert-base-uncased"
num_labels = 6

# Load dataset
train = pd.read_csv('/work/Kristoffer/Dataset/training.csv')
validation = pd.read_csv('/work/Kristoffer/Dataset/validation.csv')
test = pd.read_csv('/work/Kristoffer/Dataset/test.csv')

# Label mapping
labelMapping = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "suprise": 5}
labelReverseMapping = {v: k for k, v in labelMapping.items()}
if isinstance(train['label'].iloc[0], str):
    train['label'] = train['label'].map(labelMapping)
    validation['label'] = validation['label'].map(labelMapping)
    test['label'] = test['label'].map(labelMapping)

# Convert to Hugging Face Datasets
trainDataset = Dataset.from_pandas(train).rename_column("label", "labels")
valDataset = Dataset.from_pandas(validation).rename_column("label", "labels")
testDataset = Dataset.from_pandas(test).rename_column("label", "labels")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Tokenize datasets
trainDataset = trainDataset.map(tokenize_function, batched=True).with_format("torch")
valDataset = valDataset.map(tokenize_function, batched=True).with_format("torch")
testDataset = testDataset.map(tokenize_function, batched=True).with_format("torch")

# Compute evaluation metrics
def computeMetrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Objective function for Optuna
def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("epochs", 3, 5)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.3)
    
    # Load model with dropout modification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, hidden_dropout_prob=dropout_rate
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Ensure both match
    logging_strategy="epoch",
    learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
    per_device_train_batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
    num_train_epochs=trial.suggest_int("epochs", 5, 10, step=1),
    weight_decay=0.01,
    save_steps=500,  # Can be removed since we now save per epoch
    logging_dir='./logs',
    load_best_model_at_end=True,
    report_to="none",
)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        trainDataset=trainDataset,
        evalDataset=valDataset,
        computeMetrics=computeMetrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results["eval_loss"]  # Minimize loss

# Run random search with Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)  # Run 10 random trials

# Print best hyperparameters
print("Best Hyperparameters:", study.best_params)

# Train final model with best hyperparameters
best_params = study.best_params
final_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
final_training_args = TrainingArguments(
    output_dir='./best_model',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=study.best_trial.params["learning_rate"],  # FIXED
    per_device_train_batch_size=study.best_trial.params["batch_size"],  # FIXED
    num_train_epochs=study.best_trial.params["epochs"],  # FIXED
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    report_to="none",
)


final_trainer = Trainer(
    model=final_model,
    args=final_training_args,
    trainDataset=trainDataset,
    evalDataset=valDataset,
    computeMetrics=computeMetrics,
)
final_trainer.train()

# Save final model
final_model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

# Evaluate on test set
results = final_trainer.evaluate(testDataset)
print(f"Test Accuracy: {results.get('eval_accuracy')}")

# Print runtime
# Print best hyperparameters
print("Best Hyperparameters:", study.best_params)
print(f"Total runtime: {time.time() - start_time:.2f} seconds")
