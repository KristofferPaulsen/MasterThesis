# Import necessary libraries
import pandas as pd
from sklearn.metrics import accuracyScore, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the fine-tuned BERT model and tokenizer
model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Choose whether to use GPU if available or CPU for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define emotion labels
label_reverse_mapping = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
label_mapping = {v: k for k, v in label_reverse_mapping.items()}

# Load my conversation dataset (contains sentence text and its corresponding label)
df = pd.read_csv('/work/Kristoffer/Dataset/Conversation.csv')
labels = list(label_reverse_mapping.keys())

# Classifies each sentence individually and compares predicted vs actual labels
def classifyAndEvaluate(df):
    all_predicted = []
    all_actual = []
    conversationResults = {}

    # Go through each sentence row in the dataset
    for index, row in df.iterrows():
        sentence = row['text']
        true_label = row['label']
        conversationID = row['conversationID']

        # Tokenize and prepare input sentence
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # Get prediction from model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        # Convert label number back to emotion name
        predictedEmotion = label_reverse_mapping[predicted_class]
        actualEmotion = label_reverse_mapping[true_label] if isinstance(true_label, int) else true_label

        # Print result
        print(f"Sentence: {sentence}")
        print(f"Predicted Emotion: {predictedEmotion}, Actual Emotion: {actualEmotion}")
        print("-" * 50)

        # Store results for overall evaluation
        all_predicted.append(predicted_class)
        all_actual.append(true_label)

        #store results per conversation
        if conversationID not in conversationResults:
            conversationResults[conversationID] = {'predicted': [], 'actual': []}
        conversationResults[conversationID]['predicted'].append(predicted_class)
        conversationResults[conversationID]['actual'].append(true_label)

    # Print overall model accuracy and performance
    accuracy = accuracyScore(all_actual, all_predicted)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_actual, all_predicted, target_names=list(label_reverse_mapping.values())))

    # Accuracy per conversation
    print("\nPerformance by Conversation:")
    for conv_id, results in conversationResults.items():
        conversation_accuracy = accuracyScore(results['actual'], results['predicted'])
        print(f"Conversation {conv_id}: Accuracy = {conversation_accuracy:.4f}")

# Classifies sentence blocks (3 sentences at a time) using context from the previous block
def classifySentenceBlocksWithContext(df):
    all_predicted = []
    all_actual = []
    conversationResults = {}

    # Group sentences by each conversation
    for conversationID, group in df.groupby('conversationID'):
        conversationResults[conversationID] = {'predicted': [], 'actual': []}
        previousBlock = None  # To hold context

        # Process sentences in blocks of 3
        for i in range(0, len(group), 3):
            blockSentences = group.iloc[i:i+3]
            currentBlockText = ' '.join(blockSentences['text'].tolist())

            # Include context from previous block if available
            fullInputText = previousBlock + " " + currentBlockText if previousBlock else currentBlockText

            # Predict emotion for current block
            predictedEmotion = classifySentenceBlock(fullInputText)
            print(f"Conversation {conversationID} - Block {i//3 + 1} Prediction: {predictedEmotion}")

            # Update previous block for next iteration
            previousBlock = currentBlockText
            actualEmotion = blockSentences.iloc[0]['label']
            predictedLabel = label_mapping[predictedEmotion]

            # Store predicted and actual labels
            all_predicted.append(predictedLabel)
            all_actual.append(actualEmotion)
            conversationResults[conversationID]['predicted'].append(predictedLabel)
            conversationResults[conversationID]['actual'].append(actualEmotion)

    # Print overall accuracy and performance
    accuracy = accuracyScore(all_actual, all_predicted)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        all_actual,
        all_predicted,
        target_names=list(label_reverse_mapping.values()),
        labels=list(range(len(label_reverse_mapping))),
        zero_division=0
    ))

    # Print performance per conversation with predictions
    print("\nPerformance by Conversation:")
    for conv_id, results in conversationResults.items():
        conv_accuracy = accuracyScore(results['actual'], results['predicted'])
        predictedEmotions = [label_reverse_mapping[i] for i in results['predicted']]
        actualEmotions = [label_reverse_mapping[i] for i in results['actual']]

        print(f"Conversation {conv_id}: Accuracy = {conv_accuracy:.4f}")
        print("Predicted:", predictedEmotions)
        print("Actual   :", actualEmotions)
        print("-" * 50)

# Classify a single block of sentences 
def classifySentenceBlock(blockText):
    # Tokenize and predict the emotion of a single block
    inputs = tokenizer(blockText, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    
    # Convert prediction to emotion name
    predictedEmotion = label_reverse_mapping[predicted_class]
    return predictedEmotion

# Classify a block using previous emotion context 
def classifySentenceBlock_with_context(blockText, previous_emotion):
    inputs = tokenizer(blockText, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    predictedEmotion = label_reverse_mapping[predicted_class]
    return predictedEmotion

# Run the classification on the dataset
classifySentenceBlocksWithContext(df)
