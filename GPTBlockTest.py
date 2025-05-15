import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import openai
import matplotlib.pyplot as plt 
import numpy as np 
import re

# Set your OpenAI API key


# Label mappings
labelReverseMapping = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
labelMapping = {v: k for k, v in labelReverseMapping.items()}

# Load the dataset
df = pd.read_csv('/work/Kristoffer/Dataset/Conversation.csv')

# Emotion cleaner and extractor
def extract_valid_emotion(responseText):
    validEmotions = set(labelMapping.keys())
    responseText = responseText.lower().strip()
    match = re.search(r'(sadness|joy|love|anger|fear|surprise)', responseText)
    if match:
        emotion = match.group(1)
        return emotion
    else:
        raise ValueError(f"Invalid emotion detected in model response: '{responseText}'")

# GPT-4 block classification
def classifySentenceBlock(block_text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an emotion classifier. Only respond with one of the following emotions: sadness, joy, love, anger, fear, surprise."
            },
            {
                "role": "user",
                "content": f"Classify the following text into one of these emotions:\n\n{block_text}"
            }
        ],
        temperature=0.0,
        max_tokens=10
    )
    rawOutput = response.choices[0].message['content']
    predictedEmotion = extract_valid_emotion(rawOutput)
    return predictedEmotion

# Function to classify a block with context from previous block's text
def classifySentenceBlock_with_context(current_block_text, previous_block_text):
    context_prompt = (
        f"Previous block:\n{previous_block_text}\n\n"
        f"Current block:\n{current_block_text}\n\n"
        "Classify the emotion of the current block, considering the previous block's content. "
        "Only respond with one of the following emotions: sadness, joy, love, anger, fear, surprise."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an emotion classifier. Only respond with one of the following emotions: sadness, joy, love, anger, fear, surprise."
            },
            {
                "role": "user",
                "content": context_prompt
            }
        ],
        temperature=0.0,
        max_tokens=10
    )
    rawOutput = response.choices[0].message['content']
    predictedEmotion = extract_valid_emotion(rawOutput)
    return predictedEmotion


def classifySentenceBlocksWithContext(df):
    allPredicted = []
    allActual = []

    for conversation_id, group in df.groupby('conversation_id'):
        previous_block_text = None

        for i in range(0, len(group), 3):
            block_sentences = group.iloc[i:i + 3]
            block_text = ' '.join(block_sentences['text'].tolist())

            try:
                if previous_block_text is None:
                    # First block — classify without context
                    block_emotion = classifySentenceBlock(block_text)
                    print(f"[Conversation {conversation_id}] First Block Emotion: {block_emotion}")
                else:
                    # Classify using previous block's text as context
                    block_emotion = classifySentenceBlock_with_context(block_text, previous_block_text)
                    print(f"[Conversation {conversation_id}] Block Emotion with Context: {block_emotion}")

                previous_block_text = block_text  # Update context for next block

                # Map to label and store
                allPredicted.append(labelMapping[block_emotion])
                actual_emotion = group.iloc[i]['label']
                allActual.append(actual_emotion)

            except ValueError as e:
                print(f"Error during classification: {e}")
                continue

    # Evaluation
    accuracy = accuracy_score(allActual, allPredicted)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(allActual, allPredicted, target_names=labelReverseMapping.values()))


        # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        allActual,
        allPredicted,
        labels=list(labelReverseMapping.keys())
    )

    emotions = list(labelReverseMapping.values())
    x = np.arange(len(emotions))  # label locations
    width = 0.35  # bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, precision, width, label='Precision', color='skyblue')
    rects2 = ax.bar(x + width / 2, recall, width, label='Recall', color='salmon')

    # Add labels, title, and legend
    ax.set_ylabel('Score')
    ax.set_title('Precision and Recall per Emotion')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45)
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Annotate bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig("precision_recall_chart.png", dpi=300)
    plt.show()

# Run classification
classifySentenceBlocksWithContext(df)
