from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_path = "./saved_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define label mapping
labelReverseMapping = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

def classifySentenceWithContext(conversation, sentence_index):
    """
    Classifies the emotion of a sentence in a conversation.
    - If sentence_index is 0 (first sentence), classify it alone.
    - Otherwise, use previous sentences as context.
    """
    targetSentence = conversation[sentence_index]
    
    # If it's the first sentence, classify it alone
    if sentence_index == 0:
        inputText = targetSentence
    else:
        context = " ".join(conversation[:sentence_index])  # All previous sentences
        inputText = f"{context} [SEP] {targetSentence}"

    # Tokenize and classify
    inputs = tokenizer(inputText, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictedClass = torch.argmax(logits, dim=1).item()
    
    return labelReverseMapping[predictedClass]

# Example dataset with conversations
conversations = [
    [
        "I don’t know where I am. I think I took a wrong turn.",
        "Stay calm. Where are you now?",
        "I have no idea. It’s dark, and my phone is at 5%.",
        "Okay, take a deep breath. Can you see any landmarks?",
        "There’s a gas station across the street.",
        "Wait—I see a sign. I think I recognize this road!",
        "Great! Keep walking toward something familiar.",
        "I think I see my apartment building now.",
        "I made it home. Thank you for staying on the phone with me.",
        "Of course. I’d never leave you alone in a situation like that.",
        "I was really freaking out for a second.",
        "That’s understandable. I’m just glad you’re safe."
    ],
    [
        "I don’t think I can do this presentation tomorrow.",
        "Why not? You’ve been preparing for weeks.",
        "I just know I’ll mess up and embarrass myself.",
        "Hey, I believe in you. You’ve worked so hard for this.",
        "But what if I forget everything the moment I step up there?",
        "Then I’ll be in the audience cheering for you the whole time.",
        "That actually makes me feel a little better.",
        "See? You’ve got this. Trust yourself.",
        "You know what? I think I can do this after all.",
        "Yes! That’s the confidence I was waiting for.",
        "Thanks for always having my back.",
        "Always."
    ],
    [
    "I almost got hit by a car today.",
    "What?! Are you okay?",
    "Yeah, but it was terrifying. It came out of nowhere.",  # FEAR
    "That’s so dangerous. People need to pay more attention when driving.",
    "I know, right? It’s like they didn’t even see me.",
    "Some people just don’t care about pedestrians.",  # ANGER
    "It makes me so mad. I could’ve been seriously hurt.",
    "I don’t even want to think about what could have happened.",
    "It just reminds me how fragile life is.",  # SADNESS
    "Yeah… that’s a scary thought.",
    "I just need a moment to process everything.",
    "Take your time. I’m here if you need to talk."
    ],
    [
    "Guess what? I just got accepted into my dream school!",
    "Are you serious?! That’s amazing!",
    "Yeah, I’m still in shock. I can’t believe it!",  # JOY
    "You totally deserve it. I knew you could do it.",
    "Thanks, I was so nervous when I submitted my application.",
    "You worked hard for this, and I’m so proud of you.",  # LOVE
    "That means a lot coming from you.",
    "Of course. I know how much this means to you.",
    "Oh my god, wait—They’re offering me a scholarship too!",  # SURPRISE
    "WHAT?! That’s incredible!",
    "I’m actually shaking right now.",
    "This is your moment. Enjoy it!"
    ],
    [
    "I can’t believe she talked about me behind my back.",
    "Are you sure it wasn’t a misunderstanding?",
    "No. I saw the messages myself.",  # ANGER
    "Wow… I’m really sorry. That must hurt.",
    "Yeah, I thought we were closer than that.",
    "I just don’t understand why she’d do this to me.",  # SADNESS
    "Maybe you should talk to her and see what she has to say.",
    "I don’t even know if I want to.",
    "Wait… she just texted me. She said someone else used her phone.",  # SURPRISE
    "That… actually makes sense. Maybe I overreacted.",
    "It’s worth hearing her out at least.",
    "Yeah, I’ll talk to her first before jumping to conclusions."
    ],    
    [
    "I can’t believe you ignored my message yesterday!",
    "Wait, what? I didn’t ignore it.",
    "You left me on read for hours!",
    "I wasn’t ignoring you, I just got really busy.",
    "I just felt like you didn’t care.",
    "I would never do that on purpose. You’re my best friend.",
    "I guess I overreacted…",
    "It’s okay. Let’s talk things through.",
    ],
    [
    "I just found out my coworker took credit for my idea!",
    "What?! That’s so wrong!",
    "Yeah, and now our manager thinks it was all his work.",
    "I’d be furious. Are you going to talk to them?",
    "I don’t know. I feel so defeated.",
    "You deserve the recognition. Don’t let them take that from you.",
    "Actually… our manager just called me in. He found out the truth!",
    "Really? That’s amazing!",
    ],
    
    ["Wait… Is that you?!",
    "Oh my god! I can’t believe it!",
    "It’s been YEARS!",
    "I know! How have you been?",
    "So much has changed, but seeing you brings back so many memories.",
    "Same here. I’ve really missed you.",
    "We have so much to catch up on!",
    "Let’s not lose touch again, okay?",
    ],
    [
    "I don’t think I can go on this roller coaster.",
    "Come on, it’ll be fun!",
    "What if something goes wrong?",
    "I promise you’ll be safe. I’ll be right next to you.",
    "I don’t know…",
    "I won’t force you, but I think you’d love it.",
    "Alright, let’s do it!",
    "That’s the spirit!",
    ],
    [
    "I just got the keys to my new apartment!",
    "That’s so exciting!",
    "I can’t believe I finally have my own place.",
    "You worked hard for this. You deserve it!",
    "I feel so independent.",
    "And just think of all the memories you’ll make here.",
    "Oh my god, I just found out my parents are surprising me with furniture!",
    "That’s incredible!",
    ],
    [
    "I think someone is following me.",
    "Are you serious? Where are you?",
    "I don’t know, but I feel really unsafe.",
    "Get somewhere public and call me right now.",
    "Okay… I made it to a cafe.",
    "Stay there. I’m coming to get you.",
    "Thank you so much.",
    "I’m just glad you’re okay.",
    ],
    [
    "I didn’t get the job.",
    "Oh no, I’m so sorry.",
    "I really thought I had a chance.",
    "Hey, just because they didn’t pick you doesn’t mean you aren’t amazing.",
    "I still feel like I failed.",
    "No way. You’re talented, and the right opportunity will come along.",
    "Thanks… I needed to hear that.",
    "Of course! And when you do get that dream job, we’re celebrating!",
    ],
    [
    "I don’t think we can be friends anymore.",
    "What? Why are you saying that?",
    "You’ve changed. It’s like you don’t even care anymore.",
    "That’s not true! I’ve just been dealing with a lot.",
    "Then why didn’t you tell me? I had to hear from someone else that you were struggling.",
    "I didn’t want to burden you with my problems.",
    "That’s not how friendships work. I would’ve been there for you.",
    "I know… and I regret pushing you away.",
    "It hurt, you know? I felt like I lost my best friend.",
    "You didn’t lose me. I just needed time to figure things out.",
    "I’m still mad, but… I do miss you.",
    "I miss you too. Can we start over?",
    "I think we can. But next time, just talk to me.",
    "Deal.",
    ]

]

# Classify emotions for the first sentence, every third sentence, and the last sentence
for i, conversation in enumerate(conversations):
    print(f"\n=== Conversation {i+1} ===")
    
    # Classify the first sentence
    firstSentenceEmotion = classifySentenceWithContext(conversation, 0)
    print(f"Sentence 1: {conversation[0]}  →  Emotion: {firstSentenceEmotion}")
    
    # Classify every third sentence
    for j in range(3, len(conversation), 3):
        emotion = classifySentenceWithContext(conversation, j)
        print(f"Sentence {j+1}: {conversation[j]}  →  Emotion: {emotion}")

    # Always classify the last sentence
    if len(conversation) - 1 not in range(0, len(conversation), 3):  # Avoid duplicate classification
        lastSentenceEmotion = classifySentenceWithContext(conversation, len(conversation) - 1)
        print(f"Sentence {len(conversation)}: {conversation[-1]}  →  Emotion: {lastSentenceEmotion}")

