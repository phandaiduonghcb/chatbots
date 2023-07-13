import random
import torch
import spacy
import numpy as np
import pandas as pd
from model import EmbeddingClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE = "data.pth"
data = torch.load(FILE)

classes = data['classes']
model_state = data["model_state"]
tag_to_response = data["tag_to_response"]
max_words = data["max_words"]
embed_len = data["embed_length"]

model = EmbeddingClassifier(max_words, embed_len, len(classes)).to(device)
model.load_state_dict(model_state)
model.eval()

nlp = spacy.load("en_core_web_lg", exclude=["ner","parser"])
excluded_pos_tags = ['PUNCT', 'SYM', 'X', 'AUX']
keyword_pos_tags = ['NOUN', 'PROPN']
course_df = pd.read_csv('Coursera.csv')
 

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    doc = nlp(sentence)
    token_vectors = []
    keywords = []
    for token in doc:
        if token.pos_ not in excluded_pos_tags:
            token_vectors.append(nlp(token.lemma_)[0].vector)
        if token.pos_ in keyword_pos_tags:
            keywords.append(token.text)
    X = torch.tensor(np.average(token_vectors, axis=0)).reshape(1,-1)
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = classes[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        print(f"{bot_name}: {random.choice(tag_to_response[tag])}")
        if tag == "course":
            print(keywords)
            counts = course_df['Course Description'].str.count('|'.join(keywords))
            # Sort the DataFrame in descending order by the number of occurrences
            sorted_df = course_df.iloc[(-counts).argsort()]

            # Select the top N rows
            top_rows = sorted_df.head(10)

            # Print the top N rows
            print(top_rows)
    else:
        print(f"{bot_name}: I do not understand...")
