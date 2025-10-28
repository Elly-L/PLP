# ==========================================
# 🧠 AI Tools Assignment - Part 2 Task 3
# NLP with spaCy — Named Entity Recognition & Sentiment
# ==========================================

# STEP 1: Install and import dependencies
!pip install -U spacy pandas

import spacy
import pandas as pd
from spacy.matcher import PhraseMatcher
from google.colab import files
from collections import Counter

# STEP 2: Load spaCy model (English)
# This downloads a small pre-trained model
!python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")
print("✅ spaCy model loaded successfully!")

# STEP 3: Upload your reviews dataset (optional)
# -----------------------------------------------
print("\n📤 You can upload your own 'reviews.csv' (column: 'review') or skip to use sample data.")
uploaded = files.upload()

if len(uploaded) > 0:
    filename = list(uploaded.keys())[0]
    df = pd.read_csv(filename)
    # Expecting a column called 'review'
    if 'review' not in df.columns:
        df.columns = ['review']  # rename if needed
    reviews = df['review'].dropna().tolist()
    print(f"\n✅ Loaded {len(reviews)} reviews from {filename}")
else:
    print("\n⚠️ No file uploaded. Using sample reviews instead.")
    reviews = [
        "I love the Acme Turbo Blender — it crushed ice and made smoothies in seconds!",
        "The ZetaPhone battery dies fast, I'm disappointed.",
        "Great sound from SoundMax headphones, great value for money.",
        "Terrible packaging! The product was leaking when it arrived.",
        "Absolutely love the camera quality on my PixelPhone. Stunning shots!"
    ]

# STEP 4: Named Entity Recognition (NER)
print("\n🔍 Named Entity Recognition Results:\n")
for doc in nlp.pipe(reviews):
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"REVIEW: {doc.text}")
    print(f"ENTITIES: {entities}")
    print("-"*60)

# STEP 5: PhraseMatcher for product name detection (custom gazetteer)
products = ["Acme Turbo Blender", "ZetaPhone", "SoundMax headphones", "PixelPhone"]
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(p) for p in products]
matcher.add("PRODUCT", patterns)

print("\n🛍️ Custom Product Name Matches:\n")
for doc in nlp.pipe(reviews):
    matches = matcher(doc)
    found = [doc[start:end].text for _, start, end in matches]
    print(f"REVIEW: {doc.text}")
    print(f"FOUND PRODUCTS: {found}")
    print("-"*60)

# STEP 6: Simple rule-based sentiment analysis
positive_words = {"love", "great", "excellent", "amazing", "good", "fast", "stunning", "value"}
negative_words = {"disappoint", "bad", "terrible", "slow", "dies", "leak", "poor", "broken"}

def simple_sentiment(text):
    doc = nlp(text.lower())
    pos = [t.text for t in doc if t.lemma_ in positive_words]
    neg = [t.text for t in doc if t.lemma_ in negative_words]
    if len(pos) > len(neg):
        return "Positive 😊"
    elif len(neg) > len(pos):
        return "Negative 😞"
    else:
        return "Neutral 😐"

print("\n💬 Sentiment Analysis:\n")
for r in reviews:
    sent = simple_sentiment(r)
    print(f"REVIEW: {r}")
    print(f"SENTIMENT: {sent}")
    print("-"*60)

print("\n🎯 Task 3 Complete — Take screenshots of NER, Product Matches, and Sentiment outputs.")