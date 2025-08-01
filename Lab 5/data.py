import random
import pandas as pd
from faker import Faker

# Initialize Faker for realistic data generation
fake = Faker()
Faker.seed(42)

def generate_sentiment_data(num_samples=50):
    """Generate synthetic sentiment analysis data"""
    data = []
    positive_keywords = ["excellent", "amazing", "wonderful", "love", "great", "best", "favorite", "awesome"]
    negative_keywords = ["terrible", "awful", "worst", "hate", "bad", "disappointing", "poor", "horrible"]
    
    for _ in range(num_samples):
        # Generate realistic product/movie names
        product = fake.catch_phrase()
        
        if random.random() > 0.45:  # 55% positive reviews
            sentiment = 1
            template = random.choice([
                f"I absolutely love this {product}! It's {random.choice(positive_keywords)}.",
                f"This {product} is {random.choice(positive_keywords)}. Highly recommend!",
                f"Best {product} I've ever used. {random.choice(positive_keywords)} experience!",
                f"Five stars for this {product}. {fake.sentence()} {random.choice(positive_keywords)}!"
            ])
        else:  # 45% negative reviews
            sentiment = 0
            template = random.choice([
                f"I'm disappointed with this {product}. It's {random.choice(negative_keywords)}.",
                f"Worst {product} I've ever bought. {random.choice(negative_keywords)} experience!",
                f"Don't waste your money on this {product}. {random.choice(negative_keywords)} quality.",
                f"Terrible experience with this {product}. {fake.sentence()} {random.choice(negative_keywords)}."
            ])
        
        # Add personalization
        review = template.replace("I", fake.first_name())
        data.append({"text": review, "label": sentiment})
    
    return pd.DataFrame(data)

def generate_ner_data(num_samples=30):
    """Generate synthetic named entity recognition data"""
    data = []
    entity_types = ["PER", "ORG", "LOC", "MISC"]
    
    for _ in range(num_samples):
        # Create entities
        person = fake.name()
        organization = fake.company()
        location = fake.city()
        misc = fake.job().split()[0] + " conference"
        
        # Create sentence templates
        templates = [
            f"{person} from {organization} attended the {misc} in {location}.",
            f"The {misc} was held in {location} with keynote speaker {person}.",
            f"{organization} announced a new partnership with {location} officials at {misc}.",
            f"{person} of {organization} will visit {location} for the annual {misc}."
        ]
        
        text = random.choice(templates)
        
        # Create entity annotations
        entities = []
        if "PER" in text:
            entities.append({"entity": "PER", "word": person, "start": text.index(person), "end": text.index(person) + len(person)})
        if "ORG" in text:
            entities.append({"entity": "ORG", "word": organization, "start": text.index(organization), "end": text.index(organization) + len(organization)})
        if "LOC" in text:
            entities.append({"entity": "LOC", "word": location, "start": text.index(location), "end": text.index(location) + len(location)})
        if "MISC" in text:
            entities.append({"entity": "MISC", "word": misc, "start": text.index(misc), "end": text.index(misc) + len(misc)})
        
        data.append({"text": text, "entities": entities})
    
    return pd.DataFrame(data)

def generate_qa_data(num_samples=20):
    """Generate synthetic question-answering data"""
    data = []
    topics = [
        ("Eiffel Tower", "architecture", "Paris, France", "1889", "Gustave Eiffel"),
        ("Great Wall of China", "history", "northern China", "7th century BC", "Qin Shi Huang"),
        ("Machine Learning", "computer science", "data patterns", "1959", "Arthur Samuel"),
        ("Photosynthesis", "biology", "plants", "1770s", "Jan Ingenhousz"),
        ("Blockchain", "technology", "distributed ledger", "2008", "Satoshi Nakamoto")
    ]
    
    for topic, category, location, year, person in topics:
        # Generate context paragraphs
        context = f"The {topic} is a {category}-related structure located in {location}. " \
                  f"It was first created in {year} by {person}. " \
                  f"{fake.paragraph(nb_sentences=2)}"
        
        # Generate questions
        questions = [
            f"Who created the {topic}?",
            f"Where is the {topic} located?",
            f"When was the {topic} first created?",
            f"What category does the {topic} belong to?",
            f"What is the main subject of this passage?"
        ]
        
        # Generate answers
        answers = [
            {"text": person, "answer_start": context.index(person)},
            {"text": location, "answer_start": context.index(location)},
            {"text": year, "answer_start": context.index(year)},
            {"text": category, "answer_start": context.index(category)},
            {"text": topic, "answer_start": context.index(topic)}
        ]
        
        for i, question in enumerate(questions):
            data.append({
                "context": context,
                "question": question,
                "answer": answers[i]["text"],
                "answer_start": answers[i]["answer_start"]
            })
    
    return pd.DataFrame(data[:num_samples])

def generate_all_datasets():
    """Generate all synthetic datasets and save to CSV"""
    print("Generating synthetic sentiment data...")
    sentiment_df = generate_sentiment_data(100)
    sentiment_df.to_csv("synthetic_sentiment.csv", index=False)
    
    print("Generating synthetic NER data...")
    ner_df = generate_ner_data(50)
    ner_df.to_csv("synthetic_ner.csv", index=False)
    
    print("Generating synthetic QA data...")
    qa_df = generate_qa_data(30)
    qa_df.to_csv("synthetic_qa.csv", index=False)
    
    print("Synthetic datasets generated successfully!")
    return sentiment_df, ner_df, qa_df

# Generate and display sample data
if __name__ == "__main__":
    sentiment, ner, qa = generate_all_datasets()
    
    print("\nSample Sentiment Data:")
    print(sentiment.head(3))
    
    print("\nSample NER Data:")
    print(ner.head(2))
    
    print("\nSample QA Data:")
    print(qa.head(2))