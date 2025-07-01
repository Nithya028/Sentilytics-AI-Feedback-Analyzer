from transformers import pipeline

# Initialize the zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Candidate main topics and their subtopics
CANDIDATE_TOPICS = ["Delivery", "Quality", "Clothes", "Support", "Packaging", "Price", "App Experience"]
CANDIDATE_SUBTOPICS = {
    "Delivery": ["Fast Delivery", "Late Delivery", "Free Delivery"],
    "Quality": ["Material Quality", "Durability"],
    "Clothes": ["Fit", "Size", "Style"],
    "Support": ["Customer Support", "Responsiveness"],
    "Packaging": ["Box Quality", "Eco-Friendly Packaging"],
    "Price": ["Expensive", "Worth the Price", "Overpriced"],
    "App Experience": ["UI Design", "Ease of Use", "Responsiveness"]
}

# Adjustable threshold
TOPIC_THRESHOLD = 0.6
SUBTOPIC_THRESHOLD = 0.5

def extract_topics(text):
    text = text.strip()

    # Step 1: Predict main topics
    main_result = classifier(text, CANDIDATE_TOPICS, multi_label=True)
    main_topics = [
        label for label, score in zip(main_result["labels"], main_result["scores"])
        if score >= TOPIC_THRESHOLD
    ]

    # Step 2: Predict subtopics only for selected main topics
    subtopics = {}
    for topic in main_topics:
        sub_labels = CANDIDATE_SUBTOPICS.get(topic, [])
        if not sub_labels:
            continue

        sub_result = classifier(text, sub_labels, multi_label=True)
        filtered_subtopics = [
            label for label, score in zip(sub_result["labels"], sub_result["scores"])
            if score >= SUBTOPIC_THRESHOLD
        ]
        if filtered_subtopics:
            subtopics[topic] = filtered_subtopics

    return main_topics, subtopics
