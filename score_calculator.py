def calculate_adorescore(emotions, topics):
    score = 0
    breakdown = {}

    for emotion in emotions:
        intensity = emotion["intensity"]
        name = emotion["emotion"].lower()
        polarity = 1

        if name in {"disappointment", "anger", "sadness", "confusion", "disapproval"}:
            polarity = -1
        elif name in {"neutral"}:
            polarity = 0

        score += intensity * polarity * 100

    score = max(-100, min(100, int(score)))

    for topic in topics:
        breakdown[topic] = score  # Simplified for demonstration

    return {
        "overall": score,
        "breakdown": breakdown
    }