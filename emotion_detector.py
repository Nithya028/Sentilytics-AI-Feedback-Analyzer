from transformers import AutoTokenizer
import onnxruntime as ort
import torch
import numpy as np

EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

ACTIVATION_LEVELS = {
    "High": {"grief", "rage", "excitement", "fear"},
    "Medium": {"joy", "anger", "love", "sadness", "curiosity", "remorse"},
    "Low": {"relief", "neutral", "confusion", "pride", "disappointment"}
}

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions-onnx")
session = ort.InferenceSession("models/model.onnx", providers=["CPUExecutionProvider"])

def get_activation_level(emotion):
    for level, emotions in ACTIVATION_LEVELS.items():
        if emotion.lower() in emotions:
            return level
    return "Medium"

def classify_emotions(text, top_k=5):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    ort_inputs = {
        session.get_inputs()[0].name: inputs["input_ids"].astype(np.int64),
        session.get_inputs()[1].name: inputs["attention_mask"].astype(np.int64),
    }
    ort_outs = session.run(None, ort_inputs)

    logits = torch.tensor(ort_outs[0])
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
    top_probs, top_idxs = torch.topk(probs, top_k)

    results = []
    for i in range(top_k):
        idx = top_idxs[i].item()
        emotion = EMOTIONS[idx]
        confidence = round(top_probs[i].item(), 2)
        results.append({
            "emotion": emotion,
            "activation": get_activation_level(emotion),
            "intensity": confidence
        })
    return results
