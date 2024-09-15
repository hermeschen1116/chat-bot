import torch
from torch import Tensor


def generate_dummy_representation(target_emotion: int) -> Tensor:
    while True:
        dummy = torch.clamp(torch.rand(7, dtype=torch.float32), -1, 1)
        if torch.argmax(dummy) == target_emotion:
            return dummy


def get_sentiment_composition(analysis_result: list) -> Tensor:
    sentiment_possibility: dict = {}
    for emotion_score in analysis_result[0]:
        values: list = list(emotion_score.values())
        sentiment_possibility[values[0]] = values[1]

    emotions: list = [
        "neutral",
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
    ]
    sentiment_composition: list = [
        sentiment_possibility[emotion] for emotion in emotions
    ]

    return torch.softmax(
        torch.tensor(sentiment_composition), dim=-1, dtype=torch.float32
    )