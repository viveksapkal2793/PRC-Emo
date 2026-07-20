from dataclasses import dataclass


@dataclass
class EmotionSample:
    text: str
    label: str
    target_speaker: str = ""
    target_utterance: str = ""
