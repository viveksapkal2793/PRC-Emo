import argparse
import re
from typing import Dict, Optional, Sequence, Tuple

from .types import EmotionSample
from .utils import normalize_text, section_between

PROMPT_SECTION_BOUNDARY = (
    r"\n\s*### (?:"
    r"Given the following conversation as a context|"
    r"Visual Expressions of the speaker present in this utterance|"
    r"Audio Characteristics of the speaker present in this utterance|"
    r"Audio and Visual Descriptions of the speaker present in this utterance|"
    r"Given the characteristic of this speaker|"
    r"Given the speaker(?:'|’)?s Explicit Emotion Interpretation and Implicit Emotion Interpretation|"
    r"Semantic Contrastive Cues for the speaker in this utterance|"
    r"Reference Similar Emotional Expressions|"
    r"Available emotion labels"
    r")"
)


def extract_prompt_section(text: str, start_pattern: str) -> str:
    return section_between(text, start_pattern, [PROMPT_SECTION_BOUNDARY])


def parse_user_target(user_text: str) -> Tuple[str, str]:
    user_text = normalize_text(user_text)
    match = re.search(
        r"emotional label of\s+(.+?)\s+in the utterance\s+([\"'])(.+?)\2",
        user_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return normalize_text(match.group(1)), normalize_text(match.group(3))
    return "", ""


def parse_prompt_sample(record: Dict, args: argparse.Namespace) -> Optional[EmotionSample]:
    messages = record.get("messages", [])
    if len(messages) < 3:
        return None

    system_text = normalize_text(messages[0].get("content", ""))
    user_text = normalize_text(messages[1].get("content", ""))
    label = normalize_text(messages[-1].get("content", "")).lower()
    target_speaker, target_utterance = parse_user_target(user_text)

    context = extract_prompt_section(
        system_text,
        r"### Given the following conversation as a context\s*",
    )
    speaker_desc = extract_prompt_section(
        system_text,
        r"### Given the characteristic of this speaker:\s*",
    )
    emotion_block = extract_prompt_section(
        system_text,
        r"### Given the speaker(?:'|’)?s Explicit Emotion Interpretation and Implicit Emotion Interpretation.*?:\s*",
    )
    visual_desc = extract_prompt_section(
        system_text,
        r"### Visual Expressions of the speaker present in this utterance:\s*",
    )
    audio_desc = extract_prompt_section(
        system_text,
        r"### Audio Characteristics of the speaker present in this utterance:\s*",
    )
    llm_audio_visual_desc = extract_prompt_section(
        system_text,
        r"### Audio and Visual Descriptions of the speaker present in this utterance:\s*",
    )
    semantic_cues = extract_prompt_section(
        system_text,
        r"### Semantic Contrastive Cues for the speaker in this utterance:\s*",
    )
    reference_similar = extract_prompt_section(
        system_text,
        r"### Reference Similar Emotional Expressions:\s*",
    )

    explicit = ""
    implicit = ""
    explicit_match = re.search(
        r"Explicit Emotion Interpretation:\s*(.*?)(?:\n-\s*Implicit Emotion Interpretation:|\Z)",
        emotion_block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    implicit_match = re.search(
        r"Implicit Emotion Interpretation:\s*(.*)",
        emotion_block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if explicit_match:
        explicit = normalize_text(explicit_match.group(1).lstrip("- "))
    if implicit_match:
        implicit = normalize_text(implicit_match.group(1).lstrip("- "))

    embedding_text = build_embedding_input(
        context=context,
        target_speaker=target_speaker,
        target_utterance=target_utterance,
        explicit=explicit,
        implicit=implicit,
        speaker_desc=speaker_desc,
        visual_desc=visual_desc,
        audio_desc=audio_desc,
        llm_audio_visual_desc=llm_audio_visual_desc,
        semantic_cues=semantic_cues,
        reference_similar=reference_similar,
        args=args,
    )
    return EmotionSample(embedding_text, label, target_speaker, target_utterance)


def build_embedding_input(
    context: str,
    target_speaker: str,
    target_utterance: str,
    explicit: str,
    implicit: str,
    speaker_desc: str,
    visual_desc: str,
    audio_desc: str,
    llm_audio_visual_desc: str,
    semantic_cues: str,
    reference_similar: str,
    args: argparse.Namespace,
) -> str:
    include_context = args.include_conversation_context or args.include_target_speaker
    instruction = (
        "Represent the emotional state and conversational behavior of the target utterance."
        if include_context
        else "Represent the emotional state of the utterance."
    )
    sections = [
        f"Instruct: {instruction}",
        f"Target Utterance:\n\"{target_utterance}\"",
    ]
    if include_context:
        sections.insert(1, f"Conversation Context:\n{context}")
        sections.insert(2, f"Target Speaker:\n{target_speaker}")
    if args.include_visual_description:
        sections.append(f"Visual Description:\n{visual_desc or 'Not available.'}")
    if args.include_audio_description:
        sections.append(f"Audio Description:\n{audio_desc or 'Not available.'}")
    if args.include_llm_aud_vis_desc:
        sections.append(f"Audio and Visual Descriptions:\n{llm_audio_visual_desc or 'Not available.'}")
    if args.include_speaker_description:
        sections.append(f"Speaker Description:\n{speaker_desc or 'Not available.'}")
    if args.include_explicit_emotion:
        sections.append(f"Explicit Emotion:\n{explicit or 'Not available.'}")
    if args.include_implicit_emotion:
        sections.append(f"Implicit Emotion:\n{implicit or 'Not available.'}")
    if args.include_semantic_contrastive_cues:
        sections.append(f"Semantic Contrastive Cues:\n{semantic_cues or 'Not available.'}")
    if args.include_reference_similar_emotions:
        sections.append(f"Reference Similar Emotions:\n{reference_similar or 'Not available.'}")
    return "\n\n".join(sections)
