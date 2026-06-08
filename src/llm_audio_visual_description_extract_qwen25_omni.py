import argparse
import json
import os
import re
import sys
import time
import traceback
import wave
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig

try:
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
except ImportError as exc:
    raise ImportError(
        "This script requires a transformers build with Qwen2_5OmniProcessor and "
        "Qwen2_5OmniThinkerForConditionalGeneration. Activate the same environment "
        "used for Qwen2.5-Omni training."
    ) from exc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
dataset_name = "meld"
prompt_type = "AudioVisualDescriptions_V1"

DEFAULT_MODEL_PATH = "/scratch/data/bikash_rs/Vivek/PRC-Emo/models/qwen_2_5_omni_7b"
DEFAULT_DATA_FOLDER = "./data"
DEFAULT_VIDEO_DIRS = {
    "train": "/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/train_splits",
    "valid": "/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/dev_splits_complete",
    "test": "/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/output_repeated_splits_test",
}
DEFAULT_AUDIO_DIRS = {
    "train": "/scratch/data/bikash_rs/Vivek/dataset/MELD_audio/train",
    "valid": "/scratch/data/bikash_rs/Vivek/dataset/MELD_audio/dev",
    "test": "/scratch/data/bikash_rs/Vivek/dataset/MELD_audio/test",
}

NO_VISUAL_DESCRIPTION = "No visual description available."
NO_AUDIO_DESCRIPTION = "No audio description available."


VISUAL_PROMPT_TEMPLATE = """You are an expert in analyzing facial expressions and non-verbal visual behavior in conversational settings.

Task
Analyze the visible facial expressions, gaze behavior, head movements, posture, and other observable non-verbal actions associated with the target utterance.
Focus only on what is visually observable.
Use natural language descriptions.
Use at least 10 words, but no more than 30 words.
Do NOT infer emotions, intentions, or hidden feelings.
Do NOT use emotion labels such as anger, sadness, fear, joy, surprise, disgust, or neutral.
Focus Utterance

"{utt}"

Required JSON Format

{{
"VisualDescription": ""
}}

Critical Rules
Output ONLY the JSON object.
No explanations or thinking processes.
No markdown or code formatting.
Keys must be exactly as shown.
Focus on observable visual behavior.
Do NOT use emotion labels, emotional states, or emotional descriptors such as:
anger, sadness, fear, joy, surprise, disgust, neutral, happy, upset, calm, excited, worried, frustrated.

Example Invalid Response:
{{
  "VisualDescription": "The speaker maintains a neutral facial expression, ..."
}}

Reason:
This describes an emotional state rather than observable behavior.

Example Valid Response:
{{
"VisualDescription": "The speaker displays moderate facial expressiveness with brief eyebrow movement, controlled mouth activity, and steady visual engagement, showing occasional moments of visible hesitation during delivery."
}}"""


AUDIO_PROMPT_TEMPLATE = """You are an expert in analyzing vocal delivery and observable speech characteristics in conversational settings.

Task
Analyze the vocal characteristics associated with the target utterance.
Describe observable speech properties such as pitch variation, speaking rate, vocal energy, pauses, hesitations, emphasis, articulation, and voice stability.
Focus only on observable acoustic behavior.
Use natural language descriptions.
Use at least 10 words, but no more than 30 words.
Do NOT infer emotions, intentions, or hidden feelings.
Do NOT use emotion labels such as anger, sadness, fear, joy, surprise, disgust, or neutral.
Focus Utterance

"{utt}"

Required JSON Format

{{
"AudioDescription": ""
}}

Critical Rules
Output ONLY the JSON object.
No explanations or thinking processes.
No markdown or code formatting.
Keys must be exactly as shown.
Focus on observable vocal behavior.
Do NOT use emotion labels, emotional states, or emotional descriptors such as:
anger, sadness, fear, joy, surprise, disgust, neutral, happy, upset, calm, excited, worried, frustrated.


Example Invalid Response:
{{
  "AudioDescription": "The speaker sounds angry and upset, with a raised voice and fast speaking rate..."
}}

Example Valid Response:
{{
"AudioDescription": "The speaker uses a moderately paced delivery with stable vocal energy, slight variation in emphasis, and occasional pauses that introduce minor hesitations during speech."
}}"""


class BatchPreprocessorLLMAudioVisualDescriptions:
    @staticmethod
    def load_raw_data(path_data):
        raw_data = json.load(open(path_data, encoding="utf-8"))
        if isinstance(raw_data, dict):
            new_data_list = []
            for k, v in raw_data.items():
                v["s_id"] = k
                new_data_list.append(v)
            return new_data_list
        if isinstance(raw_data, list):
            return raw_data
        raise ValueError(f"Unsupported data format in {path_data}")

    def preprocess(self, all_conversations, selected_utterance_limit: Optional[int] = None):
        entries = []
        for sample in all_conversations:
            conv_id = sample["s_id"]
            type_data = sample["type_data"]
            for utter_idx, utt in enumerate(sample["sentences"]):
                speaker = sample.get("speakers", [f"SPEAKER_{utter_idx}"])[utter_idx]
                tagged_utt = f'{speaker}: "{utt}"'
                entries.append(
                    {
                        "conv_id": conv_id,
                        "utter_idx": utter_idx,
                        "utterance": tagged_utt,
                        "type_data": type_data,
                        "all_speakers": sample.get("speakers", []),
                    }
                )
                if selected_utterance_limit is not None and len(entries) >= selected_utterance_limit:
                    return entries
        return entries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MELD visual and audio descriptions with Qwen2.5-Omni."
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_folder", type=str, default=DEFAULT_DATA_FOLDER)
    parser.add_argument("--output_folder", type=str, default=DEFAULT_DATA_FOLDER)
    parser.add_argument("--splits", nargs="+", default=["valid", "test", "train"], choices=["train", "valid", "test"])
    parser.add_argument("--debug", action="store_true", help="Process only 10 train utterances and log per-utterance time.")
    parser.add_argument("--debug_utterances", type=int, default=10)
    parser.add_argument("--max_media_seconds", type=float, default=30.0)
    parser.add_argument("--video_num_frames", type=int, default=4, help="Lower is faster; 4 is a good quality/speed default.")
    parser.add_argument("--video_fps", type=float, default=None, help="Optional FPS sampling. If set, overrides dense loading.")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--batch_by_modality",
        action="store_true",
        help="Faster: process all visual prompts first, then all audio prompts. Default keeps visual then audio per utterance.",
    )
    parser.add_argument("--attn_implementation", type=str, default=None, choices=[None, "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--compile_model", action="store_true", help="Experimental; may not help quantized multimodal models.")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--save_every", type=int, default=25, help="Save partial output every N utterances per split.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meld_train_video_dir", type=str, default=DEFAULT_VIDEO_DIRS["train"])
    parser.add_argument("--meld_valid_video_dir", type=str, default=DEFAULT_VIDEO_DIRS["valid"])
    parser.add_argument("--meld_test_video_dir", type=str, default=DEFAULT_VIDEO_DIRS["test"])
    parser.add_argument("--meld_train_audio_dir", type=str, default=DEFAULT_AUDIO_DIRS["train"])
    parser.add_argument("--meld_valid_audio_dir", type=str, default=DEFAULT_AUDIO_DIRS["valid"])
    parser.add_argument("--meld_test_audio_dir", type=str, default=DEFAULT_AUDIO_DIRS["test"])
    parser.add_argument(
        "--meld_train_media_id_offset",
        type=int,
        default=0,
        help="Subtract this from train JSON dialogue ids when building MELD media filenames.",
    )
    parser.add_argument(
        "--meld_valid_media_id_offset",
        type=int,
        default=1039,
        help="Subtract this from valid JSON dialogue ids when building MELD media filenames.",
    )
    parser.add_argument(
        "--meld_test_media_id_offset",
        type=int,
        default=1153,
        help="Subtract this from test JSON dialogue ids when building MELD media filenames.",
    )
    return parser.parse_args()


def get_video_dirs(args):
    return {
        "train": args.meld_train_video_dir,
        "valid": args.meld_valid_video_dir,
        "test": args.meld_test_video_dir,
    }


def get_audio_dirs(args):
    return {
        "train": args.meld_train_audio_dir,
        "valid": args.meld_valid_audio_dir,
        "test": args.meld_test_audio_dir,
    }


def get_media_id_offsets(args):
    return {
        "train": args.meld_train_media_id_offset,
        "valid": args.meld_valid_media_id_offset,
        "test": args.meld_test_media_id_offset,
    }


def get_media_dialogue_id(conv_id: str, split: str, media_id_offsets: Dict[str, int]):
    try:
        media_dialogue_id = int(conv_id) - media_id_offsets.get(split, 0)
    except ValueError:
        return conv_id
    if media_dialogue_id < 0:
        raise ValueError(
            f"Media dialogue id became negative for split={split}, conv_id={conv_id}, "
            f"offset={media_id_offsets.get(split, 0)}."
        )
    return str(media_dialogue_id)


def build_media_path(base_dir: str, conv_id: str, utter_idx: int, suffix: str):
    return os.path.join(base_dir, f"dia{conv_id}_utt{utter_idx}.{suffix}")


def get_duration_with_av(path: str) -> Optional[float]:
    try:
        import av

        container = av.open(path)
        duration = None
        if container.duration is not None:
            duration = float(container.duration) / 1_000_000.0
        if duration is None:
            stream_durations = []
            for stream in container.streams:
                if stream.duration is not None and stream.time_base is not None:
                    stream_durations.append(float(stream.duration * stream.time_base))
            duration = max(stream_durations) if stream_durations else None
        container.close()
        return duration
    except Exception:
        return None


def validate_video_file(video_path: str, max_seconds: float) -> Tuple[bool, str, Optional[float]]:
    if not os.path.exists(video_path):
        return False, f"missing video file: {video_path}", None
    try:
        import av

        container = av.open(video_path)
        duration = None
        if container.duration is not None:
            duration = float(container.duration) / 1_000_000.0
        if duration is None:
            stream_durations = []
            for stream in container.streams:
                if stream.duration is not None and stream.time_base is not None:
                    stream_durations.append(float(stream.duration * stream.time_base))
            duration = max(stream_durations) if stream_durations else None
        container.close()
        if duration is not None and duration > max_seconds:
            return False, f"video too long: duration={duration:.2f}s max={max_seconds:.2f}s", duration
        return True, "", duration
    except Exception as exc:
        return False, f"corrupt video file: {type(exc).__name__}: {exc}", None


def validate_audio_file(audio_path: str, max_seconds: float) -> Tuple[bool, str, Optional[float]]:
    if not os.path.exists(audio_path):
        return False, f"missing audio file: {audio_path}", None
    try:
        with wave.open(audio_path, "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
        duration = frame_count / frame_rate if frame_rate else None
        if duration is not None and duration > max_seconds:
            return False, f"audio too long: duration={duration:.2f}s max={max_seconds:.2f}s", duration
        return True, "", duration
    except Exception as exc:
        return False, f"corrupt audio file: {type(exc).__name__}: {exc}", None


def extract_json_field(text: str, key: str, fallback: str) -> str:
    text = text.strip()
    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            value = parsed.get(key, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            pass

    key_match = re.search(rf'"?{re.escape(key)}"?\s*:\s*"([^"]+)"', text, re.DOTALL)
    if key_match:
        return key_match.group(1).strip()

    colon_pos = text.rfind(f"{key}:")
    if colon_pos != -1:
        value = text[colon_pos + len(key) + 1 :].strip()
        value = value.strip("{} \n\t\"")
        if value:
            return value

    return fallback


def build_messages(prompt: str, modality: str, media_path: str):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": modality, modality: media_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def move_inputs_to_device(inputs, model):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(model.device)
        else:
            moved[key] = value
    return moved


def generate_batch(model, processor, batch_items, args):
    conversations = [item["messages"] for item in batch_items]
    template_kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "padding": True,
        "num_frames": args.video_num_frames,
        "load_audio_from_video": False,
        "use_audio_in_video": False,
    }
    if args.video_fps is not None:
        template_kwargs["fps"] = args.video_fps
    inputs = processor.apply_chat_template(conversations, **template_kwargs)
    inputs = move_inputs_to_device(inputs, model)
    input_width = inputs["input_ids"].shape[1]

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p if args.temperature > 0 else None,
        "repetition_penalty": args.repetition_penalty,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "use_cache": not args.no_cache,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)

    decoded = []
    for row_idx in range(len(batch_items)):
        generated_part = generated[row_idx, input_width:]
        decoded.append(
            processor.batch_decode(
                generated_part.unsqueeze(0),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        )
    return decoded


def load_model_and_processor(args):
    print("Loading Qwen2.5-Omni model in 4-bit quantization ...")
    tensor_data_type = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=tensor_data_type,
    )
    load_kwargs = {
        "quantization_config": bnb_config,
        "torch_dtype": tensor_data_type,
        "device_map": "auto",
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    if args.compile_model:
        model = torch.compile(model)
    model.eval()

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    print("Model loaded successfully.")
    return model, processor


def init_output_for_conversation(sample):
    speakers = sample.get("speakers", [])
    utterances = [
        f"{speaker}: {text}" if speakers else text
        for speaker, text in zip(speakers, sample["sentences"])
    ]
    if not utterances:
        utterances = list(sample["sentences"])
    return {
        "utterances": utterances,
        "emotion_predictions": [
            {
                "visual_description": "",
                "audio_description": "",
                "visual_prompt": "",
                "audio_prompt": "",
                "visual_status": "pending",
                "audio_status": "pending",
            }
            for _ in sample["sentences"]
        ],
    }


def save_json(path: str, data):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def chunked(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def apply_generation_results(output, batch_items, generated_texts):
    for item, generated_text in zip(batch_items, generated_texts):
        conv_id = item["conv_id"]
        utter_idx = item["utter_idx"]
        modality = item["modality"]
        key = "VisualDescription" if modality == "video" else "AudioDescription"
        desc_key = "visual_description" if modality == "video" else "audio_description"
        status_key = "visual_status" if modality == "video" else "audio_status"
        fallback = NO_VISUAL_DESCRIPTION if modality == "video" else NO_AUDIO_DESCRIPTION
        output[conv_id]["emotion_predictions"][utter_idx][desc_key] = extract_json_field(
            generated_text, key, fallback
        )
        output[conv_id]["emotion_predictions"][utter_idx][status_key] = "generated"
        output[conv_id]["emotion_predictions"][utter_idx][f"{desc_key}_raw"] = generated_text


def run_batch_safely(model, processor, batch_items, args):
    try:
        return generate_batch(model, processor, batch_items, args)
    except Exception:
        if len(batch_items) == 1:
            raise
        outputs = []
        for item in batch_items:
            outputs.extend(run_batch_safely(model, processor, [item], args))
        return outputs


def prepare_generation_items(entries, output, args, video_dirs, audio_dirs, media_id_offsets):
    items_by_utterance = []
    media_cache: Dict[Tuple[str, str], Tuple[bool, str, Optional[float]]] = {}

    for entry in entries:
        conv_id = entry["conv_id"]
        utter_idx = entry["utter_idx"]
        type_data = entry["type_data"]
        utterance = entry["utterance"]

        visual_prompt = VISUAL_PROMPT_TEMPLATE.format(utt=utterance)
        audio_prompt = AUDIO_PROMPT_TEMPLATE.format(utt=utterance)
        pred = output[conv_id]["emotion_predictions"][utter_idx]
        pred["visual_prompt"] = visual_prompt
        pred["audio_prompt"] = audio_prompt

        media_dialogue_id = get_media_dialogue_id(conv_id, type_data, media_id_offsets)
        video_path = build_media_path(video_dirs[type_data], media_dialogue_id, utter_idx, "mp4")
        audio_path = build_media_path(audio_dirs[type_data], media_dialogue_id, utter_idx, "wav")
        pred["media_dialogue_id"] = media_dialogue_id
        pred["video_path"] = video_path
        pred["audio_path"] = audio_path

        if pred.get("visual_status") == "generated" and pred.get("visual_description"):
            video_item = None
        else:
            video_key = ("video", video_path)
            if video_key not in media_cache:
                media_cache[video_key] = validate_video_file(video_path, args.max_media_seconds)
            video_valid, video_reason, video_duration = media_cache[video_key]
            pred["visual_media_duration"] = video_duration
            if video_valid:
                video_item = {
                    "conv_id": conv_id,
                    "utter_idx": utter_idx,
                    "modality": "video",
                    "messages": build_messages(visual_prompt, "video", video_path),
                }
            else:
                video_item = None
                pred["visual_description"] = NO_VISUAL_DESCRIPTION
                pred["visual_status"] = video_reason

        if pred.get("audio_status") == "generated" and pred.get("audio_description"):
            audio_item = None
        else:
            audio_key = ("audio", audio_path)
            if audio_key not in media_cache:
                media_cache[audio_key] = validate_audio_file(audio_path, args.max_media_seconds)
            audio_valid, audio_reason, audio_duration = media_cache[audio_key]
            pred["audio_media_duration"] = audio_duration
            if audio_valid:
                audio_item = {
                    "conv_id": conv_id,
                    "utter_idx": utter_idx,
                    "modality": "audio",
                    "messages": build_messages(audio_prompt, "audio", audio_path),
                }
            else:
                audio_item = None
                pred["audio_description"] = NO_AUDIO_DESCRIPTION
                pred["audio_status"] = audio_reason

        items_by_utterance.append((video_item, audio_item))

    return items_by_utterance


def process_items_per_utterance(model, processor, output, items_by_utterance, args, output_path):
    processed_utterances = 0
    for visual_item, audio_item in tqdm(items_by_utterance, desc="Utterances"):
        start_time = time.perf_counter()
        for item in (visual_item, audio_item):
            if item is None:
                continue
            try:
                generated_texts = run_batch_safely(model, processor, [item], args)
                apply_generation_results(output, [item], generated_texts)
            except Exception as exc:
                traceback.print_exc()
                pred = output[item["conv_id"]]["emotion_predictions"][item["utter_idx"]]
                if item["modality"] == "video":
                    pred["visual_description"] = NO_VISUAL_DESCRIPTION
                    pred["visual_status"] = f"inference error: {type(exc).__name__}: {exc}"
                else:
                    pred["audio_description"] = NO_AUDIO_DESCRIPTION
                    pred["audio_status"] = f"inference error: {type(exc).__name__}: {exc}"

        processed_utterances += 1
        if args.debug:
            elapsed = time.perf_counter() - start_time
            print(f"[debug timing] utterance {processed_utterances}: {elapsed:.2f}s")
        if args.save_every > 0 and processed_utterances % args.save_every == 0:
            save_json(output_path, output)


def process_items_batched_by_modality(model, processor, output, items_by_utterance, args, output_path):
    visual_items = [pair[0] for pair in items_by_utterance if pair[0] is not None]
    audio_items = [pair[1] for pair in items_by_utterance if pair[1] is not None]
    processed_batches = 0
    for modality_name, modality_items in (("visual", visual_items), ("audio", audio_items)):
        for batch_items in tqdm(list(chunked(modality_items, args.batch_size)), desc=f"{modality_name} batches"):
            try:
                generated_texts = run_batch_safely(model, processor, batch_items, args)
                apply_generation_results(output, batch_items, generated_texts)
            except Exception as exc:
                traceback.print_exc()
                for item in batch_items:
                    pred = output[item["conv_id"]]["emotion_predictions"][item["utter_idx"]]
                    if item["modality"] == "video":
                        pred["visual_description"] = NO_VISUAL_DESCRIPTION
                        pred["visual_status"] = f"inference error: {type(exc).__name__}: {exc}"
                    else:
                        pred["audio_description"] = NO_AUDIO_DESCRIPTION
                        pred["audio_status"] = f"inference error: {type(exc).__name__}: {exc}"
            processed_batches += 1
            if args.save_every > 0 and processed_batches % args.save_every == 0:
                save_json(output_path, output)


def output_path_for_split(args, split):
    model_name = os.path.basename(os.path.normpath(args.model_path))
    debug_suffix = "_debug" if args.debug else ""
    return os.path.join(args.output_folder, f"{dataset_name}.{split}_{prompt_type}_{model_name}{debug_suffix}.json")


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    torch.manual_seed(args.seed)

    if args.debug:
        args.splits = ["train"]
        args.overwrite = True
        print(f"- Debug mode: processing first {args.debug_utterances} train utterances only.")

    os.makedirs(args.output_folder, exist_ok=True)
    model, processor = load_model_and_processor(args)
    data_preprocessor = BatchPreprocessorLLMAudioVisualDescriptions()
    video_dirs = get_video_dirs(args)
    audio_dirs = get_audio_dirs(args)
    media_id_offsets = get_media_id_offsets(args)

    for split in args.splits:
        data_name_pattern = f"{dataset_name}.{split}"
        raw_path = os.path.join(args.data_folder, f"{data_name_pattern}.json")
        output_path = output_path_for_split(args, split)

        raw_data = data_preprocessor.load_raw_data(raw_path)
        for sample in raw_data:
            sample["type_data"] = split

        selected_limit = args.debug_utterances if args.debug else None
        entries = data_preprocessor.preprocess(raw_data, selected_utterance_limit=selected_limit)
        entry_keys = {(entry["conv_id"], entry["utter_idx"]) for entry in entries}

        if os.path.exists(output_path) and not args.overwrite:
            output = json.load(open(output_path, encoding="utf-8"))
            print(f"- Loaded existing output for {split}: {len(output)} conversations")
        else:
            output = {}

        for sample in raw_data:
            conv_id = sample["s_id"]
            if not any(key[0] == conv_id for key in entry_keys):
                continue
            if conv_id not in output:
                output[conv_id] = init_output_for_conversation(sample)

        items_by_utterance = prepare_generation_items(entries, output, args, video_dirs, audio_dirs, media_id_offsets)
        if args.batch_by_modality:
            process_items_batched_by_modality(model, processor, output, items_by_utterance, args, output_path)
        else:
            process_items_per_utterance(model, processor, output, items_by_utterance, args, output_path)

        save_json(output_path, output)
        print(f"- Saved {len(output)} conversations for {split}: {output_path}")

    print("- Extraction complete.")


if __name__ == "__main__":
    main()
