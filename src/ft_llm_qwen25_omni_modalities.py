import argparse
import glob
import json
import os
import shutil
from typing import Iterable, List, Optional, Tuple

from PIL import Image

from ft_llm_qwen25_omni import (
    CurriculumDataset,
    JsonlMessageDataset,
    MultimodalTrainer,
    OmniEmotionDataCollator,
    attach_lora,
    build_quant_config,
    create_training_args,
    get_label_map,
    load_base_model,
    load_peft_model,
    load_processor,
    process,
    set_random_seed,
    torch,
)


def determine_selected_modalities(args) -> List[str]:
    if args.text_only_input:
        return []

    explicit_selection = args.video_input or args.image_input or args.audio_input
    if explicit_selection:
        selected = []
        if args.video_input:
            selected.append("video")
        if args.image_input:
            selected.append("image")
        if args.audio_input:
            selected.append("audio")
        return selected

    return []


def iter_media_paths_from_content(content) -> Iterable[Tuple[str, str]]:
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "video" and item.get("video"):
                yield "video", item["video"]
            elif item_type == "audio" and item.get("audio"):
                yield "audio", item["audio"]
            elif item_type == "image" and item.get("image"):
                yield "image", item["image"]


def _filter_message_content_modalities(content, keep_modalities: Optional[List[str]] = None):
    if not isinstance(content, list):
        return content

    keep_modalities = set(keep_modalities or [])
    filtered_content = []
    for item in content:
        if not isinstance(item, dict):
            filtered_content.append(item)
            continue

        item_type = item.get("type")
        if item_type == "text":
            filtered_content.append(item)
        elif item_type in {"video", "audio", "image"} and item_type in keep_modalities:
            filtered_content.append(item)

    return filtered_content


def enforce_selected_modalities(sample, selected_modalities: Optional[List[str]] = None, text_only_input: bool = False):
    normalized_sample = dict(sample)
    keep_modalities = [] if text_only_input else list(selected_modalities or [])

    if text_only_input or selected_modalities:
        normalized_messages = []
        for message in sample.get("messages", []):
            normalized_message = dict(message)
            normalized_message["content"] = _filter_message_content_modalities(
                message.get("content"),
                keep_modalities=keep_modalities,
            )
            normalized_messages.append(normalized_message)
        normalized_sample["messages"] = normalized_messages

    media_path_keys = {
        "video": "video_path",
        "audio": "audio_path",
        "image": "image_path",
    }
    if text_only_input:
        for sample_key in media_path_keys.values():
            normalized_sample.pop(sample_key, None)
    elif selected_modalities:
        allowed = set(selected_modalities)
        for media_type, sample_key in media_path_keys.items():
            if media_type not in allowed:
                normalized_sample.pop(sample_key, None)

    return normalized_sample


def extract_media_paths(sample, selected_modalities: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    media_paths = []
    for message in sample.get("messages", []):
        media_paths.extend(iter_media_paths_from_content(message.get("content")))

    for media_key in ("video_path", "audio_path", "image_path"):
        media_path = sample.get(media_key)
        if media_path:
            media_type = media_key.replace("_path", "")
            media_paths.append((media_type, media_path))

    deduped = []
    seen = set()
    for media_type, media_path in media_paths:
        if selected_modalities and media_type not in selected_modalities:
            continue
        key = (media_type, media_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def validate_video_file(video_path: str, media_cache: dict) -> Tuple[bool, Optional[str]]:
    cache_key = ("video", video_path)
    if cache_key in media_cache:
        return media_cache[cache_key]

    if not os.path.exists(video_path):
        result = (False, f"missing video file: {video_path}")
        media_cache[cache_key] = result
        return result

    try:
        import av

        container = av.open(video_path)
        container.close()
        result = (True, None)
    except Exception as exc:
        result = (False, f"{type(exc).__name__}: {exc}")

    media_cache[cache_key] = result
    return result


def validate_audio_file(audio_path: str, media_cache: dict) -> Tuple[bool, Optional[str]]:
    cache_key = ("audio", audio_path)
    if cache_key in media_cache:
        return media_cache[cache_key]

    if not os.path.exists(audio_path):
        result = (False, f"missing audio file: {audio_path}")
    else:
        result = (True, None)

    media_cache[cache_key] = result
    return result


def validate_image_file(image_path: str, media_cache: dict) -> Tuple[bool, Optional[str]]:
    cache_key = ("image", image_path)
    if cache_key in media_cache:
        return media_cache[cache_key]

    if image_path is None or not os.path.exists(image_path):
        result = (False, f"missing image file: {image_path}")
        media_cache[cache_key] = result
        return result

    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
        result = (True, None)
    except Exception as exc:
        result = (False, f"{type(exc).__name__}: {exc}")

    media_cache[cache_key] = result
    return result


def filter_invalid_media_records(records, split_name: str, selected_modalities: Optional[List[str]] = None, media_cache: Optional[dict] = None):
    media_cache = media_cache if media_cache is not None else {}
    filtered_records = []
    skipped_records = []

    for sample in records:
        sample_media = extract_media_paths(sample, selected_modalities=selected_modalities)
        is_valid = True
        failure_reason = None

        for media_type, media_path in sample_media:
            if media_type == "video":
                valid, reason = validate_video_file(media_path, media_cache)
            elif media_type == "audio":
                valid, reason = validate_audio_file(media_path, media_cache)
            else:
                valid, reason = validate_image_file(media_path, media_cache)

            if not valid:
                is_valid = False
                failure_reason = f"{media_type} {reason}"
                break

        if selected_modalities:
            present_media_types = {media_type for media_type, _ in sample_media}
            missing_modalities = [media_type for media_type in selected_modalities if media_type not in present_media_types]
            if missing_modalities:
                is_valid = False
                failure_reason = f"missing selected modalities: {', '.join(missing_modalities)}"

        if is_valid:
            filtered_records.append(sample)
        else:
            skipped_records.append(
                (
                    sample.get("conversation_id"),
                    sample.get("utterance_id"),
                    failure_reason,
                )
            )

    if skipped_records:
        print(f"[media validation] Skipped {len(skipped_records)} invalid samples from {split_name}.")
        for conversation_id, utterance_id, reason in skipped_records[:10]:
            print(
                "[media validation] "
                f"{split_name} conversation_id={conversation_id} utterance_id={utterance_id}: {reason}"
            )
        if len(skipped_records) > 10:
            print(f"[media validation] ... and {len(skipped_records) - 10} more skipped samples.")

    return filtered_records


def load_jsonl_dataset(path, split_name: str, args, selected_modalities: List[str], media_cache: Optional[dict] = None):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample = enforce_selected_modalities(
                sample,
                selected_modalities=selected_modalities if selected_modalities else None,
                text_only_input=args.text_only_input,
            )
            records.append(sample)

    if args.skip_invalid_media:
        records = filter_invalid_media_records(
            records,
            split_name=split_name,
            selected_modalities=None if args.text_only_input else (selected_modalities if selected_modalities else None),
            media_cache=media_cache,
        )

    return JsonlMessageDataset(records)


def build_trainer(model, processor, train_dataset, eval_dataset, training_args, label_space, args):
    return MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=OmniEmotionDataCollator(
            processor,
            video_fps=args.video_fps,
            video_num_frames=args.video_num_frames,
            text_only_input=args.text_only_input,
        ),
        processor=processor,
        label_space=label_space,
        video_fps=args.video_fps,
        video_num_frames=args.video_num_frames,
        text_only_input=args.text_only_input,
        generation_max_new_tokens=args.generation_max_new_tokens,
    )


def save_results_json(result_path, key, metrics):
    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}
    all_results[key] = metrics
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def build_preprocessed_paths(args) -> List[str]:
    return [
        f"{args.data_folder}/{args.data_name}.{d_type}.{args.kshot}shot_w{args.window}_{args.prompting_type}_{args.extract_prompting_llm_id}{args.preprocessed_data_suffix}"
        for d_type in ["train", "valid", "test"]
    ]


def maybe_generate_data(data_paths, args):
    if not args.re_gen_data:
        return

    process(data_paths, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5 Omni Thinker on MELD emotion classification with selectable media modalities."
    )
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval_test", action="store_true", default=False)
    parser.add_argument("--do_eval_dev", action="store_true", default=False)
    parser.add_argument("--ft_model_path", type=str, default=None)
    parser.add_argument("--ft_model_id", type=str, default=None)
    parser.add_argument("--prompting_type", type=str, default="ImplicitEmotion_V3")
    parser.add_argument("--base_model_id", type=str, required=True)
    parser.add_argument("--extract_prompting_llm_id", type=str, default="qwen_3_14b")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kshot", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument("--eval_delay", type=int, default=100000)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--re_gen_data", action="store_true", default=False)
    parser.add_argument("--data_name", type=str, default="meld")
    parser.add_argument("--data_folder", type=str, default="./data/")
    parser.add_argument("--output_folder", type=str, default="./finetuned_llm/")
    parser.add_argument("--curriculum", action="store_true", default=False)
    parser.add_argument("--bucket_number", type=int, default=8)
    parser.add_argument("--curriculum_update_epochs", type=int, default=None)
    parser.add_argument("--video_fps", type=float, default=1.0)
    parser.add_argument("--video_num_frames", type=int, default=1)
    parser.add_argument("--text_only_input", action="store_true", default=False)
    parser.add_argument("-video_input", "--video_input", action="store_true", default=False)
    parser.add_argument("-image_input", "--image_input", action="store_true", default=False)
    parser.add_argument("-audio_input", "--audio_input", action="store_true", default=False)
    parser.add_argument("--generation_max_new_tokens", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")
    parser.add_argument("--multimodal_chat_format", action="store_true", default=True)
    parser.add_argument("--skip_invalid_media", action="store_true", default=False)
    parser.add_argument("--preprocessed_data_suffix", type=str, default="_Aud_Vis_Omni_text_only.jsonl")
    parser.add_argument("--meld_train_video_dir", type=str, default="/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/train_splits")
    parser.add_argument("--meld_valid_video_dir", type=str, default="/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/dev_splits_complete")
    parser.add_argument("--meld_test_video_dir", type=str, default="/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/output_repeated_splits_test")
    parser.add_argument("--meld_train_audio_dir", type=str, default="/scratch/data/bikash_rs/Vivek/dataset/MELD_audio/train")
    parser.add_argument("--meld_valid_audio_dir", type=str, default="/scratch/data/bikash_rs/Vivek/dataset/MELD_audio/dev")
    parser.add_argument("--meld_test_audio_dir", type=str, default="/scratch/data/bikash_rs/Vivek/dataset/MELD_audio/test")

    args, unknown = parser.parse_known_args()
    if args.data_name != "meld":
        raise ValueError("This Omni modality-selection script is currently wired for MELD only.")

    selected_modalities = determine_selected_modalities(args)
    print(args)
    if args.text_only_input:
        print("Selected input mode: text-only")
    elif selected_modalities:
        print(f"Selected non-text modalities: {selected_modalities}")
    else:
        print("Selected input mode: no explicit modality filter; dataset JSON controls which modalities are used.")

    set_random_seed(args.seed)

    all_path_folder_preprocessed_data = build_preprocessed_paths(args)
    maybe_generate_data(all_path_folder_preprocessed_data, args)

    media_cache = {}
    full_dataset = load_jsonl_dataset(
        all_path_folder_preprocessed_data[0],
        split_name="train",
        args=args,
        selected_modalities=selected_modalities,
        media_cache=media_cache,
    )
    valid_dataset = load_jsonl_dataset(
        all_path_folder_preprocessed_data[1],
        split_name="valid",
        args=args,
        selected_modalities=selected_modalities,
        media_cache=media_cache,
    )
    test_dataset = load_jsonl_dataset(
        all_path_folder_preprocessed_data[2],
        split_name="test",
        args=args,
        selected_modalities=selected_modalities,
        media_cache=media_cache,
    )

    curriculum_manager = None
    if args.curriculum and args.do_train:
        curriculum_manager = CurriculumDataset(list(full_dataset), bucket_number=args.bucket_number, curriculum=True)

    label_space = get_label_map(args.data_name)
    tensor_dtype = torch.bfloat16
    bnb_config = build_quant_config(tensor_dtype)

    if not args.do_train:
        ft_model_path = f"{args.output_folder}/{args.ft_model_id}" if args.ft_model_path is None else args.ft_model_path
        processor = load_processor(ft_model_path)
        model = load_peft_model(ft_model_path, torch.float32, bnb_config)
        eval_args = create_training_args(ft_model_path, 1, args)
        trainer = build_trainer(model, processor, full_dataset, valid_dataset, eval_args, label_space, args)

        if args.do_eval_dev:
            dev_metrics = trainer.evaluate(valid_dataset, metric_key_prefix="dev")
            print(dev_metrics)
        if args.do_eval_test:
            test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
            print(test_metrics)
        raise SystemExit(0)

    if args.curriculum and args.curriculum_update_epochs is None:
        args.curriculum_update_epochs = max(1, (args.epoch or 1) // args.bucket_number)

    model_id = args.base_model_id
    base_output_dir = f"{args.output_folder}/{args.ft_model_id}"

    if args.curriculum:
        print("=" * 50)
        print("Starting curriculum learning with phased multimodal training")
        print("=" * 50)

        remaining_epochs = args.epoch or 0
        model = None
        trainer = None

        for phase in range(args.bucket_number):
            current_dataset = curriculum_manager.get_curriculum_dataset(phase)
            phase_output_dir = f"{base_output_dir}_phase_{phase}"
            os.makedirs(phase_output_dir, exist_ok=True)

            current_epochs = min(args.curriculum_update_epochs, remaining_epochs) if remaining_epochs > 0 else 0
            remaining_epochs -= current_epochs
            if current_epochs <= 0:
                continue

            training_args = create_training_args(phase_output_dir, current_epochs, args)

            if phase == 0:
                processor = load_processor(model_id)
                model = load_base_model(model_id, tensor_dtype, bnb_config)
                model = attach_lora(model, args)
            else:
                prev_phase_dir = f"{base_output_dir}_phase_{phase - 1}"
                del model
                del trainer
                torch.cuda.empty_cache()
                processor = load_processor(prev_phase_dir)
                model = load_peft_model(prev_phase_dir, tensor_dtype, bnb_config)

            trainer = build_trainer(model, processor, current_dataset, valid_dataset, training_args, label_space, args)
            trainer.train()
            trainer.save_model(phase_output_dir)
            processor.save_pretrained(phase_output_dir)

            if remaining_epochs <= 0:
                break

        if remaining_epochs > 0:
            full_phase_output_dir = f"{base_output_dir}_final_full_finetune"
            os.makedirs(full_phase_output_dir, exist_ok=True)
            last_phase_dir = f"{base_output_dir}_phase_{args.bucket_number - 1}"

            del model
            del trainer
            torch.cuda.empty_cache()

            processor = load_processor(last_phase_dir)
            model = load_peft_model(last_phase_dir, tensor_dtype, bnb_config)
            training_args = create_training_args(full_phase_output_dir, remaining_epochs, args)
            trainer = build_trainer(model, processor, full_dataset, valid_dataset, training_args, label_space, args)
            trainer.train()
            trainer.save_model(full_phase_output_dir)
            processor.save_pretrained(full_phase_output_dir)

            final_test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test_final_full")
            save_results_json(os.path.join(base_output_dir, "all_phase_test_results.json"), "final_full", final_test_metrics)
            ft_model_path = full_phase_output_dir
        else:
            ft_model_path = f"{base_output_dir}_phase_{min(phase, args.bucket_number - 1)}"

        phase_dirs = glob.glob(f"{base_output_dir}_phase_*")
        for phase_dir in phase_dirs:
            try:
                shutil.rmtree(phase_dir)
            except Exception as exc:
                print(f"Failed to delete {phase_dir}: {exc}")

        print(f"Training complete. Final checkpoint: {ft_model_path}")

    else:
        print("Starting standard multimodal training without curriculum learning")
        output_dir = f"{args.output_folder}/{args.ft_model_id}"
        os.makedirs(output_dir, exist_ok=True)

        processor = load_processor(model_id)
        model = load_base_model(model_id, tensor_dtype, bnb_config)
        model = attach_lora(model, args)

        training_args = create_training_args(output_dir, args.epoch, args)
        trainer = build_trainer(model, processor, full_dataset, valid_dataset, training_args, label_space, args)
        trainer.train()
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)

        if args.do_eval_test:
            test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
            save_results_json(os.path.join(output_dir, "all_phase_test_results.json"), "test", test_metrics)
