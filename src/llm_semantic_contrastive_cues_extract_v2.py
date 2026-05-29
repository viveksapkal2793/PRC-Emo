import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import DataLoader
import traceback
from tqdm import tqdm
import json
import re
from transformers import LlamaTokenizer, AutoModel, AutoTokenizer, LlamaForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = "meld"
data_folder = "./data/"
prompt_type = "SemanticContrastiveCues_V1"
debug_mode = False
debug_samples_per_label = 2
debug_label_names = ["fear", "sadness", "anger", "disgust", "neutral", "joy", "surprise"]

print("Loading model ...")
# trained with chat and instruction

model_name = "/scratch/data/bikash_rs/Vivek/PRC-Emo/models/qwen_3_14b"  # standard model, please switch to your local model before running.
tensor_data_type = torch.bfloat16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=tensor_data_type
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=tensor_data_type,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded successfully from local path.")


def get_emotion_labels(dataset_name):
    label_map = {
        "iemocap": ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
        "emorynlp": ["Joyful", "Mad", "Peaceful", "Neutral", "Sad", "Powerful", "Scared"],
        "meld": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
        "dailydialog": ["no_emotion", "happiness", "sadness", "surprise", "anger", "fear", "disgust"],
    }
    dataset_key = dataset_name.lower() if dataset_name else ""
    if dataset_key not in label_map:
        raise ValueError(f"Unknown dataset name '{dataset_name}' - no emotion label list found.")
    return label_map[dataset_key]


def build_debug_selection(data, dataset_name, label_names, samples_per_label):
    label_list = get_emotion_labels(dataset_name)
    label_by_index = {idx: name for idx, name in enumerate(label_list)}
    targets = {name: 0 for name in label_names if name in label_list}
    selected = {}
    for sample in data:
        labels = sample.get("labels", [])
        for idx, lab in enumerate(labels):
            if isinstance(lab, int):
                label_name = label_by_index.get(lab)
            else:
                label_name = str(lab).lower()
            if label_name in targets and targets[label_name] < samples_per_label:
                selected.setdefault(sample["s_id"], set()).add(idx)
                targets[label_name] += 1
        if targets and all(count >= samples_per_label for count in targets.values()):
            break
    return selected, targets


class BatchPreprocessor(object):
    def __init__(self, tokenizer, dataset_name=None, window_ct=2) -> None:
        self.tokenizer = tokenizer
        self.separate_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.dataset_name = dataset_name
        self.window_ct = window_ct

    @staticmethod
    def load_raw_data(path_data):
        raw_data = json.load(open(path_data))
        if isinstance(raw_data, dict):
            new_data_list = []
            for k, v in raw_data.items():
                v["s_id"] = k
                new_data_list.append(v)
            return new_data_list
        if isinstance(raw_data, list):
            return raw_data

    @staticmethod
    def get_speaker_name(s_id, gender, data_name):
        if data_name == "iemocap":
            speaker = {
                "Ses01": {"F": "Mary", "M": "James"},
                "Ses02": {"F": "Patricia", "M": "John"},
                "Ses03": {"F": "Jennifer", "M": "Robert"},
                "Ses04": {"F": "Linda", "M": "Michael"},
                "Ses05": {"F": "Elizabeth", "M": "William"},
            }
            s_id_first_part = s_id[:5]
            return speaker[s_id_first_part][gender].upper()
        if data_name in ["meld", "emorynlp"]:
            gender_idx = gender.index(1)
            return f"SPEAKER_{gender_idx}"
        if data_name == "dailydialog":
            return f"SPEAKER_{gender}"

    def sentence_mixed_by_surrounding(self, sentences, around_window, s_id, genders, data_name):
        new_sentences = []
        for i, cur_sent in enumerate(sentences):
            tmp_s = ""
            for j in range(max(0, i - around_window), min(len(sentences), i + around_window + 1)):
                if i == j:
                    tmp_s += " </s>"
                tmp_s += f" {self.get_speaker_name(s_id, genders[j], data_name=data_name)}: {sentences[j]}"
                if i == j:
                    tmp_s += " </s>"
            new_sentences.append(tmp_s)
        return new_sentences

    def __call__(self, batch):
        raw_sentences = []
        raw_sentences_flatten = []
        labels = []

        lengths = [len(sample["sentences"]) for sample in batch]
        max_len_conversation = max(lengths)
        padding_utterance_masked = torch.BoolTensor(
            [[False] * l_i + [True] * (max_len_conversation - l_i) for l_i in lengths]
        )

        intra_speaker_masekd_all = torch.BoolTensor(len(batch), max_len_conversation, max_len_conversation)
        for i, sample in enumerate(batch):
            sentences_mixed_arround = self.sentence_mixed_by_surrounding(
                sample["sentences"],
                around_window=self.window_ct,
                s_id=sample["s_id"],
                genders=sample["genders"],
                data_name=self.dataset_name,
            )

            padded_conversation = sentences_mixed_arround + ["<pad_sentence>"] * (max_len_conversation - lengths[i])
            raw_sentences.append(padded_conversation)
            raw_sentences_flatten += padded_conversation

            labels += [int(label) for label in sample["labels"]] + [-1] * (max_len_conversation - lengths[i])

            intra_speaker_masekd = torch.BoolTensor(len(padded_conversation), len(padded_conversation)).fill_(False)
            for j in range(len(sample["genders"])):
                for k in range(len(sample["genders"])):
                    gender_j = sample["genders"][j]
                    gender_k = sample["genders"][k]

                    if gender_j == gender_k:
                        intra_speaker_masekd[j][k] = True
                    else:
                        intra_speaker_masekd[j][k] = False

            intra_speaker_masekd_all[i] = intra_speaker_masekd

        if len(labels) != len(raw_sentences_flatten):
            print("len(labels)!= len(raw_sentences_flatten)")

        contextual_sentences_ids = self.tokenizer(
            raw_sentences_flatten, padding="longest", max_length=512, truncation=True, return_tensors="pt"
        )
        sent_indices, word_indices = torch.where(contextual_sentences_ids["input_ids"] == self.separate_token_id)
        gr_sent_indices = [[] for _ in range(len(raw_sentences_flatten))]
        for sent_idx, w_idx in zip(sent_indices, word_indices):
            gr_sent_indices[sent_idx].append(w_idx.item())
        cur_sentence_indexes_masked = torch.BoolTensor(contextual_sentences_ids["input_ids"].shape).fill_(False)
        for i in range(contextual_sentences_ids["input_ids"].shape[0]):
            if raw_sentences_flatten[i] == "<pad_sentence>":
                cur_sentence_indexes_masked[i][gr_sent_indices[i][0]] = True
                continue
            for j in range(contextual_sentences_ids["input_ids"].shape[1]):
                if gr_sent_indices[i][0] <= j <= gr_sent_indices[i][1]:
                    cur_sentence_indexes_masked[i][j] = True

        return (
            contextual_sentences_ids,
            torch.LongTensor(labels),
            padding_utterance_masked,
            intra_speaker_masekd_all,
            cur_sentence_indexes_masked,
            raw_sentences,
        )


class BatchPreprocessorLLMSpeakerSemanticContrastiveCues(BatchPreprocessor):
    def __init__(self, tokenizer, dataset_name=None, window_ct=2, emotion_labels=None) -> None:
        self.tokenizer = tokenizer
        self.separate_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.dataset_name = dataset_name
        self.window_ct = window_ct
        self.emotion_labels = emotion_labels or []

    @staticmethod
    def load_raw_data(path_data):
        raw_data = json.load(open(path_data))
        if isinstance(raw_data, dict):
            new_data_list = []
            for k, v in raw_data.items():
                v["s_id"] = k
                new_data_list.append(v)
            return new_data_list
        if isinstance(raw_data, list):
            return raw_data

    @staticmethod
    def get_speaker_name(s_id, gender, data_name):
        if data_name == "iemocap":
            speaker = {
                "Ses01": {"F": "Mary", "M": "James"},
                "Ses02": {"F": "Patricia", "M": "John"},
                "Ses03": {"F": "Jennifer", "M": "Robert"},
                "Ses04": {"F": "Linda", "M": "Michael"},
                "Ses05": {"F": "Elizabeth", "M": "William"},
            }
            s_id_first_part = s_id[:5]
            return speaker[s_id_first_part][gender].upper()
        if data_name in ["meld", "emorynlp"]:
            gender_idx = gender.index(1)
            return f"SPEAKER_{gender_idx}"
        if data_name == "dailydialog":
            return f"SPEAKER_{gender}"

    def preprocess(self, all_conversations, selected_utterances=None):
        gr_by_len = {}
        for _, sample in enumerate(all_conversations):
            tagged = []
            speaker_sequence = []
            for idx, utt in enumerate(sample["sentences"]):
                if self.dataset_name == "meld":
                    name = sample["speakers"][idx]
                else:
                    name = self.get_speaker_name(sample["s_id"], sample["genders"][idx], self.dataset_name)
                tagged.append(f'{name}: "{utt}"')
                speaker_sequence.append(name)

            for idx, utt in enumerate(tagged):
                if selected_utterances is not None:
                    selected_for_conv = selected_utterances.get(sample["s_id"], set())
                    if idx not in selected_for_conv:
                        continue
                window_size = self.window_ct
                window_start = max(0, idx - window_size + 1)
                context = tagged[window_start : idx + 1]
                convo_str = " ".join(context)
                current_speaker = tagged[idx].split(":")[0]
                _ = get_emotion_labels(self.dataset_name)

                prompt = (
                    "You are an expert in analyzing subtle emotional behavior and conversational emotional dynamics through dialogue context.\n\n"
                    "### Task\n\n"
                    "1. Generate semantic contrastive cues that capture subtle emotional characteristics which help distinguish the speaker's emotional behavior from semantically similar emotional expressions.\n"
                    "2. Focus on discriminative emotional characteristics rather than explicit emotion labels.\n"
                    "3. You MUST take into account the entire past conversation context, including what the current speaker and other speakers have said earlier.\n\n"
                    "### Important Requirements\n\n"
                    "1. Do NOT use explicit emotion labels such as: anger, sadness, fear, joy, surprise, disgust, neutral\n"
                    "Semantic contrastive cues should: focus on emotional style and conversational behavior\n"
                    "1. avoid full sentences\n"
                    "2. be concise semantic phrases\n"
                    "3. be provided as a single comma-separated string\n"
                    "4. remain under 30 total words\n"
                    "5. Focus on characteristics such as: emotional restraint, hesitation, conversational tension, emotional stability, interpersonal warmth, defensiveness, expressive intensity, emotional withdrawal, supportive engagement, emotional escalation\n\n"
                    "### Conversation Context\n"
                    f"{convo_str}\n\n"
                    "### Focus Utterance\n"
                    f"{current_speaker}: \"{utt}\"\n\n"
                    "### Required JSON Format\n"
                    "{\n\"SemanticContrastiveCues\": \"\"\n}\n\n"
                    "### Critical Rules\n\n"
                    "1. Output ONLY the JSON object\n"
                    "2. No explanations/thinking processes\n"
                    "3. No markdown/code formatting\n"
                    "4. Keys must be exactly as shown\n"
                    "5. Do NOT use explicit emotion labels anywhere in the output\n\n"
                    "Example Valid Response:\n"
                    "{\n\"SemanticContrastiveCues\": \"restrained reassurance, low expressive enthusiasm, stable conversational tone, limited emotional escalation\"\n}"
                )

                inputs = self.tokenizer(prompt, return_tensors="pt")
                length = inputs["input_ids"].shape[-1]
                entry = {
                    "input_ids": inputs["input_ids"],
                    "conv_id": sample["s_id"],
                    "utter_idx": idx,
                    "prompting_input": prompt,
                    "type_data": sample["type_data"],
                    "speaker_name": current_speaker,
                    "all_speakers": speaker_sequence,
                }
                gr_by_len.setdefault(length, []).append(entry)
        return gr_by_len


raw_data = []
type_data_list = ["train"] if debug_mode else ["valid", "test", "train"]
debug_suffix = "_debug" if debug_mode else ""
for type_data in type_data_list:
    data_name_pattern = f"{dataset_name}.{type_data}"
    path_processed_data = (
        f"{data_folder}/{data_name_pattern}_{prompt_type}_{model_name.split('/')[-1]}{debug_suffix}.json"
    )
    org_raw_data = BatchPreprocessorLLMSpeakerSemanticContrastiveCues.load_raw_data(
        f"{data_folder}/{data_name_pattern}.json"
    )
    if os.path.exists(path_processed_data):
        processed_data = json.load(open(path_processed_data, "rt"))
        print(
            f"- Successfully processed {len(processed_data)}/{len(org_raw_data)} conversations in data-type = {type_data}"
        )
        json.dump(
            processed_data,
            open(path_processed_data + "_backup.json", "wt"),
            indent=2,
        )
        org_raw_data = [e for e in org_raw_data if e["s_id"] not in processed_data]
    print(f"- Continue processing {len(org_raw_data)} conversations in data-type = {type_data}")
    for e in org_raw_data:
        e["type_data"] = type_data
    raw_data = raw_data + org_raw_data

debug_selection = None
if debug_mode:
    debug_selection, debug_counts = build_debug_selection(
        raw_data, dataset_name, debug_label_names, debug_samples_per_label
    )
    print(f"- Debug selection counts: {debug_counts}")


data_preprocessor = BatchPreprocessorLLMSpeakerSemanticContrastiveCues(
    tokenizer,
    dataset_name=dataset_name,
    window_ct=5,
    emotion_labels=["happy", "sad", "neutral", "angry", "excited", "frustrated"],
)

gr_by_len = data_preprocessor.preprocess(raw_data, selected_utterances=debug_selection)
all_data = {}
print_one_time = True

for len_promting, speaker_promts in tqdm(gr_by_len.items(), desc="Prompt Length Groups"):
    for batch_size in [32, 16, 8, 4, 2, 1]:
        try:
            all_promtings_texts = [e["prompting_input"] for e in speaker_promts]

            data_loader = DataLoader(all_promtings_texts, batch_size=batch_size, shuffle=False)

            output_sc_cues = []
            with torch.no_grad():
                for i, speaker_promts_in_batch in enumerate(data_loader):
                    inputs = tokenizer(
                        speaker_promts_in_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024,
                    ).to(device)

                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=800,
                        temperature=0.3,
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.2,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1,
                    )

                    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    batch_outputs = []
                    for j, gen_text in enumerate(generated_texts):
                        prompt_text = speaker_promts_in_batch[j]
                        generated_part = gen_text.replace(prompt_text, "", 1).strip()

                        semantic_contrastive_cues = ""
                        try:
                            json_match = re.search(r"\{\s*\"SemanticContrastiveCues\".*?\}", generated_part, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                cues_data = json.loads(json_str)
                                semantic_contrastive_cues = cues_data.get("SemanticContrastiveCues", "")
                            else:
                                raise ValueError("No JSON found")

                        except Exception as e:
                            cues_pos = generated_part.rfind("SemanticContrastiveCues:")
                            if cues_pos != -1:
                                semantic_contrastive_cues = generated_part[
                                    cues_pos + len("SemanticContrastiveCues:") :
                                ].strip()
                            else:
                                semantic_contrastive_cues = generated_part

                        batch_outputs.append({
                            "semantic_contrastive_cues": semantic_contrastive_cues
                        })
                        output_sc_cues.append({
                            "semantic_contrastive_cues": semantic_contrastive_cues
                        })

                    if print_one_time:
                        if len(speaker_promts_in_batch) > 0:
                            sample_idx = min(3, len(speaker_promts_in_batch) - 1)
                            print("- Example Prompt:", speaker_promts_in_batch[sample_idx])
                            print("- Generated Text:", generated_texts[sample_idx])
                            print("- Structured Output:", batch_outputs)
                            for i in range(min(5, len(output_sc_cues))):
                                print(f"{i+1}.", output_sc_cues[-len(output_sc_cues) + i])
                            print_one_time = False

                for i, out in enumerate(output_sc_cues):
                    speaker_promts[i]["semantic_contrastive_cues"] = out["semantic_contrastive_cues"]

            break

        except Exception as e:
            traceback.print_exc()
            print(e)
            if batch_size == 1:
                print(["Errr "] * 10)
                print("CUDA out of Memory (batch = 1)!!!!")

for type_data in type_data_list:
    data_name_pattern = f"{dataset_name}.{type_data}"
    path_processed_data = (
        f"{data_folder}/{data_name_pattern}_{prompt_type}_{model_name.split('/')[-1]}{debug_suffix}.json"
    )

    processed_data = {}

    if os.path.exists(path_processed_data):
        processed_data = json.load(open(path_processed_data, "rt"))
        print(f"- Loaded processed [old] {len(processed_data)} conversations for data-type = {type_data}")

    new_data = {}

    for len_promting, speaker_promts in gr_by_len.items():
        for entry in speaker_promts:
            if type_data != entry["type_data"]:
                continue

            conv_id = entry["conv_id"]
            utter_idx = entry["utter_idx"]

            if conv_id not in new_data:
                new_data[conv_id] = {
                    "utterances": [],
                    "predictions": {},
                }

            if not new_data[conv_id]["utterances"]:
                raw_conv = next((c for c in raw_data if c["s_id"] == conv_id), None)
                if raw_conv:
                    speaker_sequence = entry["all_speakers"]
                    print(entry["all_speakers"])
                    new_data[conv_id]["utterances"] = [
                        f"{speaker}: {text}" for speaker, text in zip(speaker_sequence, raw_conv["sentences"])
                    ]

            new_data[conv_id]["predictions"][str(utter_idx)] = {
                "semantic_contrastive_cues": entry.get("semantic_contrastive_cues", "unknown"),
                "prompt": entry["prompting_input"],
            }

    print(f"- Successfully processed [new] {len(all_data)} conversations for data-type = {type_data}")

    final_data = {}
    for conv_id in set(processed_data.keys()) | set(new_data.keys()):
        if conv_id in new_data:
            final_data[conv_id] = new_data[conv_id]
        else:
            final_data[conv_id] = processed_data[conv_id]

        final_data[conv_id].setdefault("utterances", [])
        final_data[conv_id].setdefault("predictions", {})

    formatted_output = {}
    for conv_id, data in final_data.items():
        ordered_predictions = []
        for idx in range(len(data["utterances"])):
            pred = data["predictions"].get(
                str(idx),
                {
                    "semantic_contrastive_cues": "No prediction",
                },
            )
            ordered_predictions.append(pred)

        formatted_output[conv_id] = {
            "utterances": data["utterances"],
            "emotion_predictions": ordered_predictions,
        }

    with open(path_processed_data, "w") as f:
        json.dump(formatted_output, f, indent=2)

    print(f"- Saved {len(formatted_output)} conversations for {type_data}")
    if debug_mode:
        print("- Debug mode complete. Exiting before full-set processing.")
        sys.exit(0)
