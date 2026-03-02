import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from trl import setup_chat_format
import json

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "/scratch/data/bikash_rs/vivek/PRC-Emo/finetuned_llm/meld_iitjhome_ep4_step-1_lrs-linear3e-4_0shot_r32_w5_ImplicitEmotion_V3_seed42_L2048_llmdescqwen_3_14b_ED100000_final_full_finetune"
BASE_MODEL_PATH = "/iitjhome/bikash_rs/vivek/models/qwen_3_8b"
DATA_FILE = "/scratch/data/bikash_rs/vivek/PRC-Emo/data/meld.test.0shot_w5_ImplicitEmotion_V3_qwen_3_14b.jsonl"

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading model...")

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ✅ Load tokenizer from BASE model first
print(f"Loading tokenizer from: {BASE_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

# ✅ Load base model
print(f"Loading base model from: {BASE_MODEL_PATH}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    trust_remote_code=True
)

# ✅ CRITICAL: Apply chat format (this modifies vocab size to match training)
print("Applying chat format (setup_chat_format)...")
base_model, tokenizer = setup_chat_format(base_model, tokenizer)

print(f"Vocab size after setup_chat_format: {len(tokenizer)}")
print(f"Model embed_tokens size: {base_model.model.embed_tokens.weight.shape}")

# ✅ Now load LoRA adapter (vocab sizes should match)
print(f"\nLoading LoRA adapter from: {MODEL_PATH}")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

print("✅ Model loaded successfully!\n")

# ============================================================
# LOAD TEST SAMPLE
# ============================================================
print(f"Loading test sample from: {DATA_FILE}")
with open(DATA_FILE, 'r') as f:
    first_line = f.readline()
    sample = json.loads(first_line)

# Extract the prompt (system + user messages)
messages = sample['messages']
system_msg = messages[0]['content']
user_msg = messages[1]['content']
true_label = messages[2]['content']

print("\n" + "="*80)
print("📥 TEST SAMPLE")
print("="*80)
print(f"\n🎭 System Prompt (first 500 chars):")
print(system_msg[:500] + "...")
print(f"\n❓ User Question:")
print(user_msg)
print(f"\n✅ True Label: {true_label}")
print("="*80)

# ============================================================
# PREPARE INPUT
# ============================================================
# Construct the conversation WITHOUT the assistant's answer
conversation = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": user_msg}
]

# Apply chat template
prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,  # This adds <|im_start|>assistant\n
    enable_thinking=False
)

print("\n" + "="*80)
print("🔧 FORMATTED PROMPT")
print("="*80)
print(f"Length: {len(prompt)} chars")
print(f"\nFirst 500 chars:")
print(prompt[:500])
print(f"\nLast 200 chars:")
print(prompt[-200:])
print("="*80)

# ============================================================
# TOKENIZE INPUT
# ============================================================
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\n" + "="*80)
print("🔢 TOKENIZATION INFO")
print("="*80)
print(f"Input shape: {inputs['input_ids'].shape}")
print(f"First 20 token IDs: {inputs['input_ids'][0][:20].tolist()}")
print(f"Last 20 token IDs: {inputs['input_ids'][0][-20:].tolist()}")

# Decode to verify
decoded_input = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
print(f"\nDecoded input (last 300 chars):")
print(decoded_input[-300:])
print("="*80)

# ============================================================
# GENERATE OUTPUT
# ============================================================
print("\n" + "="*80)
print("🚀 GENERATING OUTPUT...")
print("="*80)

generation_config = {
    'max_new_tokens': 5,
    'do_sample': False,
    'temperature': 0.0,
    'top_p': 1.0,
    'eos_token_id': tokenizer.convert_tokens_to_ids("<|im_end|>"),
    'pad_token_id': tokenizer.pad_token_id,
}

print("Generation parameters:")
for key, value in generation_config.items():
    print(f"  {key}: {value}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        **generation_config
    )

# ============================================================
# DECODE OUTPUT
# ============================================================
# Decode full output
full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

# Decode only the generated tokens (excluding input)
input_length = inputs['input_ids'].shape[1]
generated_tokens = outputs[0][input_length:]
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

print("\n" + "="*80)
print("📤 MODEL OUTPUT")
print("="*80)
print(f"\n🔹 Full output length: {len(full_output)} chars")
print(f"🔹 Generated length: {len(generated_text)} chars")

print(f"\n📝 Generated text (raw):")
print(generated_text)

# ✅ IMPROVED: Extract emotion from verbose output
def extract_emotion(text):

    emotions = ['neutral','surprise','fear','sadness','joy','disgust','anger']

    text = text.lower()

    # scan from end (most reliable)
    for emotion in emotions:
        if emotion in text:
            return emotion

    return "unknown"

cleaned = extract_emotion(generated_text)

print(f"\n📝 Extracted emotion:")
print(f"'{cleaned}'")

print("\n" + "="*80)
print("✅ COMPARISON")
print("="*80)
print(f"True label:  '{true_label}'")
print(f"Predicted:   '{cleaned}'")
print(f"Match: {cleaned.lower() == true_label.lower()}")
print("="*80)

# ============================================================
# TEST MULTIPLE SAMPLES
# ============================================================
print("\n\n" + "="*80)
print("🔄 TESTING 5 MORE SAMPLES")
print("="*80)

with open(DATA_FILE, 'r') as f:
    samples = [json.loads(line) for line in f.readlines()[:6]]

results = []
for i, sample in enumerate(samples[1:6], start=2):
    messages = sample['messages']
    system_msg = messages[0]['content']
    user_msg = messages[1]['content']
    true_label = messages[2]['content']
    
    # Prepare input
    conversation = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)
    
    # Decode
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    print("\nRAW TOKENS:")
    print(outputs[0][-10:])
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    
    # Clean
    # cleaned = generated_text.split("<|im_end|>")[0].strip()
    # cleaned = cleaned.replace("<|im_start|>", "").strip()
    cleaned = extract_emotion(generated_text)
    
    match = cleaned.lower() == true_label.lower()
    results.append({
        "sample": i,
        "true": true_label,
        "predicted": cleaned,
        "match": match
    })
    
    print(f"\nSample {i}:")
    print(f"  True: '{true_label}'")
    print(f"  Pred: '{cleaned}'")
    print(f"  Match: {match}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)
correct = sum(1 for r in results if r['match'])
total = len(results)
print(f"Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
print("="*80)