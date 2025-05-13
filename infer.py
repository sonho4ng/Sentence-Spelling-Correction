import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM
from Levenshtein import distance as levenshtein_distance
import math
from collections import Counter
import re
import pandas as pd

# ================== Config ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoint/detection"
model_name_mlm = "xlm-roberta-base"


@st.cache_resource
def load_models():
    tokenizer_detect = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    model_detect = AutoModelForTokenClassification.from_pretrained(checkpoint_path).to(device)

    tokenizer_correct = AutoTokenizer.from_pretrained(model_name_mlm)
    model_correct = AutoModelForMaskedLM.from_pretrained(model_name_mlm).to(device)
    return tokenizer_detect, model_detect, tokenizer_correct, model_correct


@st.cache_data
def build_trigram_counter():
    df = pd.read_csv("data/error_dataset.csv")
    corpus = df['ground_truth']
    trigram_counter = Counter()
    for sentence in corpus:
        tokens = re.findall(r'\w+', str(sentence).lower())
        tokens = ['<s>'] + tokens + ['</s>']
        for i in range(len(tokens) - 2):
            trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
            trigram_counter[trigram] += 1
    return trigram_counter


tokenizer_detect, model_detect, tokenizer_correct, model_correct = load_models()
trigram_counter = build_trigram_counter()


# ================== Functions ==================
def detect_error(model, tokenizer, tokens, max_len=128):
    model.eval()
    tokenizer.model_max_length = max_len
    encoding = tokenizer(tokens, is_split_into_words=True, padding="max_length", truncation=True,
                         max_length=max_len, return_tensors='pt')
    word_ids = encoding.word_ids()
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**encoding).logits
        preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    result = [0] * len(tokens)
    prev_word_idx = None
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:
            result[word_idx] = preds[token_idx]
            prev_word_idx = word_idx
    return result


def compute_trigram_score(left_context, candidate_token):
    w1 = left_context[-2] if len(left_context) >= 2 else '<s>'
    w2 = left_context[-1] if len(left_context) >= 1 else '<s>'
    w3 = candidate_token
    trigram = (w1, w2, w3)
    bigram_count = sum(count for (a, b, _), count in trigram_counter.items() if a == w1 and b == w2) + 1
    trigram_count = trigram_counter.get(trigram, 0) + 1
    return math.log(trigram_count / bigram_count)


def normalize_trigram(score, min_score=-10, max_score=0):
    return (score - min_score) / (max_score - min_score) if max_score != min_score else 0.0


def correct_error_encoder_top_k(model, tokenizer, corrupted_tokens, error_flags,
                                top_k=10, max_len=64, confidence_threshold=0.8,
                                word_length_threshold=5, trigram_weight=0.6, levenshtein_weight=0.4):
    model.eval()
    tokenizer.model_max_length = max_len
    masked_tokens = [tokenizer.mask_token if flag else tok for tok, flag in zip(corrupted_tokens, error_flags)]
    masked_text = " ".join(masked_tokens)

    inputs = tokenizer(masked_text, return_tensors="pt", max_length=max_len, padding="max_length", truncation=True).to(
        device)
    word_ids = inputs.word_ids()

    with torch.no_grad():
        logits = model(**inputs).logits

    replaced_tokens = corrupted_tokens.copy()
    printed = set()

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in printed or error_flags[word_idx] != 1:
            continue
        printed.add(word_idx)

        token_logits = logits[0, token_idx]
        probs = torch.softmax(token_logits, dim=-1)
        topk_ids = torch.topk(probs, top_k).indices.tolist()
        topk_tokens = [tokenizer.convert_ids_to_tokens(tid).lstrip("▁") for tid in topk_ids]
        topk_confs = [probs[tid].item() for tid in topk_ids]

        model_confidence = topk_confs[0]
        left_context = replaced_tokens[:word_idx]

        if model_confidence > confidence_threshold:
            best_pred = topk_tokens[0]
        elif len(corrupted_tokens[word_idx]) < word_length_threshold:
            best_pred = max(topk_tokens, key=lambda x: compute_trigram_score(left_context, x))
        else:
            best_score = -math.inf
            for cand in topk_tokens:
                trigram_s = compute_trigram_score(left_context, cand)
                trigram_s_norm = normalize_trigram(trigram_s)
                lev_sim = 1 - levenshtein_distance(corrupted_tokens[word_idx], cand) / max(
                    len(corrupted_tokens[word_idx]), len(cand))
                combined_s = trigram_weight * trigram_s_norm + levenshtein_weight * lev_sim
                if combined_s > best_score:
                    best_score = combined_s
                    best_pred = cand
        replaced_tokens[word_idx] = best_pred

    return " ".join(replaced_tokens)


# ================== Streamlit App ==================
st.title("Vietnamese Sentence Correction (Transformer + Trigram + Levenshtein)")

user_input = st.text_area("Nhập câu có lỗi chính tả:",
                          "Đặc biệt việc này cũng giúp tạo vị tFế cạnh tranh cho các kế TOÁN viên vrà tránh nguy cơ bị đào thải")

if st.button("Sửa Lỗi"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập câu cần sửa!")
    else:
        tokens = user_input.split()
        error_flags = detect_error(model_detect, tokenizer_detect, tokens)
        corrected_sentence = correct_error_encoder_top_k(
            model_correct, tokenizer_correct, tokens, error_flags
        )
        st.success("Kết quả:")
        st.write("**Câu gốc:** ", user_input)
        st.write("**Câu sau khi sửa:** ", corrected_sentence)
