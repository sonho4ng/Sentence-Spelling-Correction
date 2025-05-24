import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path_detection = "checkpoint/detection"
checkpoint_path_correction = "vinai/bartpho-syllable"

@st.cache_resource
def load_models():
    tokenizer_detect = AutoTokenizer.from_pretrained(checkpoint_path_detection, use_fast=True)
    model_detect = AutoModelForTokenClassification.from_pretrained(checkpoint_path_detection).to(device)

    tokenizer_correct = AutoTokenizer.from_pretrained(checkpoint_path_correction)
    model_correct = AutoModelForMaskedLM.from_pretrained(checkpoint_path_correction).to(device)
    return tokenizer_detect, model_detect, tokenizer_correct, model_correct


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


def correct_error_top_k(model, tokenizer, corrupted_tokens, error_flags,
                                top_k=5, max_len=128, confidence_threshold=0.8,
                                word_length_threshold=5):
    model.eval()
    tokenizer.model_max_length = max_len
    
    masked_tokens = corrupted_tokens[:]
    mask_token = tokenizer.mask_token
    for i in range(len(error_flags)):
        if error_flags[i] != 0:
            masked_tokens[i] = mask_token
    remove_index = []
    for i in range(1, len(masked_tokens)):
        if masked_tokens[i] == mask_token and masked_tokens[i-1] == mask_token:
            remove_index.append(i)
    for i, idx in enumerate(remove_index):
        masked_tokens.pop(idx-i)
    masked_text = " ".join(masked_tokens)

    inputs = tokenizer(
        masked_text,
        return_tensors="pt",
        max_length=max_len,
        padding="max_length",
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            num_beams=top_k,
            num_return_sequences=top_k,
            early_stopping=True
        )
    preds = []
    for pred in outputs:
        tmp = tokenizer.decode(pred, skip_special_tokens=True).strip()
        if len(tmp) > 0:
            preds.append(tmp)
            
    return preds


tokenizer_detect, model_detect, tokenizer_correct, model_correct = load_models()

# ================== Streamlit App ==================
st.title("Vietnamese Sentence Spelling Correction")

user_input = st.text_area("Nhập câu có lỗi chính tả:",
                          "e.g., Đặc biệt việc này cũng giúp tạo vị tFế cạnh tranh cho các kế TOÁN viên vrà tránh nguy cơ bị đào thải")

if st.button("Sửa Lỗi"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập câu cần sửa!")
    else:
        st.write("**Câu gốc:** ", user_input)
        st.success("Kết quả:")
        tokens = user_input.split()
        
        error_flags = detect_error(model_detect, tokenizer_detect, tokens)
        tokens_highlighted = tokens[:]
        for i in range(len(tokens)):
            if error_flags[i] != 0:
                tokens[i] = "*" + tokens[i] + "*"
        st.write("**Các lỗi phát hiện trong câu:** ", " ".join(tokens_highlighted))
        
        corrected_sentences = correct_error_top_k(
            model_correct, tokenizer_correct, tokens, error_flags, top_k=5
        )
        st.write("**Câu sau khi sửa:**")
        for i, pred in enumerate(corrected_sentences):
            st.write(f"{i+1}. ", pred)