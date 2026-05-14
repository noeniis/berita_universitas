from __future__ import annotations

import html as html_lib
import os
import re
import tempfile
import time
import unicodedata
from typing import Dict, List, Tuple

import gdown
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from config import DRIVE_IDS, JW_CFG, MODEL_CFG

# ==============================================================
# KONFIGURASI HALAMAN
# ==============================================================

st.set_page_config(
    page_title="Penyunting Kata Berita",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================
# STEMMER
# ==============================================================

stemmer = StemmerFactory().create_stemmer()

# ==============================================================
# STYLE VISUAL
# ==============================================================

st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: Arial, sans-serif; }

.text-preview-box {
    background: #fafafa;
    border: 1.5px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 24px;
    line-height: 2.05em;
    font-size: 1.03rem;
    color: #111827;
    word-spacing: 1px;
}

.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================
# UTILITAS TEKS
# ==============================================================

def normalize_unicode(text: str) -> str:
    if pd.isna(text):
        return ""
    return unicodedata.normalize("NFKC", str(text))

def clean_whitespace(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\u200b", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()

def normalize_token(token: str) -> str:
    if pd.isna(token):
        return ""
    return clean_whitespace(normalize_unicode(token)).lower().strip()

def escape_html(text: str) -> str:
    return html_lib.escape(text)

def extract_sentence_spans(text: str) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    if not text or not text.strip():
        return spans
    pattern = re.compile(r".+?(?:[.!?](?:\s+|$)|$)", flags=re.DOTALL)
    for match in pattern.finditer(text):
        sent = match.group().strip()
        if sent:
            spans.append((sent, match.start(), match.end()))
    return spans

def tokenize_with_spans(sentence: str) -> List[Tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\b\w+\b", sentence, flags=re.UNICODE)]

def is_number_token(token: str) -> bool:
    return bool(re.fullmatch(r"[\d.,:/%-]+", token))

def is_all_caps_token(token: str) -> bool:
    return len(token) >= 2 and token.isupper()

def is_title_case_name(token: str, position: int, kbbi_set: set, inggris_set: set, serapan_set: set, whitelist_set: set) -> bool:
    t = normalize_token(token)
    if not t or len(t) <= 2:
        return False
    if t in kbbi_set or t in inggris_set or t in serapan_set or t in whitelist_set:
        return False
    return token[0].isupper() and not token.isupper()

# ==============================================================
# LOAD RESOURCES
# ==============================================================

_TMP = tempfile.gettempdir()
MODEL_LOCAL = os.path.join(_TMP, "model_indobert_best")
LEXICON_LOCAL = os.path.join(_TMP, "leksikon")
os.makedirs(MODEL_LOCAL, exist_ok=True)
os.makedirs(LEXICON_LOCAL, exist_ok=True)

LEXICON_COL_MAP = {
    "kbbi": "kata",
    "kata_inggris": "headword",
    "akronim": "akronim",
    "daftar_lembaga": "Nama Lembaga",
    "daftar_nama_orang": "Nama",
    "istilah_islam": "Kata",
}

@st.cache_resource(show_spinner=False)
def load_model():
    config_path = os.path.join(MODEL_LOCAL, "config.json")
    if not os.path.exists(config_path):
        gdown.download_folder(id=DRIVE_IDS["model_indobert"], output=MODEL_LOCAL, quiet=False, use_cookies=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(MODEL_LOCAL)
    model = BertForSequenceClassification.from_pretrained(MODEL_LOCAL, num_labels=MODEL_CFG["num_labels"])
    model.to(device)
    model.eval()
    return tokenizer, model, device

@st.cache_resource(show_spinner=False)
def load_lexicons():
    def read_csv_safe(path: str) -> pd.DataFrame:
        for enc in ["utf-8", "utf-8-sig", "latin1"]:
            try:
                df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
                df.columns = [c.strip() for c in df.columns]
                return df
            except: continue
        return pd.DataFrame()

    def to_set(df: pd.DataFrame, col: str) -> set:
        if df is None or df.empty: return set()
        if col not in df.columns: col = df.columns[0] if len(df.columns) > 0 else None
        if col is None: return set()
        vals = df[col].dropna().astype(str)
        return set(normalize_token(v) for v in vals if len(normalize_token(v)) >= 2)

    lex_dfs: Dict[str, pd.DataFrame] = {}
    keys = ["kbbi", "kata_inggris", "kata_serapan", "akronim", "daftar_lembaga", "daftar_nama_orang", "istilah_islam"]
    if "sample_correct_2025" in DRIVE_IDS: keys.append("sample_correct_2025")

    for key in keys:
        local_path = os.path.join(LEXICON_LOCAL, f"{key}.csv")
        if not os.path.exists(local_path):
            try: gdown.download(id=DRIVE_IDS[key], output=local_path, quiet=True)
            except: pass
        lex_dfs[key] = read_csv_safe(local_path)

    kbbi_set = to_set(lex_dfs.get("kbbi", pd.DataFrame()), "kata")
    df_ing = lex_dfs.get("kata_inggris", pd.DataFrame())
    inggris_set = (to_set(df_ing, "headword") - kbbi_set) if not df_ing.empty else set()

    whitelist_set = set()
    for key in ["akronim", "daftar_lembaga", "daftar_nama_orang", "istilah_islam"]:
        whitelist_set.update(to_set(lex_dfs.get(key, pd.DataFrame()), LEXICON_COL_MAP[key]))

    df_domain = lex_dfs.get("sample_correct_2025", pd.DataFrame())
    if not df_domain.empty:
        vals = df_domain[df_domain.columns[0]].dropna().astype(str).str.lower()
        for row in vals:
            whitelist_set.update(re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", row))

    serapan_map: Dict[str, str] = {}
    serapan_set = set()
    df_s = lex_dfs.get("kata_serapan", pd.DataFrame())
    if not df_s.empty:
        col_asal = next((c for c in df_s.columns if "asal" in c.lower() or "asing" in c.lower()), df_s.columns[0])
        col_serapan = next((c for c in df_s.columns if "serapan" in c.lower() or "hasil" in c.lower()), df_s.columns[-1])
        for _, row in df_s.iterrows():
            asal = normalize_token(str(row[col_asal]))
            serapan = normalize_token(str(row[col_serapan]))
            if asal and serapan:
                serapan_map[asal] = serapan
                serapan_set.add(asal)

    return kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, sorted(kbbi_set)

# ==============================================================
# ALGORITMA JARO-WINKLER
# ==============================================================

def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    if s1 == s2: return 1.0
    if not s1 or not s2: return 0.0
    len1, len2 = len(s1), len(s2)
    match_dist = max(0, max(len1, len2) // 2 - 1)
    s1m, s2m = [False] * len1, [False] * len2
    matches = 0
    for i in range(len1):
        for j in range(max(0, i - match_dist), min(i + match_dist + 1, len2)):
            if not s2m[j] and s1[i] == s2[j]:
                s1m[i] = s2m[j] = True
                matches += 1
                break
    if matches == 0: return 0.0
    transpositions, k = 0, 0
    for i in range(len1):
        if s1m[i]:
            while not s2m[k]: k += 1
            if s1[i] != s2[k]: transpositions += 1
            k += 1
    jaro = (matches/len1 + matches/len2 + (matches - transpositions/2)/matches) / 3
    prefix = 0
    for i in range(min(4, len(s1), len(s2))):
        if s1[i] == s2[i]: prefix += 1
        else: break
    return jaro + prefix * p * (1 - jaro)

# ==============================================================
# PREDIKSI
# ==============================================================

def classify_token(token: str, kbbi_set, inggris_set, whitelist_set, serapan_set) -> str:
    t = normalize_token(token)
    if not t: return "KOSONG"
    if t in whitelist_set: return "WHITELIST_KHUSUS"
    if t in serapan_set: return "KATA_SERAPAN"
    if t in kbbi_set or stemmer.stem(t) in kbbi_set: return "KBBI_VALID"
    if t in inggris_set: return "KATA_INGGRIS"
    return "TIDAK_DIKENAL"

def predict_jw(token: str, kbbi_set, inggris_set, whitelist_set, serapan_set, kbbi_list, threshold: float, top_k: int = 5) -> dict:
    t = normalize_token(token)
    status = classify_token(t, kbbi_set, inggris_set, whitelist_set, serapan_set)
    if status in ("WHITELIST_KHUSUS", "KBBI_VALID", "KATA_SERAPAN", "KOSONG", "KATA_INGGRIS"):
        return {"pred": 0, "max_sim": 1.0, "best_match": t, "top_k_recs": [], "status": status}
    sims = sorted([(w, jaro_winkler_similarity(t, w)) for w in kbbi_list], key=lambda x: x[1], reverse=True)
    best_word, best_sim = sims[0]
    return {"pred": 0 if best_sim >= threshold else 1, "max_sim": round(best_sim, 4), "best_match": best_word, "top_k_recs": [w for w, _ in sims[:top_k]], "status": status}

def predict_bert(kalimat: str, token: str, tokenizer, model, device) -> dict:
    enc = tokenizer(clean_whitespace(kalimat), normalize_token(token), max_length=MODEL_CFG["max_length"], truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), token_type_ids=enc["token_type_ids"].to(device))
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    return {"pred": int(np.argmax(probs)), "prob_correct": round(float(probs[0]), 4), "prob_error": round(float(probs[1]), 4)}

# ==============================================================
# PIPELINE ANALISIS
# ==============================================================

def analyze_text(text: str, model_choice: str, tokenizer, bert_model, device, kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, kbbi_list, skip_proper_noun: bool = True) -> List[dict]:
    sentences = extract_sentence_spans(text)
    results: List[dict] = []

    for sent, sent_start, _ in sentences:
        tokens = tokenize_with_spans(sent)
        for pos, (tok, start, end) in enumerate(tokens):
            t = normalize_token(tok)
            if not t or len(t) < 2 or is_number_token(t) or is_all_caps_token(tok) or t in whitelist_set: continue
            if skip_proper_noun and is_title_case_name(tok, pos, kbbi_set, inggris_set, serapan_set, whitelist_set): continue

            jw_res = predict_jw(t, kbbi_set, inggris_set, whitelist_set, serapan_set, kbbi_list, threshold=JW_CFG["threshold"], top_k=JW_CFG["top_k"])
            status = jw_res["status"]

            if status in ["KATA_INGGRIS", "KATA_SERAPAN", "WHITELIST_KHUSUS", "KBBI_VALID"]:
                bert_res = {"pred": 0, "prob_correct": 1.0, "prob_error": 0.0}
            else:
                bert_res = predict_bert(sent, t, tokenizer, bert_model, device)

            jw_pred, bert_pred = jw_res["pred"], bert_res["pred"]
            final_pred = 1 if (jw_pred == 1 or bert_pred == 1) else 0

            recs = jw_res["top_k_recs"]
            catatan = ""

            if status == "KATA_INGGRIS":
                flag, tipe = "KATA_INGGRIS", "Kata Bahasa Inggris"
                padanan = serapan_map.get(t)
                catatan = f"Padanan KBBI: '{padanan}'" if padanan else "Gunakan huruf miring jika dipertahankan"
            elif status == "KATA_SERAPAN":
                flag, tipe = "KATA_SERAPAN", "Kata Serapan"
                hasil_serapan = serapan_map.get(t)
                if hasil_serapan:
                    catatan = f"Saran serapan baku: '{hasil_serapan}'"
                    recs = [hasil_serapan]
            elif final_pred == 1:
                flag, tipe = "TYPO", "Typo / Kata tidak baku"
            else: continue

            results.append({
                "token": tok, "token_norm": t, "kalimat": sent, "start": sent_start + start, "end": sent_start + end,
                "flag": flag, "tipe_error": tipe, "prob_error": bert_res["prob_error"], "jw_sim": jw_res["max_sim"],
                "best_match": jw_res["best_match"], "rekomendasi": recs, "catatan": catatan
            })
    return results

# ==============================================================
# RENDER UI
# ==============================================================

FLAG_STYLES = {
    "TYPO": {"label": "Typo / Kata tidak baku", "bg": "#ffd6d6", "border": "#d64545", "text": "#7a1111"},
    "KATA_INGGRIS": {"label": "Kata bahasa Inggris", "bg": "#fff2b3", "border": "#d4a017", "text": "#6b4f00"},
    "KATA_SERAPAN": {"label": "Kata asal serapan", "bg": "#dbeafe", "border": "#3b82f6", "text": "#1e3a8a"},
}

def build_tooltip(row: dict) -> str:
    flag, token = row["flag"], row["token"]
    if flag == "TYPO":
        lines = [f"⚠️ “{token}” terdeteksi typo/tidak baku"]
        if row.get("rekomendasi"): lines.append("Saran: " + ", ".join(row["rekomendasi"][:3]))
    elif flag == "KATA_SERAPAN":
        lines = [f"📌 “{token}” adalah kata asal serapan"]
        if row.get("catatan"): lines.append(row["catatan"])
    else:
        lines = [f"🌐 “{token}” kata bahasa Inggris", row.get("catatan", "")]
    return "\n".join(lines)

def render_highlighted_text(text: str, rows: List[dict]) -> str:
    row_map = {(r["start"], r["end"]): r for r in rows}
    parts, cursor = [], 0
    for m in re.finditer(r"\b\w+\b", text, flags=re.UNICODE):
        parts.append(escape_html(text[cursor:m.start()]))
        row = row_map.get((m.start(), m.end()))
        if row:
            s = FLAG_STYLES.get(row["flag"], FLAG_STYLES["TYPO"])
            parts.append(f'<span title="{escape_html(build_tooltip(row))}" style="background:{s["bg"]}; color:{s["text"]}; border:1px solid {s["border"]}; border-radius:6px; padding:1px 5px; font-weight:600; cursor:help;">{escape_html(m.group())}</span>')
        else: parts.append(escape_html(m.group()))
        cursor = m.end()
    parts.append(escape_html(text[cursor:]))
    return f'<div style="line-height:1.95; font-size:1.02rem; white-space:pre-wrap; word-break:break-word;">{"".join(parts)}</div>'

# ==============================================================
# APP INTERFACE
# ==============================================================

with st.sidebar:
    st.markdown("### 📝 Penyunting Kata Berita")
    st.caption("Alat normalisasi teks berita UIN Jakarta")
    st.markdown("---")
    show_inggris = st.toggle("Tandai kata Inggris", value=True)
    show_serapan = st.toggle("Tandai kata serapan", value=True)
    skip_proper_noun = st.toggle("Abaikan nama orang/tempat", value=True)
    st.markdown("---")
    st.caption("Noeni Indah Sulistiyani · Teknik Informatika · UIN Jakarta")

st.title("📝 Penyunting Kata Berita")
st.markdown("Pengecekan bahasa otomatis menggunakan metode **Hybrid Jaro-Winkler & IndoBERT**.")

with st.spinner("Memuat model dan leksikon..."):
    tokenizer, bert_model, device = load_model()
    kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, kbbi_list = load_lexicons()

input_text = st.text_area("Masukkan teks berita:", height=180, placeholder="Tulis atau tempel teks di sini...")

if st.button("🔍 Analisis", type="primary", use_container_width=True) and input_text.strip():
    t0 = time.time()
    results = analyze_text(input_text, "Hybrid-OR", tokenizer, bert_model, device, kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, kbbi_list, skip_proper_noun)
    elapsed = round(time.time() - t0, 2)
    
    display_res = [r for r in results if (r["flag"] != "KATA_INGGRIS" or show_inggris) and (r["flag"] != "KATA_SERAPAN" or show_serapan)]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Typo / Tidak Baku", sum(1 for r in display_res if r["flag"] == "TYPO"))
    c2.metric("Kata Asing/Serapan", sum(1 for r in display_res if r["flag"] in ["KATA_INGGRIS", "KATA_SERAPAN"]))
    c3.metric("Total Ditandai", len(display_res))
    c4.metric("Waktu", f"{elapsed}s")

    st.markdown("### 📄 Hasil Pengecekan")
    st.markdown(render_highlighted_text(input_text, display_res), unsafe_allow_html=True)

    if display_res:
        st.markdown("### 📊 Daftar Kata")
        st.dataframe(pd.DataFrame([{
            "Kata": r["token"], 
            "Jenis": FLAG_STYLES[r["flag"]]["label"], 
            "Saran": ", ".join(r["rekomendasi"][:3]) if r["rekomendasi"] else "-",
            "Keterangan": r["catatan"] or "-"
        } for r in display_res]), use_container_width=True, hide_index=True)
