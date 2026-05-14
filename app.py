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
# STYLE VISUAL (CSS)
# ==============================================================

st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: Arial, sans-serif; }

/* Kotak Putih untuk Teks Anotasi */
.text-preview-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 24px;
    line-height: 2.0em;
    font-size: 1.05rem;
    color: #111827;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 20px;
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
# UTILITAS TEKS & RESOURCES (Sama Seperti Sebelumnya)
# ==============================================================

def normalize_unicode(text: str) -> str:
    if pd.isna(text): return ""
    return unicodedata.normalize("NFKC", str(text))

def clean_whitespace(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text).replace("\u200b", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()

def normalize_token(token: str) -> str:
    if pd.isna(token): return ""
    return clean_whitespace(normalize_unicode(token)).lower().strip()

def escape_html(text: str) -> str:
    return html_lib.escape(text)

def extract_sentence_spans(text: str) -> List[Tuple[str, int, int]]:
    spans = []
    if not text or not text.strip(): return spans
    pattern = re.compile(r".+?(?:[.!?](?:\s+|$)|$)", flags=re.DOTALL)
    for match in pattern.finditer(text):
        sent = match.group().strip()
        if sent: spans.append((sent, match.start(), match.end()))
    return spans

def tokenize_with_spans(sentence: str) -> List[Tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\b\w+\b", sentence, flags=re.UNICODE)]

def is_number_token(token: str) -> bool:
    return bool(re.fullmatch(r"[\d.,:/%-]+", token))

def is_all_caps_token(token: str) -> bool:
    return len(token) >= 2 and token.isupper()

def is_title_case_name(token: str, position: int, kbbi_set: set, inggris_set: set, serapan_set: set, whitelist_set: set) -> bool:
    t = normalize_token(token)
    if not t or len(t) <= 2: return False
    if t in kbbi_set or t in inggris_set or t in serapan_set or t in whitelist_set: return False
    return token[0].isupper() and not token.isupper()

# ==============================================================
# LOAD RESOURCES (Model & Leksikon)
# ==============================================================

_TMP = tempfile.gettempdir()
MODEL_LOCAL = os.path.join(_TMP, "model_indobert_best")
LEXICON_LOCAL = os.path.join(_TMP, "leksikon")
os.makedirs(MODEL_LOCAL, exist_ok=True)
os.makedirs(LEXICON_LOCAL, exist_ok=True)

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(os.path.join(MODEL_LOCAL, "config.json")):
        gdown.download_folder(id=DRIVE_IDS["model_indobert"], output=MODEL_LOCAL, quiet=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(MODEL_LOCAL)
    model = BertForSequenceClassification.from_pretrained(MODEL_LOCAL, num_labels=MODEL_CFG["num_labels"])
    model.to(device).eval()
    return tokenizer, model, device

@st.cache_resource(show_spinner=False)
def load_lexicons():
    def read_csv_safe(path):
        try: return pd.read_csv(path, encoding="utf-8-sig")
        except: return pd.read_csv(path, encoding="latin1")

    lex_dfs = {}
    keys = ["kbbi", "kata_inggris", "kata_serapan", "akronim", "daftar_lembaga", "daftar_nama_orang", "istilah_islam", "sample_correct_2025"]
    for key in keys:
        p = os.path.join(LEXICON_LOCAL, f"{key}.csv")
        if not os.path.exists(p): gdown.download(id=DRIVE_IDS[key], output=p, quiet=True)
        lex_dfs[key] = read_csv_safe(p)

    def to_set(df, col):
        if df is None or df.empty: return set()
        c = col if col in df.columns else df.columns[0]
        return set(normalize_token(v) for v in df[c].dropna() if len(normalize_token(v)) >= 2)

    kbbi_set = to_set(lex_dfs["kbbi"], "kata")
    inggris_set = to_set(lex_dfs["kata_inggris"], "headword") - kbbi_set
    whitelist_set = set()
    for k in ["akronim", "daftar_lembaga", "daftar_nama_orang", "istilah_islam", "sample_correct_2025"]:
        whitelist_set.update(to_set(lex_dfs[k], lex_dfs[k].columns[0]))

    serapan_map, serapan_set = {}, set()
    df_s = lex_dfs["kata_serapan"]
    col_asal = next(c for c in df_s.columns if "asal" in c.lower() or "asing" in c.lower())
    col_hasil = next(c for c in df_s.columns if "serapan" in c.lower() or "hasil" in c.lower())
    for _, r in df_s.iterrows():
        a, h = normalize_token(str(r[col_asal])), normalize_token(str(r[col_hasil]))
        if a and h: serapan_map[a] = h; serapan_set.add(a)

    return kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, sorted(kbbi_set)

# ==============================================================
# LOGIKA PREDIKSI & JW
# ==============================================================

def jaro_winkler_similarity(s1, s2):
    if s1 == s2: return 1.0
    len1, len2 = len(s1), len(s2)
    match_dist = max(0, max(len1, len2) // 2 - 1)
    s1m, s2m = [False]*len1, [False]*len2
    matches = 0
    for i in range(len1):
        for j in range(max(0, i-match_dist), min(i+match_dist+1, len2)):
            if not s2m[j] and s1[i] == s2[j]:
                s1m[i] = s2m[j] = True; matches += 1; break
    if matches == 0: return 0.0
    trans, k = 0, 0
    for i in range(len1):
        if s1m[i]:
            while not s2m[k]: k += 1
            if s1[i] != s2[k]: trans += 1
            k += 1
    j = (matches/len1 + matches/len2 + (matches - trans/2)/matches) / 3
    p = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]: p += 1
        else: break
    return j + p * 0.1 * (1 - j)

def analyze_text(text, tokenizer, bert_model, device, kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, kbbi_list, skip_proper_noun):
    sentences = extract_sentence_spans(text)
    results = []
    for sent, s_start, _ in sentences:
        tokens = tokenize_with_spans(sent)
        for pos, (tok, start, end) in enumerate(tokens):
            t = normalize_token(tok)
            if not t or len(t) < 2 or is_number_token(t) or is_all_caps_token(tok) or t in whitelist_set: continue
            if skip_proper_noun and is_title_case_name(tok, pos, kbbi_set, inggris_set, serapan_set, whitelist_set): continue

            # Jalur Cepat: Cek Kamus
            if t in serapan_set:
                flag, status = "KATA_SERAPAN", "KATA_SERAPAN"
            elif t in kbbi_set or stemmer.stem(t) in kbbi_set:
                continue
            elif t in inggris_set:
                flag, status = "KATA_INGGRIS", "KATA_INGGRIS"
            else:
                # Jalur Analisis: JW + BERT
                sims = sorted([(w, jaro_winkler_similarity(t, w)) for w in kbbi_list], key=lambda x: x[1], reverse=True)
                best_w, best_s = sims[0]
                
                enc = tokenizer(sent, t, max_length=MODEL_CFG["max_length"], truncation=True, padding="max_length", return_tensors="pt")
                with torch.no_grad():
                    out = bert_model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), token_type_ids=enc["token_type_ids"].to(device))
                probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
                
                is_err = (best_s < JW_CFG["threshold"]) or (np.argmax(probs) == 1)
                if not is_err: continue
                flag, status = "TYPO", "TIDAK_DIKENAL"

            # Bangun Rekomendasi & Catatan
            recs = []
            catatan = ""
            if flag == "KATA_SERAPAN":
                h = serapan_map.get(t)
                catatan = f"Saran serapan baku: '{h}'"; recs = [h]
            elif flag == "KATA_INGGRIS":
                h = serapan_map.get(t)
                catatan = f"Padanan KBBI: '{h}'" if h else "Cetak miring jika dipertahankan"
            else:
                recs = [w for w, _ in sims[:5]]
            
            results.append({
                "token": tok, "flag": flag, "start": s_start + start, "end": s_start + end,
                "rekomendasi": recs, "catatan": catatan, "kalimat": sent, "best_match": best_w if flag=="TYPO" else ""
            })
    return results

# ==============================================================
# UI RENDERING
# ==============================================================

FLAG_STYLES = {
    "TYPO": {"label": "Typo / Kata tidak baku", "bg": "#ffd6d6", "text": "#7a1111", "border": "#d64545"},
    "KATA_INGGRIS": {"label": "Kata bahasa Inggris", "bg": "#fff2b3", "text": "#6b4f00", "border": "#d4a017"},
    "KATA_SERAPAN": {"label": "Kata asal serapan", "bg": "#dbeafe", "text": "#1e3a8a", "border": "#3b82f6"},
}

def render_annotated_box(text, rows):
    row_map = {(r["start"], r["end"]): r for r in rows}
    parts, cursor = [], 0
    for m in re.finditer(r"\b\w+\b", text):
        parts.append(escape_html(text[cursor:m.start()]))
        r = row_map.get((m.start(), m.end()))
        if r:
            s = FLAG_STYLES[r["flag"]]
            parts.append(f'<span style="background:{s["bg"]}; color:{s["text"]}; border:1px solid {s["border"]}; border-radius:4px; padding:0 4px; font-weight:600;">{escape_html(m.group())}</span>')
        else: parts.append(escape_html(m.group()))
        cursor = m.end()
    parts.append(escape_html(text[cursor:]))
    html_content = "".join(parts)
    st.markdown(f'<div class="text-preview-box">{html_content}</div>', unsafe_allow_html=True)

# ==============================================================
# MAIN APP
# ==============================================================

st.title("📝 Penyunting Kata Berita")
st.markdown("Normalisasi teks menggunakan **Hybrid Jaro-Winkler & IndoBERT**.")

with st.sidebar:
    st.header("Pengaturan")
    show_ing = st.checkbox("Tandai Kata Inggris", True)
    show_ser = st.checkbox("Tandai Kata Serapan", True)
    skip_pn = st.checkbox("Abaikan Nama (Proper Noun)", True)
    st.divider()
    st.caption("Noeni Indah Sulistiyani - UIN Jakarta")

tokenizer, bert_model, device = load_model()
kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, kbbi_list = load_lexicons()

input_text = st.text_area("Input Teks Berita:", height=200)

if st.button("🔍 Jalankan Analisis", type="primary") and input_text:
    t0 = time.time()
    raw_results = analyze_text(input_text, tokenizer, bert_model, device, kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, kbbi_list, skip_pn)
    
    # Filter berdasarkan sidebar
    results = [r for r in raw_results if (r["flag"] != "KATA_INGGRIS" or show_ing) and (r["flag"] != "KATA_SERAPAN" or show_ser)]
    elapsed = round(time.time() - t0, 2)

    # 1. Ringkasan Statistik
    c1, c2, c3 = st.columns(3)
    c1.metric("Typo/Tidak Baku", sum(1 for r in results if r["flag"]=="TYPO"))
    c2.metric("Total Temuan", len(results))
    c3.metric("Waktu", f"{elapsed}s")

    # 2. Kotak Anotasi (KOTAK PUTIH)
    st.subheader("Hasil Anotasi")
    render_annotated_box(input_text, results)

    # 3. Tabel Ringkasan
    if results:
        st.subheader("Ringkasan Temuan")
        df_view = pd.DataFrame([{
            "Token": r["token"], "Jenis": FLAG_STYLES[r["flag"]]["label"],
            "Saran Utama": r["rekomendasi"][0] if r["rekomendasi"] else "-"
        } for r in results])
        st.dataframe(df_view, use_container_width=True, hide_index=True)

        # 4. Detail Per Token (DETAIL DI BAWAH TABEL)
        st.subheader("📑 Detail Per Token")
        for i, r in enumerate(results):
            with st.expander(f"{i+1}. {r['token']} ({FLAG_STYLES[r['flag']]['label']})"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write(f"**Kata Asli:** `{r['token']}`")
                    st.write(f"**Konteks:** ...{r['kalimat'][:50]}...")
                with col2:
                    if r["flag"] == "TYPO":
                        st.write("**Saran Perbaikan (Top 5):**")
                        st.success(", ".join(r["rekomendasi"]))
                    elif r["catatan"]:
                        st.info(r["catatan"])
