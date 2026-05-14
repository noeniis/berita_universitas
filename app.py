
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
    page_title="Sistem Penyuntingan Kata Berita UIN",
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

.token-typo {
    background: #ffd6d6;
    color: #7a1111;
    border: 1px solid #d64545;
    border-radius: 6px;
    padding: 1px 5px;
    font-weight: 600;
    cursor: help;
}

.token-english {
    background: #fff2b3;
    color: #6b4f00;
    border: 1px solid #d4a017;
    border-radius: 6px;
    padding: 1px 5px;
    font-weight: 600;
    cursor: help;
}

.token-serapan {
    background: #dbeafe;
    color: #1e3a8a;
    border: 1px solid #3b82f6;
    border-radius: 6px;
    padding: 1px 5px;
    font-weight: 600;
    cursor: help;
}

.legend-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 10px;
    margin-bottom: 18px;
}

.legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.88rem;
}

.legend-dot {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    display: inline-block;
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
    """Kembalikan daftar (kalimat, start, end) dengan posisi di teks asli."""
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
    """Token kata pada sebuah kalimat, lengkap dengan posisi relatif."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\b\w+\b", sentence, flags=re.UNICODE)]


def is_number_token(token: str) -> bool:
    return bool(re.fullmatch(r"[\d.,:/%-]+", token))


def is_all_caps_token(token: str) -> bool:
    return len(token) >= 2 and token.isupper()


def is_title_case_name(
    token: str,
    position: int,
    kbbi_set: set,
    inggris_set: set,
    serapan_set: set,
    whitelist_set: set,
) -> bool:
    """
    Heuristik ringan untuk nama orang/tempat/lembaga.
    Token yang sudah jelas valid tidak dianggap nama.
    """
    t = normalize_token(token)
    if not t or len(t) <= 2:
        return False

    if t in kbbi_set or t in inggris_set or t in serapan_set or t in whitelist_set:
        return False

    # Nama/proper noun biasanya diawali huruf kapital dan bukan ALL CAPS
    return token[0].isupper() and not token.isupper()


# ==============================================================
# PATH LOKAL UNTUK CACHE UNDUHAN
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

# ==============================================================
# LOAD RESOURCES
# ==============================================================


@st.cache_resource(show_spinner=False)
def load_model():
    """Unduh (jika belum ada) dan load model IndoBERT dari Drive."""
    config_path = os.path.join(MODEL_LOCAL, "config.json")
    if not os.path.exists(config_path):
        gdown.download_folder(
            id=DRIVE_IDS["model_indobert"],
            output=MODEL_LOCAL,
            quiet=False,
            use_cookies=False,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(MODEL_LOCAL)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_LOCAL,
        num_labels=MODEL_CFG["num_labels"],
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner=False)
def load_lexicons():
    """Unduh (jika belum ada) dan bangun semua set leksikon dari Drive."""

    def read_csv_safe(path: str) -> pd.DataFrame:
        for enc in ["utf-8", "utf-8-sig", "latin1"]:
            try:
                df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
                df.columns = [c.strip() for c in df.columns]
                return df
            except Exception:
                continue
        return pd.DataFrame()

    def to_set(df: pd.DataFrame, col: str) -> set:
        if df is None or df.empty:
            return set()

        if col not in df.columns:
            col = df.columns[0] if len(df.columns) > 0 else None

        if col is None:
            return set()

        vals = df[col].dropna().astype(str)
        normalized = [normalize_token(v) for v in vals]
        return set(v for v in normalized if v and len(v) >= 2)

    lex_dfs: Dict[str, pd.DataFrame] = {}
    keys = [
        "kbbi",
        "kata_inggris",
        "kata_serapan",
        "akronim",
        "daftar_lembaga",
        "daftar_nama_orang",
        "istilah_islam",
    ]
    if "sample_correct_2025" in DRIVE_IDS:
        keys.append("sample_correct_2025")

    for key in keys:
        local_path = os.path.join(LEXICON_LOCAL, f"{key}.csv")
        if not os.path.exists(local_path):
            try:
                gdown.download(id=DRIVE_IDS[key], output=local_path, quiet=True)
            except Exception:
                pass
        lex_dfs[key] = read_csv_safe(local_path)

    kbbi_set = to_set(lex_dfs.get("kbbi", pd.DataFrame()), "kata")

    df_ing = lex_dfs.get("kata_inggris", pd.DataFrame())
    if not df_ing.empty:
        if "headword" in df_ing.columns:
            ing_col = "headword"
        else:
            ing_col = df_ing.columns[0]
        inggris_set = to_set(df_ing, ing_col) - kbbi_set
    else:
        inggris_set = set()

    whitelist_set = set()
    for key in ["akronim", "daftar_lembaga", "daftar_nama_orang", "istilah_islam"]:
        whitelist_set.update(to_set(lex_dfs.get(key, pd.DataFrame()), LEXICON_COL_MAP[key]))

    # domain vocabulary tambahan dari sample_correct_2025
    df_domain = lex_dfs.get("sample_correct_2025", pd.DataFrame())
    if not df_domain.empty:
        first_col = df_domain.columns[0]
        vals = df_domain[first_col].dropna().astype(str).str.lower().str.strip()
        domain_vocab = set()
        for row in vals:
            toks = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", row)
            domain_vocab.update(tok.lower() for tok in toks if len(tok) >= 3)
        whitelist_set.update(domain_vocab)

    serapan_map: Dict[str, str] = {}
    serapan_set = set()
    df_s = lex_dfs.get("kata_serapan", pd.DataFrame())
    if not df_s.empty:
        col_asal = next(
            (c for c in df_s.columns if "asal" in c.lower() or "asing" in c.lower()),
            df_s.columns[0],
        )
        col_serapan = next(
            (c for c in df_s.columns if "serapan" in c.lower() or "hasil" in c.lower()),
            df_s.columns[-1],
        )
        for _, row in df_s.iterrows():
            asal = normalize_token(str(row[col_asal]))
            serapan = normalize_token(str(row[col_serapan]))
            if asal and serapan:
                serapan_map[asal] = serapan
                serapan_set.add(asal)

    kbbi_list = sorted(kbbi_set)
    return kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, kbbi_list


# ==============================================================
# ALGORITMA JARO-WINKLER
# ==============================================================


def jaro_similarity(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    match_dist = max(0, max(len1, len2) // 2 - 1)
    s1m = [False] * len1
    s2m = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        for j in range(max(0, i - match_dist), min(i + match_dist + 1, len2)):
            if s2m[j] or s1[i] != s2[j]:
                continue
            s1m[i] = s2m[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1m[i]:
            continue
        while not s2m[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    jaro = jaro_similarity(s1, s2)
    prefix = 0
    for i in range(min(4, len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


# ==============================================================
# KLASIFIKASI TOKEN
# ==============================================================


def classify_token(token: str, kbbi_set, inggris_set, whitelist_set, serapan_set) -> str:
    t = normalize_token(token)
    if not t:
        return "KOSONG"
    if t in whitelist_set:
        return "WHITELIST_KHUSUS"
    if t in serapan_set:
        return "KATA_SERAPAN"
    if t in kbbi_set:
        return "KBBI_VALID"
    if t in inggris_set:
        return "KATA_INGGRIS"

    stem = stemmer.stem(t)
    if stem in kbbi_set:
        return "KBBI_VALID"

    return "TIDAK_DIKENAL"


# ==============================================================
# PREDIKSI JARO-WINKLER
# ==============================================================


def predict_jw(
    token: str,
    kbbi_set,
    inggris_set,
    whitelist_set,
    serapan_set,
    kbbi_list,
    threshold: float,
    top_k: int = 5,
) -> dict:
    t = normalize_token(token)
    status = classify_token(t, kbbi_set, inggris_set, whitelist_set, serapan_set)

    if status in ("WHITELIST_KHUSUS", "KBBI_VALID", "KATA_SERAPAN", "KOSONG", "KATA_INGGRIS"):
        return {
            "pred": 0,
            "max_sim": 1.0,
            "best_match": t,
            "top_k_recs": [],
            "status": status,
        }

    sims = [(w, jaro_winkler_similarity(t, w)) for w in kbbi_list]
    sims.sort(key=lambda x: x[1], reverse=True)
    best_word, best_sim = sims[0]
    pred = 0 if best_sim >= threshold else 1

    return {
        "pred": pred,
        "max_sim": round(best_sim, 4),
        "best_match": best_word,
        "top_k_recs": [w for w, _ in sims[:top_k]],
        "status": status,
    }


# ==============================================================
# PREDIKSI INDOBERT
# ==============================================================


def predict_bert(kalimat: str, token: str, tokenizer, model, device) -> dict:
    text_a = clean_whitespace(kalimat)
    text_b = normalize_token(token)
    enc = tokenizer(
        text_a,
        text_b,
        max_length=MODEL_CFG["max_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            token_type_ids=enc["token_type_ids"].to(device),
        )
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return {
        "pred": pred,
        "prob_correct": round(float(probs[0]), 4),
        "prob_error": round(float(probs[1]), 4),
    }


# ==============================================================
# KEPUTUSAN HYBRID
# ==============================================================


def decide_final_pred(model_choice: str, jw_pred: int, bert_pred: int) -> int:
    if model_choice == "Jaro-Winkler":
        return jw_pred
    if model_choice == "IndoBERT":
        return bert_pred
    return 1 if (jw_pred == 1 or bert_pred == 1) else 0


# ==============================================================
# PIPELINE ANALISIS TEKS
# ==============================================================


def analyze_text(
    text: str,
    model_choice: str,
    tokenizer,
    bert_model,
    device,
    kbbi_set,
    inggris_set,
    whitelist_set,
    serapan_map,
    serapan_set,
    kbbi_list,
    skip_proper_noun: bool = True,
) -> List[dict]:
    sentences = extract_sentence_spans(text)
    results: List[dict] = []

    for sent, sent_start, _ in sentences:
        tokens = tokenize_with_spans(sent)
        for pos, (tok, start, end) in enumerate(tokens):
            t = normalize_token(tok)
            if not t or len(t) < 2:
                continue
            if is_number_token(t):
                continue
            if is_all_caps_token(tok):
                continue
            if t in whitelist_set:
                continue
            if skip_proper_noun and is_title_case_name(tok, pos, kbbi_set, inggris_set, serapan_set, whitelist_set):
                continue

            jw_res = predict_jw(
                t,
                kbbi_set,
                inggris_set,
                whitelist_set,
                serapan_set,
                kbbi_list,
                threshold=JW_CFG["threshold"],
                top_k=JW_CFG["top_k"],
            )

            status = jw_res["status"]

            # Short-circuit untuk token yang sudah valid / non-error
            if status in [
                "KATA_INGGRIS",
                "KATA_SERAPAN",
                "WHITELIST_KHUSUS",
                "KBBI_VALID",
                "ANGKA",
            ]:
                bert_res = {
                    "pred": 0,
                    "prob_correct": 1.0,
                    "prob_error": 0.0,
                }
            else:
                bert_res = predict_bert(sent, t, tokenizer, bert_model, device)

            jw_pred = jw_res["pred"]
            bert_pred = bert_res["pred"]
            final_pred = decide_final_pred(model_choice, jw_pred, bert_pred)

            if status == "KATA_INGGRIS":
                flag = "KATA_INGGRIS"
                tipe = "Kata Bahasa Inggris"
                final_pred = 0
            elif status == "KATA_SERAPAN":
                flag = "KATA_SERAPAN"
                tipe = "Kata Serapan"
                final_pred = 0
            elif final_pred == 1:
                flag = "TYPO"
                tipe = "Typo"
            else:
                continue

            recs = jw_res["top_k_recs"]
            catatan = ""
            if status == "KATA_INGGRIS":
                padanan = serapan_map.get(t)
                catatan = f"Padanan KBBI: '{padanan}'" if padanan else "Gunakan huruf miring jika dipertahankan"

            results.append(
                {
                    "token": tok,
                    "token_norm": t,
                    "kalimat": sent,
                    "sent_start": sent_start,
                    "start": sent_start + start,
                    "end": sent_start + end,
                    "flag": flag,
                    "tipe_error": tipe,
                    "jw_pred": "ERROR" if jw_pred else "OK",
                    "bert_pred": "ERROR" if bert_pred else "OK",
                    "final_pred": "ERROR" if final_pred else "OK",
                    "prob_error": bert_res["prob_error"],
                    "prob_correct": bert_res["prob_correct"],
                    "jw_sim": jw_res["max_sim"],
                    "best_match": jw_res["best_match"],
                    "rekomendasi": recs,
                    "catatan": catatan,
                    "highlight": True,
                }
            )

    return results


# ==============================================================
# RENDER TEKS BERWARNA
# ==============================================================


FLAG_STYLES = {
    "TYPO": {"label": "Salah ketik", "bg": "#ffd6d6", "border": "#d64545", "text": "#7a1111"},
    "KATA_INGGRIS": {"label": "Kata asing", "bg": "#fff2b3", "border": "#d4a017", "text": "#6b4f00"},
    "KATA_SERAPAN": {"label": "Kata serapan", "bg": "#dbeafe", "border": "#3b82f6", "text": "#1e3a8a"},
}

FLAG_ORDER = ["TYPO", "KATA_INGGRIS", "KATA_SERAPAN"]


def build_tooltip(row: dict) -> str:
    label = FLAG_STYLES.get(row["flag"], {}).get("label", row["flag"])
    lines = [
        f"Token: {row['token']}",
        f"Label: {label}",
        f"JW: {row['jw_pred']} (sim={row['jw_sim']})",
        f"BERT: {row['bert_pred']} (prob error={row['prob_error']})",
    ]
    if row.get("catatan"):
        lines.append(row["catatan"])
    if row.get("rekomendasi"):
        lines.append("Rekomendasi: " + ", ".join(row["rekomendasi"][:5]))
    return "\n".join(lines)


def render_highlighted_text(text: str, rows: List[dict]) -> str:
    row_map = {(r["start"], r["end"]): r for r in rows}
    parts: List[str] = []
    cursor = 0

    for m in re.finditer(r"\b\w+\b", text, flags=re.UNICODE):
        parts.append(escape_html(text[cursor:m.start()]))
        row = row_map.get((m.start(), m.end()))
        token_html = html_lib.escape(m.group())
        if row:
            style = FLAG_STYLES.get(row["flag"], FLAG_STYLES["TYPO"])
            tooltip = escape_html(build_tooltip(row))
            span = (
                f'<span title="{tooltip}" '
                f'style="background:{style["bg"]}; color:{style["text"]}; '
                f'border:1px solid {style["border"]}; border-radius:6px; '
                f'padding:1px 5px; font-weight:600; white-space:nowrap; cursor:help;">'
                f"{token_html}</span>"
            )
            parts.append(span)
        else:
            parts.append(f'<span style="color:#111827;">{token_html}</span>')
        cursor = m.end()
    parts.append(escape_html(text[cursor:]))

    return (
        '<div style="line-height:1.95; font-size:1.02rem; white-space:pre-wrap; '
        'word-break:break-word; color:#111827;">'
        + "".join(parts)
        + "</div>"
    )


def render_legend() -> None:
    chips = []
    for key in FLAG_ORDER:
        s = FLAG_STYLES[key]
        chips.append(
            f'<span style="display:inline-block; margin:0 10px 10px 0; padding:4px 10px; '
            f'border-radius:999px; background:{s["bg"]}; color:{s["text"]}; '
            f'border:1px solid {s["border"]}; font-size:0.92rem;">'
            f'{s["label"]}</span>'
        )
    st.markdown(
        "<div style='margin-top:4px; margin-bottom:8px;'><b>Legenda warna:</b> "
        + "".join(chips)
        + "</div>",
        unsafe_allow_html=True,
    )


# ==============================================================
# ANTARMUKA STREAMLIT
# ==============================================================

with st.sidebar:
    st.markdown("### 📝 Sistem Penyuntingan Kata")
    st.caption("Berita UIN Jakarta · Demo Streamlit")
    st.markdown("---")

    model_choice = st.selectbox(
        "Pilih Model Deteksi",
        options=["Hybrid-OR", "IndoBERT", "Jaro-Winkler"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Performa Model (test set)**")
    perf = {
        "Hybrid-OR": {"F1": "0.9976", "Recall": "0.9988", "Precision": "0.9964"},
        "IndoBERT": {"F1": "0.9952", "Recall": "0.9940", "Precision": "0.9964"},
        "Jaro-Winkler": {"F1": "0.9824", "Recall": "0.9654", "Precision": "1.0000"},
    }
    for metric, val in perf[model_choice].items():
        st.metric(metric, val)

    st.markdown("---")
    show_inggris = st.toggle("Tampilkan kata Inggris", value=True)
    show_serapan = st.toggle("Tampilkan kata serapan", value=True)
    skip_proper_noun = st.toggle("Lewati nama orang/tempat (huruf kapital)", value=True)
    st.caption("Gelar akademik, akronim ALL-CAPS, dan angka selalu dilewati otomatis.")
    st.markdown("---")
    st.caption("Noeni Indah Sulistiyani\nTeknik Informatika · UIN Jakarta")

st.title("📝 Sistem Rekomendasi Penyuntingan Kata")
st.markdown(
    "Deteksi kesalahan penulisan pada teks berita universitas menggunakan **Jaro-Winkler** dan **IndoBERT**."
)
st.markdown("---")

with st.spinner("Memuat model dan leksikon..."):
    tokenizer, bert_model, device = load_model()
    kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, kbbi_list = load_lexicons()

st.success(f"Model **{model_choice}** siap digunakan.", icon="✅")

tab_teks, tab_file = st.tabs(["✏️ Input Teks", "📂 Upload File"])

with tab_teks:
    input_text = st.text_area(
        "Masukkan teks berita:",
        height=180,
        placeholder=(
            "Contoh: Rektor UIN Jakarta menyambut positif pencapaian ini. "
            "Menurutnya capaian ini merupakan bagian dari upaya berkelanjutan "
            "universitas dalam memperkuat kualitas academic di tingkat global."
        ),
    )
    run_teks = st.button("🔍 Analisis", type="primary", use_container_width=True, key="btn_teks")

with tab_file:
    uploaded = st.file_uploader("Upload file berita (.txt atau .docx)", type=["txt", "docx"])
    file_text = ""
    run_file = False
    if uploaded:
        if uploaded.name.endswith(".txt"):
            file_text = uploaded.read().decode("utf-8", errors="replace")
        else:
            import docx as _docx

            doc = _docx.Document(uploaded)
            file_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        st.text_area("Isi file:", value=file_text, height=180, disabled=True)
        run_file = st.button("🔍 Analisis File", type="primary", use_container_width=True, key="btn_file")

text_to_run = ""
if run_teks and input_text.strip():
    text_to_run = input_text
elif run_file and file_text.strip():
    text_to_run = file_text

if text_to_run:
    with st.spinner(f"Menganalisis dengan {model_choice}..."):
        t0 = time.time()
        results = analyze_text(
            text_to_run,
            model_choice,
            tokenizer,
            bert_model,
            device,
            kbbi_set,
            inggris_set,
            whitelist_set,
            serapan_map,
            serapan_set,
            kbbi_list,
            skip_proper_noun=skip_proper_noun,
        )
        elapsed = round(time.time() - t0, 2)

    results_display = []
    for r in results:
        if r["flag"] == "KATA_INGGRIS" and not show_inggris:
            continue
        if r["flag"] == "KATA_SERAPAN" and not show_serapan:
            continue
        results_display.append(r)

    st.markdown("---")

    total_tok = len([t for t in re.findall(r"\b\w+\b", text_to_run, flags=re.UNICODE) if len(t) >= 2])
    n_err = sum(1 for r in results_display if r["flag"] == "TYPO")
    n_flag = len(results_display)
    n_inggris = sum(1 for r in results_display if r["flag"] == "KATA_INGGRIS")
    n_serapan = sum(1 for r in results_display if r["flag"] == "KATA_SERAPAN")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Token", total_tok)
    c2.metric("Token Bermasalah", n_err)
    c3.metric("Token Diflag", n_flag)
    c4.metric("Kata Inggris", n_inggris)
    c5.metric("Waktu Analisis", f"{elapsed}s")

    st.markdown("### 📄 Teks dengan Anotasi")
    st.markdown("_Arahkan kursor ke kata yang ditandai untuk melihat detail._")
    render_legend()

    if results_display:
        highlighted_html = render_highlighted_text(text_to_run, results_display)
        st.markdown(
            f'<div class="text-preview-box">{highlighted_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.success("✅ Tidak ditemukan kata yang perlu ditandai.", icon="✅")

    if results_display:
        st.markdown("### 📊 Tabel Hasil Deteksi")
        tabel = pd.DataFrame([
            {
                "Token": r["token"],
                "Flag": FLAG_STYLES.get(r["flag"], {}).get("label", r["flag"]),
                "Tipe Error": r["tipe_error"],
                "JW": r["jw_pred"],
                "BERT": r["bert_pred"],
                "Skor JW": r["jw_sim"],
                "Prob Error BERT": r["prob_error"],
                "Kandidat Terdekat": r["best_match"],
                "Rekomendasi": ", ".join(r["rekomendasi"]) if r["rekomendasi"] else "-",
                "Catatan": r["catatan"] or "-",
            }
            for r in results_display
        ])

        st.dataframe(
            tabel,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Skor JW": st.column_config.NumberColumn(format="%.4f"),
                "Prob Error BERT": st.column_config.NumberColumn(format="%.4f", min_value=0, max_value=1),
            },
        )

        st.markdown("### 🔎 Detail Per Token")
        for r in results_display:
            with st.expander(f"• **{r['token']}** — {r['tipe_error']}"):
                col_l, col_r = st.columns(2)

                with col_l:
                    st.markdown(f"**Token:** `{r['token']}`")
                    st.markdown(f"**Tipe:** {r['tipe_error']}")
                    st.markdown(f"**Jaro-Winkler:** {r['jw_pred']} (sim = {r['jw_sim']})")
                    st.markdown(f"**IndoBERT:** {r['bert_pred']} (prob error = {r['prob_error']})")
                    if r["catatan"]:
                        st.info(r["catatan"])

                with col_r:
                    st.markdown("**Rekomendasi kata (top-5):**")
                    if r["rekomendasi"]:
                        for rec in r["rekomendasi"]:
                            st.code(rec)
                    else:
                        st.caption("Tidak ada rekomendasi spesifik.")

                st.markdown("**Kalimat konteks:**")
                highlighted = re.sub(
                    rf"\b{re.escape(r['token'])}\b",
                    f"**:red[{r['token']}]**",
                    r["kalimat"],
                    count=1,
                )
                st.markdown(f"> {highlighted}")
