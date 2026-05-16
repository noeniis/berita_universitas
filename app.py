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
# RERANKING KANDIDAT DENGAN INDOBERT
# ==============================================================


def rerank_candidates_with_bert(
    sentence: str,
    original_token: str,
    candidates: List[str],
    tokenizer,
    model,
    device,
) -> List[dict]:
    """
    Untuk setiap kandidat koreksi, ganti token asli di kalimat,
    lalu score-kan dengan IndoBERT (prob class BENAR/CORRECT).
    Kembalikan daftar kandidat terurut dari skor tertinggi.
    """
    candidate_scores = []

    for cand in candidates:
        # Ganti token asli dengan kandidat dalam kalimat
        modified_sentence = re.sub(
            rf"\b{re.escape(original_token)}\b",
            cand,
            clean_whitespace(sentence),
            count=1,
        )

        enc = tokenizer(
            modified_sentence,
            max_length=MODEL_CFG["max_length"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )

        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        # probs[0] = prob CORRECT (class 0) → skor tinggi = kandidat lebih cocok
        score = float(probs[0])
        candidate_scores.append({"candidate": cand, "score": round(score, 4)})

    candidate_scores.sort(key=lambda x: x["score"], reverse=True)
    return candidate_scores


# ==============================================================
# PIPELINE ANALISIS TEKS
# ==============================================================


def analyze_text(
    text: str,
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
    """
    Pipeline Hybrid Reranking:
      1. Jaro-Winkler → deteksi token mencurigakan & generate kandidat koreksi
      2. IndoBERT     → rerank kandidat berdasarkan skor kontekstual
    Token dianggap ERROR jika JW memprediksi ERROR (pred == 1).
    Rekomendasi diurutkan ulang oleh IndoBERT berdasarkan
    seberapa baik kandidat cocok dalam konteks kalimat.
    """
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

            # ── Langkah 1: Jaro-Winkler ───────────────────────
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

            # Tentukan flag berdasarkan status leksikon
            if status == "KATA_INGGRIS":
                flag = "KATA_INGGRIS"
                tipe = "Kata Bahasa Inggris"
            elif status == "KATA_SERAPAN":
                flag = "KATA_SERAPAN"
                tipe = "Kata Serapan"
            elif jw_res["pred"] == 1:
                flag = "TYPO"
                tipe = "Typo"
            else:
                # Token valid secara leksikal → lewati
                continue

            # ── Langkah 2: IndoBERT Reranking (khusus TYPO) ───
            candidates = jw_res["top_k_recs"]
            best_match = jw_res["best_match"]
            best_score = 0.0
            reranked_candidates = candidates  # default: urutan JW

            if flag == "TYPO" and candidates:
                reranked = rerank_candidates_with_bert(
                    sentence=sent,
                    original_token=t,
                    candidates=candidates,
                    tokenizer=tokenizer,
                    model=bert_model,
                    device=device,
                )
                if reranked:
                    best_match = reranked[0]["candidate"]
                    best_score = reranked[0]["score"]
                    reranked_candidates = [x["candidate"] for x in reranked]

            # ── Catatan tambahan ───────────────────────────────
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
                    "jw_pred": "ERROR" if jw_res["pred"] else "OK",
                    "jw_sim": jw_res["max_sim"],
                    "best_match": best_match,
                    "best_score": round(best_score, 4),
                    "rekomendasi": reranked_candidates,
                    "catatan": catatan,
                    "highlight": True,
                }
            )

    return results


# ==============================================================
# RENDER TEKS BERWARNA
# ==============================================================


FLAG_STYLES = {
    "TYPO": {"label": "Salah ketik / tidak baku", "bg": "#ffd6d6", "border": "#d64545", "text": "#7a1111"},
    "KATA_INGGRIS": {"label": "Kata bahasa Inggris", "bg": "#fff2b3", "border": "#d4a017", "text": "#6b4f00"},
    "KATA_SERAPAN": {"label": "Kata serapan", "bg": "#dbeafe", "border": "#3b82f6", "text": "#1e3a8a"},
}

FLAG_ORDER = ["TYPO", "KATA_INGGRIS", "KATA_SERAPAN"]


def build_tooltip(row: dict) -> str:
    flag = row["flag"]
    token = row["token"]

    if flag == "TYPO":
        lines = [f"\u26a0\ufe0f \u201c{token}\u201d terdeteksi sebagai salah ketik"]
        if row.get("rekomendasi"):
            top = row["rekomendasi"]
            lines.append("Saran pengganti (reranked): " + ", ".join(top))
        elif row.get("best_match"):
            lines.append(f"Kata yang paling mirip: {row['best_match']}")
        lines.append("\u2192 Periksa kembali ejaan kata ini")

    elif flag == "KATA_INGGRIS":
        lines = [f"\U0001f310 \u201c{token}\u201d adalah kata berbahasa Inggris"]
        catatan = row.get("catatan", "")
        if catatan and "Padanan" in catatan:
            padanan = catatan.replace("Padanan KBBI: ", "").strip("'")
            lines.append(f"Padanan dalam bahasa Indonesia: {padanan}")
        else:
            lines.append("Gunakan padanan bahasa Indonesia jika tersedia,")
            lines.append("atau cetak miring jika tetap digunakan")

    elif flag == "KATA_SERAPAN":
        lines = [f"\U0001f4cc \u201c{token}\u201d adalah kata serapan dari bahasa asing"]
        lines.append("Pastikan penulisannya sudah sesuai KBBI")

    else:
        lines = [f"Kata: {token}"]

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
    st.markdown("### 📝 Penyunting Kata Berita")
    st.caption("Alat bantu pengecekan bahasa untuk berita universitas")
    st.caption("🔬 Model: Hybrid Reranking (JW + IndoBERT)")
    st.markdown("---")

    st.markdown("**Pengaturan Tampilan**")
    show_inggris = st.toggle("Tandai kata bahasa Inggris", value=True, help="Tampilkan kata-kata berbahasa Inggris yang ditemukan dalam teks")
    show_serapan = st.toggle("Tandai kata serapan", value=True, help="Tampilkan kata serapan asing yang sudah diserap ke bahasa Indonesia")
    skip_proper_noun = st.toggle("Abaikan nama orang/tempat", value=True, help="Nama orang, tempat, dan lembaga yang diawali huruf kapital tidak akan ditandai")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.82rem; color:#6b7280; line-height:1.6;">'
        'ℹ️ Angka, singkatan, dan akronim diabaikan secara otomatis.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Noeni Indah Sulistiyani · Teknik Informatika · UIN Jakarta")

st.title("📝 Penyunting Kata Berita")
st.markdown(
    "Tempel atau unggah teks berita, lalu sistem akan otomatis menandai kata yang perlu diperhatikan — "
    "mulai dari salah ketik, penggunaan kata asing, hingga kata serapan yang mungkin perlu disesuaikan."
)
st.markdown("---")

with st.spinner("Memuat sistem..."):
    tokenizer, bert_model, device = load_model()
    kbbi_set, inggris_set, whitelist_set, serapan_map, serapan_set, kbbi_list = load_lexicons()

st.success("Sistem siap digunakan.", icon="✅")

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
    with st.spinner("Menganalisis teks..."):
        t0 = time.time()
        results = analyze_text(
            text_to_run,
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
    c1.metric("Kata Dianalisis", total_tok)
    c2.metric("Salah Ketik", n_err)
    c3.metric("Total Ditandai", n_flag)
    c4.metric("Kata Asing", n_inggris)
    c5.metric("Waktu Analisis", f"{elapsed}s")

    st.markdown("### 📄 Hasil Pengecekan")
    st.markdown("_Arahkan kursor ke kata yang ditandai untuk melihat keterangan dan saran perbaikannya._")
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
        st.markdown("### 📊 Daftar Kata yang Ditandai")
        tabel = pd.DataFrame([
            {
                "Kata": r["token"],
                "Jenis Temuan": FLAG_STYLES.get(r["flag"], {}).get("label", r["flag"]),
                "Saran Perbaikan": ", ".join(r["rekomendasi"]) if r["rekomendasi"] else "-",
                "Keterangan": r["catatan"] or "-",
            }
            for r in results_display
        ])

        st.dataframe(
            tabel,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 🔎 Detail Per Kata")
        for r in results_display:
            flag_label = FLAG_STYLES.get(r["flag"], {}).get("label", r["flag"])
            with st.expander(f"• **{r['token']}** — {flag_label}"):
                col_l, col_r = st.columns(2)

                with col_l:
                    st.markdown(f"**Kata:** `{r['token']}`")
                    st.markdown(f"**Jenis temuan:** {flag_label}")
                    if r["flag"] == "TYPO":
                        st.markdown(f"**Rekomendasi teratas (IndoBERT):** `{r['best_match']}`")
                        if r.get("best_score", 0) > 0:
                            st.markdown(f"**Skor kontekstual:** {r['best_score']:.4f}")
                    if r["catatan"]:
                        st.info(r["catatan"])

                with col_r:
                    if r["flag"] == "TYPO" and r["rekomendasi"]:
                        st.markdown("**Saran pengganti (top 5):**")
                        for rec in r["rekomendasi"]:
                            st.code(rec)
                    elif r["flag"] == "KATA_INGGRIS":
                        st.markdown("**Saran:**")
                        catatan = r.get("catatan", "")
                        if catatan and "Padanan" in catatan:
                            padanan = catatan.replace("Padanan KBBI: ", "").strip("'")
                            st.code(padanan)
                        else:
                            st.caption("Gunakan padanan bahasa Indonesia, atau cetak miring jika dipertahankan.")
                    else:
                        st.caption("Pastikan penulisan kata ini sudah sesuai KBBI.")

                st.markdown("**Konteks kalimat:**")
                highlighted = re.sub(
                    rf"\b{re.escape(r['token'])}\b",
                    f"**:red[{r['token']}]**",
                    r["kalimat"],
                    count=1,
                )
                st.markdown(f"> {highlighted}")
