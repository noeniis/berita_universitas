# config.py
# Konfigurasi ID Google Drive dan parameter model

DRIVE_IDS = {
    # Folder model IndoBERT (fine-tuned)
    "model_indobert"    : "1kF74xhsbfFDlOcu4_DgwMBjYVSOuYSMd",

    # File leksikon
    "kbbi"              : "1JWMWhw7PJNJk-s65RMYDSH4f2pAW5mV4",
    "kata_inggris"      : "1Ho6Q5YZ00uAfw3H0wgFSQ6UJ-2iD-NFG",
    "kata_serapan"      : "1hNwi7pFTPk1XVCA3pCIhsmX5NM767Zky",
    "akronim"           : "1cyhIF4Gs1JyaTuFRAu4Jzw7uY_EbJ2Fx",
    "daftar_lembaga"    : "1CHnEqraKO_JJgLcpYeDFo3_5dcFcydPp",
    "daftar_nama_orang" : "1BiisLUPvM_rmbh6okLQkcFlMlSRjKGy6",
    "istilah_islam"     : "1JhbRw7-J7B6xyJ9GdFSyea1tjEplEEBB",
    "sample_correct_2025": "1gT7UwwHE7on1-kfJfIx2GQS61U8zH-eW",
}

# Konfigurasi model IndoBERT
MODEL_CFG = {
    "max_length" : 128,
    "num_labels" : 2,
}

# Konfigurasi Jaro-Winkler
JW_CFG = {
    "threshold" : 0.99,
    "top_k"     : 5,
    "p_factor"  : 0.1,
}
