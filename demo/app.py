"""Streamlit demo: upload wav and show emotion probabilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SER (Dusha)", layout="centered")
st.title("Распознавание эмоций по речи (Dusha)")

ckpt_default = Path("checkpoints/ser/best.pt")
checkpoint = st.text_input("Путь к checkpoint (best.pt)", value=str(ckpt_default))
uploaded = st.file_uploader("Загрузите WAV", type=["wav"])

if uploaded and st.button("Распознать"):
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        st.error(f"Файл не найден: {ckpt_path}")
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)
        try:
            from src.inference.predict import predict_file

            label, probs = predict_file(ckpt_path, tmp_path)
            st.success(f"Класс: **{label}**")
            names = ("angry", "sad", "neutral", "positive")
            for n, p in zip(names, probs, strict=True):
                st.progress(float(p), text=f"{n}: {p:.2%}")
        finally:
            tmp_path.unlink(missing_ok=True)
