"""
Convert a multi‑page FAQ PDF or DOCX into JSONL Q‑A pairs for LoRA fine‑tuning.
────
Each line in output.jsonl is:
{"text": "<s>[INST] <question> [/INST] <answer> </s>"}
"""
import argparse
import json
import re
from pathlib import Path

import pypdf
import docx


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() for page in reader.pages)
    elif path.suffix.lower() in {".docx", ".doc"}:
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type")


def to_qa_pairs(raw: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"Q[:\-]\s*(.*?)\nA[:\-]\s*(.*?)(?=\nQ[:\-]|$)", re.S | re.I)
    pairs = pattern.findall(raw)
    return [(q.strip(), a.strip()) for q, a in pairs if q and a]


def main(src: Path, out_path: Path):
    text = extract_text(src)
    pairs = to_qa_pairs(text)
    print(f"Found {len(pairs)} Q‑A pairs")

    with out_path.open("w", encoding="utf‑8") as f:
        for q, a in pairs:
            record = {
                "text": f"<s>[INST] {q} [/INST] {a} </s>"
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="FAQ PDF or DOCX")
    parser.add_argument("-o", "--out", type=Path, default=Path("faq.jsonl"))
    args = parser.parse_args()
    main(args.src, args.out)
