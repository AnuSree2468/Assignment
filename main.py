"""
AI-Powered Document Structuring & Data Extraction
-------------------------------------------------
Single-file Python implementation that converts an unstructured PDF into a
structured Excel (Key/Value/Comments) format using a hybrid approach:
  - heuristic rule-based key:value detection (colon, tab, dash, inline)
  - spaCy NER to suggest semantic keys (PERSON/DATE/ORG/GPE/WORK_OF_ART etc.)
  - semantic similarity (sentence-transformers) to map free-text phrases to
    likely keys using a configurable "schema" / canonical key list
  - contextual "Comments" captured as surrounding sentence(s)

Notes:
  - This is a practical, extendable baseline. Edge-cases (tables, scanned PDFs,
    complex layouts) may need OCR (pytesseract) or layout-aware parsers.
  - Save as `ai_structurer.py`. Example usage:
      python ai_structurer.py --input Data_Input.pdf --output Structured_Output.xlsx

Requirements (pip):
  pip install pdfplumber pandas openpyxl spacy sentence-transformers tqdm
  python -m spacy download en_core_web_sm

"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pdfplumber
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ---------------------- Configuration ----------------------
# Canonical keys that the system will try to detect / map to. You can extend
# or replace this list with a JSON config to adapt to different domains.
CANONICAL_KEYS = [
    "First Name", "Last Name", "Date of Birth", "Birth City", "Birth State",
    "Age", "Blood Group", "Nationality",
    "Joining Date of first professional role", "Designation of first professional role",
    "Salary of first professional role", "Salary currency of first professional role",
    "Current Organization", "Current Joining Date", "Current Designation",
    "Current Salary", "Current Salary Currency",
    "Previous Organization", "Previous Joining Date", "Previous end year",
    "Previous Starting Designation",
    "High School", "12th standard pass out year", "12th overall board score",
    "Undergraduate degree", "Undergraduate college", "Undergraduate year", "Undergraduate CGPA",
    "Graduation degree", "Graduation college", "Graduation year", "Graduation CGPA",
    "Certifications 1", "Certifications 2", "Certifications 3", "Certifications 4",
    "Technical Proficiency"
]

# Threshold for semantic similarity mapping (0.0 - 1.0). Tweak to be more/less strict.
SIMILARITY_THRESHOLD = 0.55

# ---------------------- Utilities ----------------------

nlp = spacy.load("en_core_web_sm")
SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
can_embeddings = SBERT_MODEL.encode(CANONICAL_KEYS, convert_to_tensor=True)

KV_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 \-/&()'.]{1,80}?)\s*[:\t\-\u2014]\s*(.+)$")
# Matches patterns like "Key: Value" or "Key - Value" or "Key\tValue"

DATE_PATTERN = re.compile(r"(\d{1,2}[\-/][A-Za-z0-9]{1,4}[\-/]\d{2,4})|([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})")
CURRENCY_PATTERN = re.compile(r"\b(INR|USD|EUR|GBP|₹|\$)\b", re.IGNORECASE)
PERCENT_PATTERN = re.compile(r"\b\d{1,3}\.\d+%|\d{1,3}%\b")

# ---------------------- Core functions ----------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts textual content from PDF using pdfplumber.
    Note: for scanned PDFs add OCR (pytesseract) pipeline.
    """
    text_parts: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            # extract_text (pdfplumber) returns reasonable flow for many PDFs
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    combined = "\n\n".join(text_parts)
    return combined


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy to preserve sentence boundaries."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def find_explicit_key_values(lines: List[str]) -> List[Tuple[str, str, int]]:
    """Detect explicit key:value lines with pattern matching. Returns list of (key, value, line_index)."""
    kvs = []
    for i, ln in enumerate(lines):
        m = KV_PATTERN.match(ln)
        if m:
            key = m.group(1).strip()
            value = m.group(2).strip()
            kvs.append((key, value, i))
    return kvs


def ner_based_candidates(sentences: List[str]) -> List[Tuple[str, str, int]]:
    """Use spaCy NER to extract entities and craft candidate key/values. Returns (key, value, sentence_index)."""
    candidates = []
    for i, sent in enumerate(sentences):
        doc = nlp(sent)
        # Collect contiguous entity types as simple heuristics
        for ent in doc.ents:
            if ent.label_ in ("PERSON",):
                # try split person into first/last
                name_parts = ent.text.split()
                if len(name_parts) >= 1:
                    candidates.append(("First Name", name_parts[0], i))
                if len(name_parts) >= 2:
                    candidates.append(("Last Name", " ".join(name_parts[1:]), i))
            elif ent.label_ in ("DATE",):
                candidates.append(("Date", ent.text, i))
            elif ent.label_ in ("ORG",):
                candidates.append(("Organization", ent.text, i))
            elif ent.label_ in ("GPE", "LOC"):
                candidates.append(("Location", ent.text, i))
            elif ent.label_ in ("MONEY",):
                candidates.append(("Monetary Amount", ent.text, i))
            elif ent.label_ in ("PERCENT",):
                candidates.append(("Percentage", ent.text, i))
    return candidates


def semantic_map_to_canonical(key_phrase: str) -> Tuple[str, float]:
    """Map a freeform key phrase to the closest canonical key via SBERT similarity.
    Returns (best_key, score). If score below threshold returns (None, score).
    """
    if not key_phrase or len(key_phrase.strip()) < 2:
        return (None, 0.0)
    emb = SBERT_MODEL.encode(key_phrase, convert_to_tensor=True)
    sim_scores = util.cos_sim(emb, can_embeddings)[0]
    best_idx = int(sim_scores.argmax())
    best_score = float(sim_scores[best_idx])
    best_key = CANONICAL_KEYS[best_idx]
    if best_score >= SIMILARITY_THRESHOLD:
        return best_key, best_score
    return None, best_score


def normalize_date(value: str) -> str:
    """Attempt to normalize common date strings into ISO-ish formats or leave raw.
    Lightweight: for production, use dateparser or parsedatetime.
    """
    # crude month name mapping
    try:
        # try to parse formats like 'March 15, 1989' or '15-Mar-89'
        import dateutil.parser as dparser
        parsed = dparser.parse(value, fuzzy=True)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return value


def build_structured_rows(text: str) -> List[Dict[str, Any]]:
    """Main orchestration: produce list of dict rows with Key, Value, Comments."""
    sentences = split_sentences(text)
    lines = [ln for part in text.splitlines() for ln in (part.strip(),) if ln.strip()]

    explicit_kv = find_explicit_key_values(lines)

    candidate_entities = ner_based_candidates(sentences)

    # Start with explicit k:v pairs
    rows: List[Dict[str, Any]] = []
    used_line_indices = set()

    for key, val, li in explicit_kv:
        mapped_key, score = semantic_map_to_canonical(key)
        final_key = mapped_key or key
        # find a sentence context nearby for comments
        context = extract_context(sentences, li)
        rows.append({"Key": final_key, "Value": val, "Comments": context})
        used_line_indices.add(li)

    # Add NER-derived candidates if they don't conflict with the explicit list
    for cand_key, cand_val, si in candidate_entities:
        # map candidate key to canonical if possible
        mapped_key, score = semantic_map_to_canonical(cand_key)
        final_key = mapped_key or cand_key
        # avoid duplicates: check if same value already in rows
        if any(r for r in rows if str(r["Value"]).strip().lower() == cand_val.strip().lower()):
            continue
        context = extract_context(sentences, si)
        rows.append({"Key": final_key, "Value": cand_val, "Comments": context})

    # Heuristic post-processing: merge organization-type keys
    rows = postprocess_rows(rows)

    # Ensure order and numbering
    final_rows = []
    for i, r in enumerate(rows, start=1):
        final_rows.append({"#": i, "Key": r["Key"], "Value": r["Value"], "Comments": r.get("Comments", "")})
    return final_rows


def extract_context(sentences: List[str], index: int, window: int = 1) -> str:
    """Return surrounding sentences as context (comments)."""
    start = max(0, index - window)
    end = min(len(sentences), index + window + 1)
    return " ".join(sentences[start:end])


def postprocess_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple post-processing: map generic canonical keys into more specific ones
    and collapse duplicates.
    """
    # collapse duplicates by Key preferring longer comments
    seen = {}
    out = []
    for r in rows:
        k = r["Key"]
        if k in seen:
            # prefer non-empty value or longer comments
            existing = seen[k]
            if (not existing["Value"] and r["Value"]) or (len(r.get("Comments",""))>len(existing.get("Comments",""))):
                seen[k] = r
        else:
            seen[k] = r
    for v in seen.values():
        out.append(v)
    # sort deterministically by Key for stable outputs
    out.sort(key=lambda x: str(x.get("Key","")))
    return out


def write_to_excel(rows: List[Dict[str, Any]], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    # Ensure columns order for compatibility with expected output
    col_order = ["#", "Key", "Value", "Comments"]
    # If # is missing because rows weren't numbered, add it
    if "#" not in df.columns:
        df.insert(0, "#", range(1, len(df) + 1))
    df = df[[c for c in col_order if c in df.columns]]
    df.to_excel(str(out_path), index=False)


# ---------------------- CLI & Main ----------------------

def main():
    parser = argparse.ArgumentParser(description="AI-backed PDF -> Structured Excel extractor")
    parser.add_argument("--input", "-i", required=True, help="Input PDF path")
    parser.add_argument("--output", "-o", required=True, help="Output Excel path (.xlsx)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    raw_text = extract_text_from_pdf(in_path)
    rows = build_structured_rows(raw_text)
    write_to_excel(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
