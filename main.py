"""
Data & Extraction Tools API
Connect to Lyzr via: Add Custom Tool -> Custom Tool (OpenAPI)
Spec URL after deploy: https://YOUR-APP.onrender.com/openapi.json
"""
import csv
import io
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SERVER_URL = os.environ.get("SERVER_URL", "https://YOUR-APP-NAME.onrender.com")

app = FastAPI(
    title="Data & Extraction Tools",
    version="1.0.0",
    description="Tools to extract structured information from text and CSV data, "
                "and to save, list, search and delete records.",
    servers=[{"url": SERVER_URL}],
)

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
PATTERNS = {
    "emails": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "urls": r"https?://[^\s<>\"')]+|www\.[^\s<>\"')]+",
    "phones": r"\+?\(?\d[\d\s().-]{6,}\d",
    "dates": (
        r"\b\d{4}-\d{1,2}-\d{1,2}\b"
        r"|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"
        rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS},?\s+\d{{4}}\b"
        rf"|\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
    ),
    "money": r"(?:[$€£₹¥]|USD|EUR|GBP|INR|Rs\.?)\s?\d+(?:,\d+)*(?:\.\d+)?",
    "percentages": r"-?\d+(?:\.\d+)?\s?%",
    "numbers": r"(?<![\w.])-?\d+(?:,\d+)*(?:\.\d+)?(?![\w])",
    "hashtags": r"(?<!\w)#\w+",
    "mentions": r"(?<![\w.])@\w+(?![\w.]*@|\.\w)",
}
ALL_TYPES = list(PATTERNS.keys())

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not now of off on once only or other our ours ourselves out
over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves also may might must shall
""".split())


def _unique(items: List[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def extract_entities_logic(text: str, types: List[str]) -> Dict[str, List[str]]:
    wanted = [t.lower() for t in types] if types else ALL_TYPES
    unknown = [t for t in wanted if t not in PATTERNS]
    if unknown:
        raise ValueError(f"Unknown types {unknown}. Allowed: {ALL_TYPES}")
    result: Dict[str, List[str]] = {}
    for t in wanted:
        found = re.findall(PATTERNS[t], text, flags=re.IGNORECASE)
        if t == "phones":
            found = [p for p in found if 7 <= len(re.sub(r"\D", "", p)) <= 15
                     and not re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", p.strip())]
        result[t] = _unique(found)
    return result


def extract_key_values_logic(text: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([^:=\n]{1,60}?)\s*[:=]\s*(.+?)\s*$", line)
        if m and not m.group(1).lower().startswith(("http", "https")):
            pairs[m.group(1).strip()] = m.group(2).strip()
    return pairs


def extract_keywords_logic(text: str, top_n: int) -> List[Dict[str, Any]]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
    counts = Counter(w for w in words if w not in STOPWORDS)
    return [{"keyword": w, "count": c} for w, c in counts.most_common(top_n)]


def _to_number(value: str):
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def parse_table_logic(csv_text: str, delimiter: str, max_rows: int) -> Dict[str, Any]:
    reader = csv.DictReader(io.StringIO(csv_text.strip()), delimiter=delimiter or ",")
    rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in reader]
    columns = [c.strip() for c in (reader.fieldnames or [])]
    numeric_stats = {}
    for col in columns:
        nums = [n for n in (_to_number(r.get(col)) for r in rows) if n is not None]
        if rows and len(nums) == len([r for r in rows if r.get(col)]) and nums:
            numeric_stats[col] = {
                "count": len(nums), "min": min(nums), "max": max(nums),
                "sum": round(sum(nums), 4), "mean": round(sum(nums) / len(nums), 4),
            }
    return {
        "columns": columns,
        "row_count": len(rows),
        "rows": rows[:max_rows],
        "rows_truncated": len(rows) > max_rows,
        "numeric_column_stats": numeric_stats,
    }


def text_stats_logic(text: str) -> Dict[str, Any]:
    words = re.findall(r"\b\w+\b", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
        "avg_words_per_sentence": round(len(words) / len(sentences), 2) if sentences else 0,
        "reading_time_minutes": round(len(words) / 200, 2),
    }


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class EntitiesRequest(BaseModel):
    text: str = Field(..., description="The text to extract information from")
    types: List[str] = Field(
        default_factory=list,
        description=f"Which entity types to extract. Leave empty for all. Allowed: {ALL_TYPES}",
    )


class TextRequest(BaseModel):
    text: str = Field(..., description="The text to analyse")


class KeywordsRequest(BaseModel):
    text: str = Field(..., description="The text to find keywords in")
    top_n: int = Field(10, ge=1, le=100, description="How many top keywords to return")


class TableRequest(BaseModel):
    csv_text: str = Field(..., description="CSV content as plain text, first row is the header")
    delimiter: str = Field(",", description="Column separator, e.g. ',' or ';' or '\\t'")
    max_rows: int = Field(50, ge=1, le=1000, description="Max rows to return in the response")


class SaveRecordRequest(BaseModel):
    collection: str = Field(..., description="Name of the group to save into, e.g. 'customers'")
    data: Dict[str, Any] = Field(..., description="The record fields as key-value pairs")


class SearchRequest(BaseModel):
    collection: str = Field(..., description="Name of the collection to search")
    query: str = Field(..., description="Text to look for (case-insensitive)")
    field: str = Field("", description="Optional: only search this field. Empty = all fields")


# ---------------------------------------------------------------------------
# Extraction endpoints
# ---------------------------------------------------------------------------
@app.get("/", operation_id="health_check", summary="Check that the tool service is running")
def health():
    return {"status": "ok", "tools": ALL_TYPES}


@app.post("/extract/entities", operation_id="extract_entities",
          summary="Extract emails, phone numbers, URLs, dates, money amounts, percentages, "
                  "numbers, hashtags and @mentions from text")
def extract_entities(req: EntitiesRequest):
    try:
        return {"entities": extract_entities_logic(req.text, req.types)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/extract/key-values", operation_id="extract_key_values",
          summary="Extract 'Label: value' or 'key = value' pairs from text such as forms, "
                  "invoices or profiles")
def extract_key_values(req: TextRequest):
    pairs = extract_key_values_logic(req.text)
    return {"count": len(pairs), "fields": pairs}


@app.post("/extract/keywords", operation_id="extract_keywords",
          summary="Find the most frequent meaningful keywords in text")
def extract_keywords(req: KeywordsRequest):
    return {"keywords": extract_keywords_logic(req.text, req.top_n)}


@app.post("/extract/table", operation_id="parse_csv_table",
          summary="Parse CSV text into rows and columns, with min/max/sum/mean for numeric columns")
def parse_table(req: TableRequest):
    delimiter = "\t" if req.delimiter in ("\\t", "tab") else req.delimiter
    try:
        return parse_table_logic(req.csv_text, delimiter, req.max_rows)
    except csv.Error as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")


@app.post("/text/stats", operation_id="text_statistics",
          summary="Count characters, words, sentences, paragraphs and estimate reading time")
def text_stats(req: TextRequest):
    return text_stats_logic(req.text)


# ---------------------------------------------------------------------------
# Data storage endpoints (in-memory: resets when the server restarts)
# ---------------------------------------------------------------------------
STORE: Dict[str, Dict[str, Dict[str, Any]]] = {}


@app.post("/records", operation_id="save_record",
          summary="Save a record (any key-value data) into a named collection")
def save_record(req: SaveRecordRequest):
    record_id = uuid.uuid4().hex[:8]
    record = {"id": record_id,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "data": req.data}
    STORE.setdefault(req.collection.lower(), {})[record_id] = record
    return {"saved": True, "collection": req.collection.lower(), "record": record}


@app.get("/records/{collection}", operation_id="list_records",
         summary="List all records saved in a collection")
def list_records(collection: str):
    records = list(STORE.get(collection.lower(), {}).values())
    return {"collection": collection.lower(), "count": len(records), "records": records}


@app.get("/collections", operation_id="list_collections",
         summary="List all collection names and how many records each has")
def list_collections():
    return {"collections": {name: len(recs) for name, recs in STORE.items()}}


@app.post("/records/search", operation_id="search_records",
          summary="Search records in a collection for matching text")
def search_records(req: SearchRequest):
    q = req.query.lower()
    matches = []
    for rec in STORE.get(req.collection.lower(), {}).values():
        fields = {req.field: rec["data"].get(req.field)} if req.field else rec["data"]
        if any(q in str(v).lower() for v in fields.values() if v is not None):
            matches.append(rec)
    return {"count": len(matches), "records": matches}


@app.delete("/records/{collection}/{record_id}", operation_id="delete_record",
            summary="Delete one record by its id")
def delete_record(collection: str, record_id: str):
    removed = STORE.get(collection.lower(), {}).pop(record_id, None)
    if not removed:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"deleted": True, "id": record_id}