"""
fetch_external_resources.py
──────────────────────────────────────────────────────────────────────────────
Fetch and normalize additional ophthalmic knowledge sources into a clean
markdown corpus that is backward-compatible with scripts/chunk_data.py.

Sources covered:
  1) Hugging Face: QIAIUNCC/EYE-lit-complete
  2) Hugging Face: MedRAG/textbooks (ophthalmology/anatomy-relevant chunks)
  3) AAO Preferred Practice Patterns (PPP)
  4) StatPearls chapters via NCBI Bookshelf
  5) Merck Manual (Professional, eye-disorders)

Outputs:
  - data/processed/external_ophthalmic_resources_clean.md
  - data/raw/external_sources/external_fetch_manifest.json

Usage:
  python scripts/fetch_external_resources.py
    python scripts/fetch_external_resources.py --max-eye-lit 15000 --max-medrag 10000
    python scripts/fetch_external_resources.py --max-aao 80 --aao-pdf-pages 0
  python scripts/fetch_external_resources.py --sources eye-lit,medrag,aao,statpearls,merck
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen

from tqdm import tqdm

import fitz


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MD_DEFAULT = ROOT / "data" / "processed" / "external_ophthalmic_resources_clean.md"
OUTPUT_MANIFEST_DEFAULT = ROOT / "data" / "raw" / "external_sources" / "external_fetch_manifest.json"

USER_AGENT = "OphthalmicRAG-KBFetcher/1.0"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
HF_DATASET_API = "https://huggingface.co/api/datasets"
AAO_SITEMAP_URL = "https://www.aao.org/sitemap.xml"
MERCK_INDEX_URL = "https://www.merckmanuals.com/professional/eye-disorders"
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "").strip()
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


OPHTHALMIC_QUERY_TERMS = [
    "conjunctivitis",
    "keratitis",
    "bacterial keratitis",
    "fungal keratitis",
    "uveitis",
    "scleritis",
    "episcleritis",
    "glaucoma",
    "angle closure glaucoma",
    "open angle glaucoma",
    "retinal detachment",
    "diabetic retinopathy",
    "age-related macular degeneration",
    "macular edema",
    "retinal vein occlusion",
    "retinal artery occlusion",
    "optic neuritis",
    "cataract",
    "dry eye",
    "blepharitis",
    "dacryocystitis",
    "retinopathy of prematurity",
    "herpes zoster ophthalmicus",
    "corneal ulcer",
    "leukocoria",
    "corneal opacity",
    "cherry red spot",
    "anisocoria",
    "retinoblastoma",
    "band keratopathy",
    "corneal dystrophy",
    "hyphema",
    "central retinal artery occlusion",
    "anterior ischemic optic neuropathy",
    "ocular trauma",
    "chemical eye injury",
    "pterygium",
    "pinguecula",
    "chalazion",
    "stye",
    "subconjunctival hemorrhage",
    "ocular hypertension",
]

OPHTHALMIC_KEYWORDS = {
    "eye", "ocular", "ophthalm", "retina", "retinal", "cornea", "corneal",
    "iris", "pupil", "lens", "conjunctiva", "sclera", "macula", "fovea",
    "uveitis", "glaucoma", "keratitis", "cataract", "fundus", "optic",
    "keratoplasty", "tonometry", "fundoscopy", "slit lamp", "slit-lamp",
    "visual field", "oct", "optical coherence tomography", "gonioscopy",
    "intravitreal", "trabeculectomy",
}

SOURCE_AUTHORITY_RANK = {
    "AAO Preferred Practice Patterns": 1,
    "EyeWiki (AAO)": 1,
    "StatPearls (NCBI Bookshelf)": 2,
    "PubMed": 3,
    "PMC Open Access": 2,
    "Wikipedia (Ophthalmology)": 3,
    "Hugging Face - MedRAG Textbooks": 4,
    "Hugging Face - EYE-lit Complete": 5,
    "Merck Manual Professional": 2,
}

MEDRAG_PRIORITY_HINTS = [
    "anatomy_gray",
    "internalmed_harrison",
    "neurology_adams",
    "pathology_robbins",
    "physiology_levy",
    "pharmacology_katzung",
    "first_aid_step1",
    "first_aid_step2",
    "histology_ross",
    "cell_biology_alberts",
    "biochemistry_lippincott",
    "immunology_janeway",
]


@dataclass
class ResourceRecord:
    source_group: str
    source_name: str
    title: str
    url: str
    content: str
    metadata: Dict[str, object] = field(default_factory=dict)


def _http_get(
    url: str,
    *,
    params: Optional[Dict[str, object]] = None,
    timeout: int = 45,
    retries: int = 3,
    retry_sleep: float = 1.0,
) -> Tuple[bytes, Dict[str, str]]:
    """HTTP GET with retry and a stable user-agent."""
    if params:
        qs = urlencode(params, doseq=True)
        url = f"{url}?{qs}"

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return payload, headers
        except HTTPError as exc:
            last_error = exc
            if attempt < retries:
                wait_s = retry_sleep * (2 ** (attempt - 1))
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    if retry_after:
                        try:
                            wait_s = max(wait_s, float(retry_after))
                        except ValueError:
                            pass
                    wait_s = max(wait_s, 6.0 * attempt)
                time.sleep(wait_s)
            continue
        except (URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * (2 ** (attempt - 1)))
    raise RuntimeError(f"GET failed after {retries} retries: {url} | {last_error}")


def _http_get_text(url: str, **kwargs) -> Tuple[str, Dict[str, str]]:
    data, headers = _http_get(url, **kwargs)
    return data.decode("utf-8", errors="replace"), headers


def _get_json(url: str, **kwargs) -> Dict[str, Any]:
    text, _ = _http_get_text(url, **kwargs)
    return json.loads(text)


def _clean_text(text: str) -> str:
    text = unescape(text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def _normalize_for_hash(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _content_hash(text: str) -> str:
    normalized = _normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _strip_html_tags(html: str) -> str:
    html = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", html)
    html = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", html)
    html = re.sub(r"(?is)<(br|/p|/div|/li|/section|/article|/h1|/h2|/h3|/h4|/h5|/h6)>", "\n", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = unescape(html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    html = re.sub(r"[ ]{2,}", " ", html)
    return html.strip()


def _extract_title(html: str, fallback: str = "Untitled") -> str:
    match = re.search(r"(?is)<title>(.*?)</title>", html)
    if not match:
        return fallback
    title = _clean_text(_strip_html_tags(match.group(1)))
    # Strip common site suffixes
    for suffix in [" - PMC", " - PubMed Central", " - NCBI", " - Wikipedia"]:
        if title.endswith(suffix):
            title = title[: -len(suffix)].strip()
    title = re.sub(r"\s+-\s+American Academy of Ophthalmology$", "", title, flags=re.I)
    title = re.sub(r"\s+-\s+StatPearls\s+-\s+NCBI Bookshelf$", "", title, flags=re.I)
    title = re.sub(r"\s+-\s+Merck Manual Professional Edition$", "", title, flags=re.I)
    return title or fallback


def _extract_region(html: str) -> str:
    """Try to capture article/main region, fallback to full HTML body."""
    # 0. PMC specific containers (most precise for PubMed Central)
    for pmc_cls in ["pmc-layout__content", "pmc-article-section"]:
        match = re.search(f'(?is)class=["\'][^"\']*{pmc_cls}[^"\']*["\'][^>]*>(.*?)(?=<div[^>]*class=["\']pmc-sidenav)', html)
        if match:
            return match.group(1)

    # 1. Prefer <main> element — standard semantic HTML
    main_match = re.search(r"(?is)<main[^>]*>(.*?)</main>", html)
    if main_match:
        return main_match.group(1)

    # 2. id="maincontent" (NCBI Bookshelf, StatPearls, Merck)
    #    IMPORTANT: find() lands on the *attribute*, not the content.
    #    Advance past the closing ">" of the opening tag first.
    start = html.lower().find('id="maincontent"')
    if start != -1:
        tag_close = html.find(">", start)
        if tag_close != -1:
            snippet = html[tag_close + 1:]
            stop_patterns = [
                '<div class="bottom"',
                '<footer',
                '<div id="footer"',
                '<section class="disclaimer"',
                '<section id="Bib1"',
                'class="ref-list"',
                'id="references"',
            ]
            stop_at = len(snippet)
            snippet_low = snippet.lower()
            for pat in stop_patterns:
                idx = snippet_low.find(pat)
                if idx != -1:
                    stop_at = min(stop_at, idx)
            return snippet[:stop_at]

    # 3. <article> element (NCBI Bookshelf alternative structure)
    article_match = re.search(r"(?is)<article[^>]*>(.*?)</article>", html)
    if article_match and len(article_match.group(1)) > 500:
        return article_match.group(1)

    # 4. class="maincontent" (single quotes or class variant)
    class_match = re.search(r'(?is)class=["\']maincontent["\'][^>]*>(.*?)(?=<(?:footer|div\s+id=["\']footer))', html)
    if class_match and len(class_match.group(1)) > 500:
        return class_match.group(1)

    # 5. Full body fallback
    body_match = re.search(r"(?is)<body[^>]*>(.*?)</body>", html)
    if body_match:
        return body_match.group(1)
    return html


def _truncate(text: str, max_chars: int = 24000) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "\n"


def _title_from_text(text: str, fallback: str) -> str:
    text = _clean_text(text)
    if not text:
        return fallback
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    sentence = sentence.strip()
    if len(sentence) > 140:
        sentence = sentence[:137].rstrip() + "..."
    return sentence or fallback


def _batch(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _with_ncbi_api_key(params: Dict[str, object]) -> Dict[str, object]:
    if NCBI_API_KEY:
        return {**params, "api_key": NCBI_API_KEY}
    return params


def _extract_pdf_text(pdf_bytes: bytes, max_pages: int) -> str:
    if not pdf_bytes:
        return ""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts: List[str] = []
    pages = doc.page_count if max_pages <= 0 else min(max_pages, doc.page_count)
    for i in range(pages):
        page = doc.load_page(i)
        chunk = page.get_text("text")
        if isinstance(chunk, str):
            parts.append(chunk)
        elif isinstance(chunk, list):
            parts.append(" ".join(str(x) for x in chunk))
        else:
            parts.append(str(chunk))
    return _clean_text("\n".join(parts))


def fetch_eye_lit(max_docs: int, sleep_s: float) -> List[ResourceRecord]:
    print(f"[EYE-lit] Fetching up to {max_docs} rows using huggingface datasets...")
    if max_docs <= 0:
        return []

    try:
        import datasets
        # We use streaming=True to avoid downloading the massive dataset at once
        ds = datasets.load_dataset("QIAIUNCC/EYE-lit-complete", split="train", streaming=True)
    except Exception as e:
        print(f"[EYE-lit] Failed to load dataset: {e}")
        return []

    records: List[ResourceRecord] = []
    pbar = tqdm(total=max_docs, desc="[EYE-lit]", unit="row")
    
    offset = 0
    for row in ds:
        text = _clean_text(str(row.get("page_content", "")))
        if len(text) < 300:
            offset += 1
            continue
            
        ridx = offset
        title = _title_from_text(text, fallback=f"EYE-lit entry {ridx}")
        records.append(
            ResourceRecord(
                source_group="Hugging Face - EYE-lit Complete",
                source_name="QIAIUNCC/EYE-lit-complete",
                title=title,
                url="https://huggingface.co/datasets/QIAIUNCC/EYE-lit-complete",
                content=_truncate(text),
                metadata={"row_idx": ridx},
            )
        )
        pbar.update(1)
        offset += 1
        
        if len(records) >= max_docs:
            break

    pbar.close()
    print(f"[EYE-lit] Collected {len(records)} usable records.")
    return records


def _hf_dataset_siblings(repo: str) -> List[str]:
    payload = _get_json(f"{HF_DATASET_API}/{repo}", timeout=45, retries=3)
    siblings_obj = payload.get("siblings", [])
    siblings = siblings_obj if isinstance(siblings_obj, list) else []
    out: List[str] = []
    for s in siblings:
        if isinstance(s, dict):
            name = s.get("rfilename")
            if isinstance(name, str):
                out.append(name)
    return out


def _medrag_file_rank(rel_path: str) -> Tuple[int, str]:
    name = Path(rel_path).stem.lower()
    for idx, hint in enumerate(MEDRAG_PRIORITY_HINTS):
        if hint in name:
            return (idx, name)
    return (len(MEDRAG_PRIORITY_HINTS) + 1, name)


def fetch_medrag(max_docs: int, sleep_s: float, include_general: bool = True) -> List[ResourceRecord]:
    print(f"[MedRAG] Fetching up to {max_docs} chunks from MedRAG/textbooks...")
    if max_docs <= 0:
        return []

    siblings = _hf_dataset_siblings("MedRAG/textbooks")
    candidate_files = sorted(
        [f for f in siblings if f.startswith("chunk/") and f.endswith(".jsonl")],
        key=_medrag_file_rank,
    )

    records: List[ResourceRecord] = []
    fallback_records: List[ResourceRecord] = []
    pbar = tqdm(total=max_docs, desc="[MedRAG]", unit="chunk")
    for rel_path in candidate_files:
        url = (
            "https://huggingface.co/datasets/MedRAG/textbooks/resolve/main/"
            + quote(rel_path, safe="/")
        )
        text, _ = _http_get_text(url, timeout=60, retries=3)
        for line_idx, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            body = _clean_text(str(obj.get("content") or obj.get("contents") or ""))
            if len(body) < 220:
                continue

            raw_title = str(obj.get("title") or Path(rel_path).stem)
            chunk_id = str(obj.get("id") or f"{Path(rel_path).stem}_{line_idx}")
            title = f"{raw_title} | {chunk_id}"
            is_relevant = _is_ophthalmic_text(raw_title, body)

            rec = ResourceRecord(
                source_group="Hugging Face - MedRAG Textbooks",
                source_name="MedRAG/textbooks",
                title=title,
                url=url,
                content=_truncate(body),
                metadata={
                    "file": rel_path,
                    "chunk_id": chunk_id,
                    "relevance": "ophthalmic" if is_relevant else "general",
                },
            )

            if is_relevant:
                records.append(rec)
                pbar.update(1)
            elif include_general:
                fallback_records.append(rec)

            if len(records) >= max_docs:
                break

        if len(records) >= max_docs:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)

    if include_general and len(records) < max_docs and fallback_records:
        need = max_docs - len(records)
        records.extend(fallback_records[:need])

    print(
        f"[MedRAG] Collected {len(records)} usable records from {len(candidate_files)} files "
        f"(fallback pool: {len(fallback_records)})."
    )
    return records


def _extract_aao_ppp_urls() -> List[str]:
    sitemap, _ = _http_get_text(AAO_SITEMAP_URL, timeout=90, retries=3)
    # Keep only actual PPP guideline landing pages.
    guideline_urls = re.findall(
        r"https://www\.aao\.org/education/preferred-practice-pattern/[a-z0-9\-]+",
        sitemap,
        flags=re.I,
    )
    return sorted(set(guideline_urls))


def _is_aao_literature_search_asset(link: str) -> bool:
    low = link.lower()
    return any(
        token in low
        for token in (
            "literature-search",
            "lit-search",
            "appendix",
            "-search-pdf",
        )
    )


def _clean_aao_ppp_text(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""

    lines = [ln.strip() for ln in cleaned.splitlines()]

    drop_line_re = [
        re.compile(r"^P\d+\s*$", re.I),
        re.compile(r"^[A-Za-z][A-Za-z\s\-&,']+\s+PPP\s+P\d+\s*$", re.I),
        re.compile(r"^©\s*\d{4}\s+American Academy of Ophthalmology", re.I),
        re.compile(r"^All rights reserved\s*$", re.I),
        re.compile(r"^AMERICAN ACADEMY OF OPHTHALMOLOGY.+registered trademarks", re.I),
        re.compile(r"^Preferred Practice Pattern®?\s*$", re.I),
    ]
    drop_exact = {
        "secretary for quality of care",
        "academy staff",
        "medical editor:",
        "approved by:",
        "board of trustees",
        "skip to main content",
    }

    filtered: List[str] = []
    for ln in lines:
        low = ln.lower()
        if not ln:
            filtered.append("")
            continue
        if low in drop_exact:
            continue
        if any(rx.search(ln) for rx in drop_line_re):
            continue
        filtered.append(ln)

    # Trim common AAO front matter to the first clinically meaningful heading.
    anchor_patterns = [
        r"highlighted findings and recommendations",
        r"major recommendations",
        r"patient population",
        r"introduction",
        r"disease definition",
        r"care process",
        r"diagnosis",
        r"management",
        r"follow[- ]?up",
        r"patient education",
    ]
    anchor_idx = -1
    for i, ln in enumerate(filtered):
        low = ln.lower()
        if any(re.search(pat, low) for pat in anchor_patterns):
            anchor_idx = i
            break
    if anchor_idx >= 0 and anchor_idx < 350:
        filtered = filtered[anchor_idx:]

    # Trim noisy tails that are low value for retrieval.
    tail_markers = [
        "literature searches for this ppp",
        "appendix",
        "references",
    ]
    for i, ln in enumerate(filtered):
        if any(marker in ln.lower() for marker in tail_markers):
            # Avoid over-trimming very short documents.
            if i > 120:
                filtered = filtered[:i]
            break

    # Collapse excessive blank lines.
    out: List[str] = []
    blank_run = 0
    for ln in filtered:
        if not ln:
            blank_run += 1
            if blank_run <= 1:
                out.append("")
        else:
            blank_run = 0
            out.append(ln)

    return "\n".join(out).strip()


def _is_plausible_aao_html_content(text: str) -> bool:
    low = (text or "").lower()
    if len(low) < 1200:
        return False

    nav_tokens = [
        "my dashboard",
        "my education",
        "find an ophthalmologist",
        "meetings",
        "program highlights",
    ]
    if sum(tok in low for tok in nav_tokens) >= 2:
        return False

    clinical_tokens = [
        "ophthalm",
        "retina",
        "cornea",
        "glaucoma",
        "cataract",
        "keratitis",
        "uveitis",
        "diagnosis",
        "treatment",
    ]
    hits = sum(tok in low for tok in clinical_tokens)
    return hits >= 4


def _extract_aao_asset_links(page_html: str, page_url: str) -> List[str]:
    hrefs = re.findall(r"href=[\"']([^\"']+)[\"']", page_html, flags=re.I)
    out: List[str] = []
    for href in hrefs:
        low = href.lower()
        if "pdf" not in low:
            continue

        # AAO CMS historically uses /Assets/.../xyz-pdf, but may also expose
        # direct CDN/S3 style links or query-string PDF URLs.
        if not (
            "/assets/" in low
            or low.endswith(".pdf")
            or ".pdf?" in low
            or "cloudfront" in low
            or "amazonaws.com" in low
            or "aao.org" in low
        ):
            continue

        out.append(urljoin(page_url, href))

    unique = sorted(set(out))
    primary = [link for link in unique if not _is_aao_literature_search_asset(link)]

    def rank_key(link: str) -> Tuple[int, int]:
        low = link.lower()
        return (
            0 if "-ppp" in low else 1,
            len(low),
        )

    return sorted(primary if primary else unique, key=rank_key)


def fetch_aao_ppp(max_docs: int, pdf_pages: int, sleep_s: float) -> List[ResourceRecord]:
    print(f"[AAO PPP] Fetching up to {max_docs} guideline PDFs/pages...")
    if max_docs <= 0:
        return []

    ppp_urls = _extract_aao_ppp_urls()
    records: List[ResourceRecord] = []

    pbar = tqdm(ppp_urls, desc="[AAO PPP]", unit="page")
    for page_url in pbar:
        if len(records) >= max_docs:
            break

        try:
            page_html, _ = _http_get_text(page_url, timeout=60, retries=3)
        except Exception as exc:
            print(f"  [AAO] skip {page_url} ({exc})")
            continue

        page_title = _extract_title(page_html, fallback="AAO PPP")
        page_text = _clean_aao_ppp_text(_strip_html_tags(_extract_region(page_html)))
        added_from_assets = 0

        for asset_url in _extract_aao_asset_links(page_html, page_url):
            if len(records) >= max_docs:
                break
            if _is_aao_literature_search_asset(asset_url):
                continue
            try:
                blob, headers = _http_get(asset_url, timeout=90, retries=2)
                ctype = headers.get("content-type", "").lower()
                if "pdf" in ctype or blob.startswith(b"%PDF"):
                    pdf_text = _clean_aao_ppp_text(_extract_pdf_text(blob, max_pages=pdf_pages))
                    if len(pdf_text) < 700:
                        continue

                    asset_title = page_title
                    if added_from_assets > 0:
                        asset_title = f"{page_title} (Supplement {added_from_assets + 1})"

                    records.append(
                        ResourceRecord(
                            source_group="AAO Preferred Practice Patterns",
                            source_name="American Academy of Ophthalmology",
                            title=asset_title,
                            url=asset_url,
                            content=_truncate(pdf_text),
                            metadata={"landing_page": page_url, "pdf_url": asset_url},
                        )
                    )
                    added_from_assets += 1
            except Exception:
                continue

        if len(records) >= max_docs:
            break

        # Fallback to HTML only if the extracted page text looks clinically meaningful.
        if added_from_assets == 0 and _is_plausible_aao_html_content(page_text):
            records.append(
                ResourceRecord(
                    source_group="AAO Preferred Practice Patterns",
                    source_name="American Academy of Ophthalmology",
                    title=page_title,
                    url=page_url,
                    content=_truncate(page_text),
                    metadata={"landing_page": page_url, "format": "html"},
                )
            )

        if sleep_s > 0:
            time.sleep(sleep_s)

    print(f"[AAO PPP] Collected {len(records)} guideline records.")
    return records


def _parse_chapter_title_from_bookinfo(bookinfo: str) -> str:
    if not bookinfo:
        return "StatPearls chapter"
    m = re.search(
        r"<Parent[^>]*type=\"chapter\"[^>]*>\s*<Title>(.*?)</Title>",
        bookinfo,
        flags=re.I | re.S,
    )
    if m:
        return _clean_text(_strip_html_tags(m.group(1)))
    return "StatPearls chapter"


def _is_ophthalmic_text(title: str, text: str) -> bool:
    probe = (title + "\n" + text[:2500]).lower()
    return any(k in probe for k in OPHTHALMIC_KEYWORDS)


def fetch_statpearls(max_chapters: int, sleep_s: float) -> List[ResourceRecord]:
    print(f"[StatPearls] Fetching up to {max_chapters} ophthalmic chapters...")
    if max_chapters <= 0:
        return []

    section_to_terms: Dict[str, Set[str]] = defaultdict(set)
    pbar_terms = tqdm(OPHTHALMIC_QUERY_TERMS, desc="[StatPearls] Querying Terms", unit="term")

    for term in pbar_terms:
        try:
            payload = _get_json(
                NCBI_ESEARCH_URL,
                params=_with_ncbi_api_key({
                    "db": "books",
                    "term": f"NBK430685[BookAccessionID] AND ({term})",
                    "retmax": 120,
                    "retmode": "json",
                }),
                timeout=45,
                retries=3,
            )
        except Exception as exc:
            print(f"  [StatPearls] search skip for term '{term}' ({exc})")
            continue

        esearch = payload.get("esearchresult", {})
        ids = esearch.get("idlist", []) if isinstance(esearch, dict) else []
        for sid in ids:
            section_to_terms[str(sid)].add(term)

        if sleep_s > 0:
            time.sleep(sleep_s)

    if not section_to_terms:
        print("[StatPearls] No section IDs discovered.")
        return []

    chapter_titles: Dict[str, str] = {}
    chapter_terms: Dict[str, Set[str]] = defaultdict(set)
    section_ids = sorted(section_to_terms.keys())

    for batch_ids in _batch(section_ids, 200):
        payload = _get_json(
            NCBI_ESUMMARY_URL,
            params=_with_ncbi_api_key({"db": "books", "id": ",".join(batch_ids), "retmode": "json"}),
            timeout=45,
            retries=3,
        )
        result_obj = payload.get("result", {})
        result = result_obj if isinstance(result_obj, dict) else {}
        uids = result.get("uids", [])
        if not isinstance(uids, list):
            continue
        for sid in uids:
            rec = result.get(sid, {})
            if not isinstance(rec, dict):
                continue
            if rec.get("bookaccessionid") != "NBK430685":
                continue
            chapter_acc = str(rec.get("chapteraccessionid") or "").strip()
            if not chapter_acc:
                continue

            chapter_titles.setdefault(
                chapter_acc,
                _parse_chapter_title_from_bookinfo(str(rec.get("bookinfo", ""))),
            )
            chapter_terms[chapter_acc].update(section_to_terms.get(str(sid), set()))

    ranked_chapters = sorted(chapter_terms.items(), key=lambda kv: len(kv[1]), reverse=True)

    records: List[ResourceRecord] = []
    pbar_chapters = tqdm(ranked_chapters, desc="[StatPearls] Downloading Chapters", unit="chap")
    for chapter_acc, term_set in pbar_chapters:
        if len(records) >= max_chapters:
            break

        chapter_url = f"https://www.ncbi.nlm.nih.gov/books/{chapter_acc}/"
        try:
            html, _ = _http_get_text(chapter_url, timeout=60, retries=3)
        except Exception as exc:
            print(f"  [StatPearls] skip {chapter_acc} ({exc})")
            continue

        title = _extract_title(html, fallback=chapter_titles.get(chapter_acc, "StatPearls chapter"))
        text = _clean_text(_strip_html_tags(_extract_region(html)))

        # Strip NCBI Bookshelf navigation boilerplate that appears before the
        # actual clinical content (header, breadcrumbs, author block, etc.).
        # Anchor on common section headings that open the medical narrative.
        _sp_anchors = [
            "continuing education activity",
            "introduction\n",
            "etiology\n",
            "epidemiology\n",
            "pathophysiology\n",
            "history and physical\n",
            "evaluation\n",
            "treatment / management",
            "differential diagnosis\n",
        ]
        text_lower = text.lower()
        anchor_pos = len(text)  # default: keep all
        for anchor in _sp_anchors:
            idx = text_lower.find(anchor)
            if idx != -1 and idx < anchor_pos:
                anchor_pos = idx
        # Only trim if anchor is found within first 3000 chars (navigation zone)
        if anchor_pos < 3000:
            text = text[anchor_pos:]

        if len(text) < 700:
            continue
        if not _is_ophthalmic_text(title, text):
            continue

        terms = sorted(term_set)
        records.append(
            ResourceRecord(
                source_group="StatPearls (NCBI Bookshelf)",
                source_name="StatPearls [Internet]",
                title=title,
                url=chapter_url,
                content=_truncate(text),
                metadata={"query_terms": terms[:10]},
            )
        )

        if sleep_s > 0:
            time.sleep(sleep_s)

    print(f"[StatPearls] Collected {len(records)} chapter records.")
    return records


def _extract_merck_links(index_html: str) -> List[str]:
    hrefs = re.findall(r"href=[\"']([^\"']+)[\"']", index_html, flags=re.I)
    links: List[str] = []
    for href in hrefs:
        if "/professional/eye-disorders/" not in href:
            continue
        if href.startswith("/"):
            href = "https://www.merckmanuals.com" + href
        if not href.startswith("https://www.merckmanuals.com"):
            continue
        href = href.split("#", 1)[0].split("?", 1)[0]

        # Keep article-like pages and skip top-level section landing pages.
        rel = href.split("/professional/eye-disorders/", 1)[-1].strip("/")
        if rel:
            parts = [p for p in rel.split("/") if p]
            if len(parts) < 2:
                continue

        links.append(href)

    dedup = sorted(set(links))
    return dedup


def _collapse_repeated_sentence_blocks(text: str) -> str:
    """Collapse adjacent exact sentence repeats caused by duplicated page blocks."""
    sentence_repeat = re.compile(r"([A-Z][^.!?]{25,}?[.!?])(?:\s+\1){1,}")
    prev = None
    while text != prev:
        prev = text
        text = sentence_repeat.sub(r"\1", text)
    return text


def _clean_merck_text(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""

    lines = [ln.strip() for ln in cleaned.splitlines()]
    out: List[str] = []
    prev_norm = ""

    drop_contains = (
        "view patient education",
        "drug information for the topic",
        "test your knowledge",
        "take a quiz",
        "cookie preferences",
        "all rights reserved",
    )
    tail_markers = (
        "drug information for the topic",
        "test your knowledge",
        "cookie preferences",
        "copyright",
    )

    for ln in lines:
        if not ln:
            if out and out[-1] != "":
                out.append("")
            continue

        low = ln.lower()
        if any(marker in low for marker in tail_markers):
            break
        if any(token in low for token in drop_contains):
            continue

        if "|" in ln and len(ln) <= 90:
            continue
        if re.fullmatch(r"v\d+", ln, flags=re.I):
            continue
        if re.fullmatch(r"(?:image|table|video|audio)", low):
            continue
        if re.match(r"^by\s+.+\b(md|do|phd|mbbs)\b", ln, flags=re.I):
            continue
        if re.match(r"^reviewed(?:/revised)?\b", ln, flags=re.I):
            continue

        norm = _normalize_for_hash(ln)
        if norm and norm == prev_norm and len(norm) > 24:
            continue

        out.append(ln)
        if norm:
            prev_norm = norm

    merged = "\n".join(out).strip()
    merged = _collapse_repeated_sentence_blocks(merged)
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged.strip()


def fetch_merck(max_docs: int, sleep_s: float) -> List[ResourceRecord]:
    print(f"[Merck] Fetching up to {max_docs} pages from Merck eye-disorders...")
    if max_docs <= 0:
        return []

    index_html, _ = _http_get_text(MERCK_INDEX_URL, timeout=60, retries=3)
    links = _extract_merck_links(index_html)

    records: List[ResourceRecord] = []
    pbar_links = tqdm(links, desc="[Merck]", unit="page")
    for link in pbar_links:
        if len(records) >= max_docs:
            break

        try:
            html, _ = _http_get_text(link, timeout=60, retries=3)
        except Exception as exc:
            print(f"  [Merck] skip {link} ({exc})")
            continue

        title = _extract_title(html, fallback="Merck eye-disorders")
        text = _clean_merck_text(_strip_html_tags(_extract_region(html)))
        if len(text) < 600:
            continue

        records.append(
            ResourceRecord(
                source_group="Merck Manual Professional",
                source_name="Merck Manual Professional",
                title=title,
                url=link,
                content=_truncate(text),
                metadata={},
            )
        )

        if sleep_s > 0:
            time.sleep(sleep_s)

    print(f"[Merck] Collected {len(records)} article records.")
    return records


def _clean_wikitext(text: str) -> str:
    """Strip MediaWiki markup to extract plain text."""
    import re
    # Remove wiki templates {{...}}
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    # Remove tables {|...|}
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    # Remove files/images [[File:...]]
    text = re.sub(r'\[\[(?:File|Image):.*?\]\]', '', text, flags=re.IGNORECASE)
    # Convert simple links [[Link|Text]] to Text, and [[Link]] to Link
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # Remove external links [url text]
    text = re.sub(r'\[http\S+\s+([^\]]+)\]', r'\1', text)
    # Remove headings syntax
    text = re.sub(r'=+([^=]+)=+', r'\1', text)
    # Remove bold/italic syntax
    text = re.sub(r'\'\'\'?', '', text)
    # Remove <ref> tags
    text = re.sub(r'<ref.*?>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref.*?/>', '', text)
    return text.strip()


# ── EyeWiki (AAO): Curated Ophthalmology Wiki ───────────────────────────────
EYEWIKI_API_URL = "https://eyewiki.aao.org/w/api.php"

def _eyewiki_get_all_pages(sleep_s: float) -> List[str]:
    """Get all article titles from EyeWiki using the allpages API."""
    all_titles: List[str] = []
    apcontinue = ""

    while True:
        params: Dict[str, object] = {
            "action": "query",
            "list": "allpages",
            "aplimit": 500,
            "apnamespace": 0,
            "format": "json",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue

        try:
            payload = _get_json(EYEWIKI_API_URL, params=params, timeout=60, retries=3)
        except Exception as exc:
            print(f"  [EyeWiki] allpages error: {exc}")
            break

        pages = payload.get("query", {}).get("allpages", [])
        for p in pages:
            title = p.get("title", "")
            if title and ":" not in title:  # Skip talk/template pages
                all_titles.append(title)

        cont = payload.get("continue", {})
        apcontinue = cont.get("apcontinue", "")
        if not apcontinue:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)

    return all_titles


def fetch_eyewiki(max_docs: int, sleep_s: float) -> List[ResourceRecord]:
    """Fetch ophthalmology articles from AAO EyeWiki."""
    print(f"[EyeWiki] Fetching up to {max_docs} articles...")
    if max_docs <= 0:
        return []

    all_titles = _eyewiki_get_all_pages(sleep_s)
    print(f"  [EyeWiki] Found {len(all_titles)} total pages")

    records: List[ResourceRecord] = []
    batch_size = 5

    titles_to_fetch = all_titles[:max_docs]

    for i in range(0, len(titles_to_fetch), batch_size):
        batch = titles_to_fetch[i:i + batch_size]
        params = {
            "action": "query",
            "prop": "revisions|info",
            "rvprop": "content",
            "rvslots": "main",
            "titles": "|".join(batch),
            "format": "json",
            "inprop": "url",
        }
        try:
            payload = _get_json(EYEWIKI_API_URL, params=params, timeout=60, retries=3)
        except Exception as exc:
            print(f"  [EyeWiki] batch skip at {i} ({exc})")
            continue

        pages = payload.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1" or "missing" in page_data:
                continue
            title = page_data.get("title", "Unknown")
            url = page_data.get("fullurl", f"https://eyewiki.aao.org/{quote(title)}")

            revisions = page_data.get("revisions", [])
            if not revisions:
                continue
            
            rev = revisions[0]
            raw_wiki = rev.get("slots", {}).get("main", {}).get("*", "")

            text = _clean_text(_clean_wikitext(raw_wiki))
            if len(text) < 500:
                continue

            records.append(
                ResourceRecord(
                    source_group="EyeWiki (AAO)",
                    source_name="AAO EyeWiki",
                    title=title,
                    url=url,
                    content=_truncate(text, max_chars=60000),
                    metadata={"page_id": page_id},
                )
            )

        if sleep_s > 0:
            time.sleep(sleep_s)

        if (i + batch_size) % 100 == 0:
            print(f"  [EyeWiki] Progress: {len(records)} articles fetched ({i + batch_size}/{len(titles_to_fetch)})")

    print(f"[EyeWiki] Collected {len(records)} article records.")
    return records


# ── PMC Open Access Full-Text Articles ───────────────────────────────────────
PMC_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PMC_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

PMC_OPHTHALMIC_QUERIES = [
    "ophthalmology[Journal] AND open access[Filter]",
    "corneal disease[MeSH] AND open access[Filter]",
    "retinal disease[MeSH] AND open access[Filter]",
    "glaucoma[MeSH] AND open access[Filter]",
    "cataract[MeSH] AND open access[Filter]",
    "uveitis[MeSH] AND open access[Filter]",
    "optic nerve diseases[MeSH] AND open access[Filter]",
    "eye neoplasms[MeSH] AND open access[Filter]",
    "ocular surface[Title/Abstract] AND open access[Filter]",
    "leukocoria[Title/Abstract] AND open access[Filter]",
    "corneal opacity[Title/Abstract] AND open access[Filter]",
    "cherry red spot[Title/Abstract] AND open access[Filter]",
    "diabetic retinopathy treatment AND open access[Filter]",
    "keratitis management AND open access[Filter]",
    "strabismus AND open access[Filter]",
]


def fetch_pmc_fulltext(max_docs: int, sleep_s: float) -> List[ResourceRecord]:
    """
    Fetch full-text articles from PMC Open Access subset.

    NOTE: efetch with db=pmc&rettype=txt returns raw JATS/NLM XML, NOT plain
    text.  We therefore fetch each article's canonical HTML page and strip it
    with the same pipeline used for StatPearls/Merck, giving clean prose.
    """
    print(f"[PMC Full-Text] Fetching up to {max_docs} open-access articles...")
    if max_docs <= 0:
        return []

    all_pmcids: List[str] = []
    seen_ids: set = set()

    for query in PMC_OPHTHALMIC_QUERIES:
        if len(all_pmcids) >= max_docs * 3:  # Over-fetch; many will be short
            break
        params = _with_ncbi_api_key({
            "db": "pmc",
            "term": query,
            "retmax": min(500, max_docs),
            "retmode": "json",
            "sort": "relevance",
        })
        try:
            payload = _get_json(PMC_SEARCH_URL, params=params, timeout=45, retries=3)
        except Exception as exc:
            print(f"  [PMC] search error for query: {exc}")
            continue

        ids = payload.get("esearchresult", {}).get("idlist", [])
        for pmcid in ids:
            if pmcid not in seen_ids:
                all_pmcids.append(pmcid)
                seen_ids.add(pmcid)

        if sleep_s > 0:
            time.sleep(sleep_s)

    print(f"  [PMC] Found {len(all_pmcids)} unique PMC IDs")

    records: List[ResourceRecord] = []

    pbar = tqdm(all_pmcids, desc="[PMC]", unit="article")
    for pmcid in pbar:
        if len(records) >= max_docs:
            break

        article_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
        try:
            html, _ = _http_get_text(article_url, timeout=60, retries=3)
        except Exception as exc:
            print(f"  [PMC] skip PMC{pmcid} ({exc})")
            continue

        # Extract and clean the main article region
        title = _extract_title(html, fallback=f"PMC{pmcid}")
        text = _clean_text(_strip_html_tags(_extract_region(html)))

        # Require a meaningful amount of content
        if len(text) < 1000:
            continue
        if not _is_ophthalmic_text(title, text):
            continue

        records.append(
            ResourceRecord(
                source_group="PMC Open Access",
                source_name="PubMed Central",
                title=title,
                url=article_url,
                content=_truncate(text, max_chars=60000),
                metadata={"pmcid": pmcid},
            )
        )

        if sleep_s > 0:
            time.sleep(sleep_s * 2)  # Be polite to PMC

    pbar.close()
    print(f"[PMC Full-Text] Collected {len(records)} full-text article records.")
    return records


# ── Wikipedia: Comprehensive Ophthalmology Articles ──────────────────────────
WIKIPEDIA_ARTICLES = [
    # ── Eye Anatomy ──────────────────────────────────────────────────────────
    "Human eye", "Cornea", "Retina", "Iris (anatomy)", "Pupil", "Lens (anatomy)",
    "Sclera", "Conjunctiva", "Choroid", "Vitreous body", "Aqueous humour",
    "Anterior chamber of the eye", "Macula", "Fovea centralis", "Optic disc",
    "Optic nerve", "Ciliary body", "Trabecular meshwork", "Schlemm's canal",
    "Retinal pigment epithelium", "Bruch's membrane", "Bowman's layer",
    "Descemet's membrane", "Corneal endothelium", "Corneal epithelium",
    "Corneal stroma", "Extraocular muscles", "Lacrimal apparatus",
    "Meibomian gland", "Eyelid", "Orbital bone",
    # ── Refractive Errors ────────────────────────────────────────────────────
    "Myopia", "Hypermetropia", "Astigmatism", "Presbyopia", "Anisometropia",
    # ── Corneal Diseases ─────────────────────────────────────────────────────
    "Keratitis", "Corneal ulcer", "Fungal keratitis", "Acanthamoeba keratitis",
    "Herpes simplex keratitis", "Keratoconus", "Corneal dystrophy",
    "Fuchs' dystrophy", "Band keratopathy", "Corneal abrasion",
    "Corneal neovascularization", "Bullous keratopathy", "Corneal opacity",
    "Interstitial keratitis", "Thygeson's superficial punctate keratopathy",
    "Exposure keratopathy", "Neurotrophic keratitis",
    "Peters anomaly", "Sclerocornea",
    # ── Corneal Surgery ──────────────────────────────────────────────────────
    "Corneal transplantation", "LASIK", "Photorefractive keratectomy",
    "Corneal cross-linking", "SMILE (surgery)",
    # ── Conjunctival Diseases ────────────────────────────────────────────────
    "Conjunctivitis", "Allergic conjunctivitis", "Trachoma", "Pterygium (conjunctiva)",
    "Pinguecula", "Subconjunctival hemorrhage", "Ocular cicatricial pemphigoid",
    "Superior limbic keratoconjunctivitis", "Vernal keratoconjunctivitis",
    # ── Scleral Diseases ─────────────────────────────────────────────────────
    "Scleritis", "Episcleritis", "Scleromalacia",
    # ── Anterior Chamber / Uveal Diseases ────────────────────────────────────
    "Uveitis", "Anterior uveitis", "Intermediate uveitis", "Posterior uveitis",
    "Iritis", "Iridocyclitis", "Hyphema", "Hypopyon",
    "Vogt–Koyanagi–Harada disease", "Sympathetic ophthalmia", "Behçet's disease",
    "Sarcoidosis", "Fuchs heterochromic iridocyclitis",
    # ── Iris and Pupil ───────────────────────────────────────────────────────
    "Anisocoria", "Horner's syndrome", "Adie syndrome",
    "Argyll Robertson pupil", "Marcus Gunn pupil",
    "Relative afferent pupillary defect", "Iris coloboma", "Aniridia",
    "Heterochromia iridum", "Leukocoria", "Rubeosis iridis",
    # ── Lens and Cataract ────────────────────────────────────────────────────
    "Cataract", "Congenital cataract", "Age-related cataract",
    "Cataract surgery", "Phacoemulsification",
    "Intraocular lens", "Posterior capsule opacification",
    "Ectopia lentis", "Aphakia", "Pseudophakia",
    # ── Glaucoma ─────────────────────────────────────────────────────────────
    "Glaucoma", "Open-angle glaucoma", "Angle-closure glaucoma",
    "Normal-tension glaucoma", "Congenital glaucoma",
    "Neovascular glaucoma", "Pigmentary glaucoma",
    "Pseudoexfoliation syndrome", "Ocular hypertension",
    "Trabeculectomy", "Glaucoma surgery", "Tonometry", "Gonioscopy",
    # ── Retinal Diseases ─────────────────────────────────────────────────────
    "Diabetic retinopathy", "Age-related macular degeneration",
    "Retinal detachment", "Retinal vein occlusion",
    "Central retinal artery occlusion", "Branch retinal artery occlusion",
    "Retinopathy of prematurity", "Retinitis pigmentosa",
    "Macular hole", "Epiretinal membrane", "Macular edema",
    "Central serous chorioretinopathy", "Macular pucker",
    "Retinal tear", "Vitreous detachment", "Vitreous hemorrhage",
    "Cherry-red spot", "Cotton-wool spot", "Roth's spot",
    "Retinal vasculitis", "Coats' disease", "Eales disease",
    "Best disease", "Stargardt disease", "Leber congenital amaurosis",
    "Retinitis", "Cytomegalovirus retinitis", "Toxoplasmic retinochoroiditis",
    # ── Retinal Surgery/Treatment ────────────────────────────────────────────
    "Vitrectomy", "Scleral buckle", "Pneumatic retinopexy",
    "Anti-VEGF", "Ranibizumab", "Bevacizumab", "Aflibercept",
    "Photodynamic therapy", "Retinal laser photocoagulation",
    "Intravitreal injection",
    # ── Choroidal Diseases ───────────────────────────────────────────────────
    "Choroidal neovascularization", "Choroiditis",
    "Polypoidal choroidal vasculopathy",
    # ── Optic Nerve Diseases ─────────────────────────────────────────────────
    "Optic neuritis", "Papilledema", "Optic atrophy",
    "Ischemic optic neuropathy", "Optic disc drusen",
    "Leber's hereditary optic neuropathy", "Toxic optic neuropathy",
    "Neuromyelitis optica", "Optic nerve glioma",
    # ── Neuro-Ophthalmology ──────────────────────────────────────────────────
    "Nystagmus", "Cranial nerve III palsy", "Cranial nerve IV palsy",
    "Cranial nerve VI palsy", "Internuclear ophthalmoplegia",
    "Visual field", "Homonymous hemianopsia", "Bitemporal hemianopsia",
    "Cortical blindness", "Charles Bonnet syndrome",
    # ── Pediatric Ophthalmology ──────────────────────────────────────────────
    "Amblyopia", "Strabismus", "Esotropia", "Exotropia",
    "Retinoblastoma", "Congenital nasolacrimal duct obstruction",
    "Pediatric cataract",
    # ── Oculoplastics / Orbit ────────────────────────────────────────────────
    "Ptosis (eyelid)", "Entropion", "Ectropion", "Blepharospasm",
    "Orbital cellulitis", "Preseptal cellulitis", "Graves' ophthalmopathy",
    "Dacryocystitis", "Dacryoadenitis", "Orbital fracture",
    "Enucleation of the eye", "Evisceration (ophthalmology)",
    "Exenteration", "Blepharoplasty",
    # ── Eyelid / Adnexa ──────────────────────────────────────────────────────
    "Chalazion", "Stye", "Blepharitis", "Trichiasis", "Distichiasis",
    "Xanthelasma", "Dermatochalasis", "Lagophthalmos",
    # ── Ocular Surface Disease ───────────────────────────────────────────────
    "Dry eye syndrome", "Sjögren syndrome", "Stevens–Johnson syndrome",
    "Meibomian gland dysfunction",
    # ── Ocular Trauma ────────────────────────────────────────────────────────
    "Eye injury", "Corneal foreign body", "Globe rupture",
    "Chemical burn", "Traumatic optic neuropathy",
    "Orbital fracture", "Commotio retinae", "Berlin's edema",
    # ── Ocular Oncology ──────────────────────────────────────────────────────
    "Choroidal melanoma", "Ocular melanosis",
    "Ocular surface squamous neoplasia", "Orbital lymphoma",
    # ── Systemic Disease Eye Manifestations ──────────────────────────────────
    "Diabetic eye disease", "Hypertensive retinopathy",
    "Sickle cell retinopathy", "Ocular manifestations of HIV/AIDS",
    "Ocular toxoplasmosis", "Ocular tuberculosis",
    "Rheumatoid arthritis", "Systemic lupus erythematosus",
    "Marfan syndrome", "Down syndrome",
    "Wilson's disease", "Tay–Sachs disease", "Niemann–Pick disease",
    # ── Ophthalmic Diagnostics ───────────────────────────────────────────────
    "Slit lamp", "Optical coherence tomography", "Fluorescein angiography",
    "Fundus photography", "Visual acuity", "Snellen chart",
    "Ishihara test", "Visual field test", "Retinoscopy",
    "Ophthalmoscopy", "Pachymetry", "Specular microscopy",
    "Corneal topography", "Biometry (ophthalmology)",
    "Electroretinography", "Visual evoked potential",
    "OCT angiography", "Indocyanine green angiography",
    "B-scan ultrasonography",
    # ── Ophthalmic Pharmacology ──────────────────────────────────────────────
    "Timolol", "Latanoprost", "Brimonidine", "Dorzolamide",
    "Pilocarpine", "Atropine", "Cyclopentolate", "Tropicamide",
    "Prednisolone acetate", "Dexamethasone", "Fluorometholone",
    "Ciprofloxacin", "Moxifloxacin", "Ofloxacin",
    "Natamycin", "Voriconazole", "Cyclosporine (ophthalmic)",
    # ── Signs and Findings ───────────────────────────────────────────────────
    "Red eye (medicine)", "Photophobia", "Floater", "Photopsia",
    "Diplopia", "Metamorphopsia", "Scotoma", "Amaurosis fugax",
    "Proptosis", "Enophthalmos", "Epiphora (medicine)",
    "Chemosis", "Symblepharon", "Pannus (eye)",
    "Kayser–Fleischer ring", "Arcus senilis", "Bitot's spots",
    "Keratomalacia",
    # ── Ophthalmic Procedures ────────────────────────────────────────────────
    "YAG laser capsulotomy", "Laser iridotomy",
    "Dacryocystorhinostomy", "Strabismus surgery",
    "Punctal plug", "Amniotic membrane transplantation",
]

WIKIPEDIA_DISCOVERY_STRICT_TERMS = {
    "retina", "retinal", "macula", "cornea", "corneal", "keratitis",
    "uveitis", "glaucoma", "cataract", "conjunctiva", "sclera", "choroid",
    "optic nerve", "ophthalmic", "ophthalmology", "intravitreal", "fundus",
    "gonioscopy", "tonometry", "retinopathy", "strabismus", "amblyopia",
    "orbital", "eyelid", "lacrimal",
}


def _keyword_in_text(text: str, keyword: str) -> bool:
    kw = (keyword or "").strip().lower()
    if not kw:
        return False

    if " " in kw or "-" in kw:
        return kw in text

    return re.search(rf"\b{re.escape(kw)}\b", text) is not None


def _count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    return sum(1 for kw in keywords if _keyword_in_text(text, kw))


def _wiki_discovery_candidate_ok(title: str, snippet_html: str) -> bool:
    """Require strong ophthalmic signal to admit search-discovered pages."""
    snippet_plain = _clean_text(_strip_html_tags(snippet_html))
    probe = _clean_text(f"{title}\n{snippet_plain}").lower()

    if not _is_ophthalmic_text(title, probe):
        return False

    strict_hits = _count_keyword_hits(probe, WIKIPEDIA_DISCOVERY_STRICT_TERMS)
    broad_hits = _count_keyword_hits(probe, OPHTHALMIC_KEYWORDS)
    return strict_hits >= 2 and broad_hits >= 2


def _wiki_get_extracts(titles: List[str], sleep_s: float) -> List[ResourceRecord]:
    """Fetch plain-text extracts from Wikipedia for a batch of article titles."""
    records: List[ResourceRecord] = []
    # Wikipedia truncates extracts when too many titles are requested at once.
    # Use small batches to ensure full article content is returned.
    batch_size = 5

    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        params = {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": "true",
            "exsectionformat": "plain",
            "exlimit": str(len(batch)),
            "redirects": "1",
            "titles": "|".join(batch),
            "format": "json",
            "inprop": "url",
        }
        try:
            payload = _get_json(WIKIPEDIA_API_URL, params=params, timeout=60, retries=3)
        except Exception as exc:
            print(f"  [Wikipedia] batch skip at {i} ({exc})")
            continue

        pages = payload.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1" or "missing" in page_data:
                continue

            title = page_data.get("title", "Unknown")
            extract = page_data.get("extract", "")
            url = page_data.get("fullurl", f"https://en.wikipedia.org/wiki/{quote(title)}")

            text = _clean_text(extract)
            if len(text) < 500:
                continue
            if not _is_ophthalmic_text(title, text):
                continue

            records.append(
                ResourceRecord(
                    source_group="Wikipedia (Ophthalmology)",
                    source_name="Wikipedia",
                    title=title,
                    url=url,
                    content=_truncate(text, max_chars=60000),
                    metadata={"page_id": page_id},
                )
            )

        if sleep_s > 0:
            time.sleep(sleep_s)

    return records


def _wiki_search_ophthalmic(max_results: int, sleep_s: float) -> List[str]:
    """Search Wikipedia for additional ophthalmology-related articles beyond the curated list."""
    search_queries = [
        "ophthalmology", "eye disease", "retinal disease", "corneal disease",
        "glaucoma treatment", "cataract surgery", "uveitis",
        "optic nerve disease", "eye surgery", "ocular pharmacology",
        "pediatric eye disease", "neuro-ophthalmology",
        "eye anatomy", "visual impairment", "blindness causes",
        "eye examination", "ophthalmic imaging",
        "leukocoria differential", "corneal opacity causes",
    ]
    discovered: List[str] = []
    known_titles = {a.lower() for a in WIKIPEDIA_ARTICLES}

    for query in search_queries:
        if len(discovered) >= max_results:
            break
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": "0",
            "srlimit": 50,
            "format": "json",
        }
        try:
            payload = _get_json(WIKIPEDIA_API_URL, params=params, timeout=30, retries=2)
        except Exception:
            continue

        results = payload.get("query", {}).get("search", [])
        for item in results:
            title = item.get("title", "")
            if title.lower() not in known_titles and title not in discovered:
                if _wiki_discovery_candidate_ok(title, str(item.get("snippet", ""))):
                    discovered.append(title)
                    known_titles.add(title.lower())

        if sleep_s > 0:
            time.sleep(sleep_s)

    return discovered[:max_results]


def fetch_wikipedia(max_docs: int, sleep_s: float) -> List[ResourceRecord]:
    """Fetch ophthalmology articles from Wikipedia (curated list + search discovery)."""
    print(f"[Wikipedia] Fetching up to {max_docs} articles...")
    if max_docs <= 0:
        return []

    # Phase 1: Curated list (highest priority)
    curated_titles = WIKIPEDIA_ARTICLES[:max_docs]
    records = _wiki_get_extracts(curated_titles, sleep_s)
    print(f"  [Wikipedia] Phase 1 (curated): {len(records)} articles")

    # Phase 2: Search-discovered articles to fill remaining quota
    remaining = max_docs - len(records)
    if remaining > 0:
        discovered = _wiki_search_ophthalmic(max_results=remaining, sleep_s=sleep_s)
        if discovered:
            extra = _wiki_get_extracts(discovered, sleep_s)
            records.extend(extra)
            print(f"  [Wikipedia] Phase 2 (discovered): {len(extra)} articles")

    print(f"[Wikipedia] Collected {len(records)} total article records.")
    return records


def deduplicate_records(records: List[ResourceRecord], seen_hashes: Set[str]) -> List[ResourceRecord]:
    deduped: List[ResourceRecord] = []
    removed = 0

    for rec in records:
        h = _content_hash(rec.content)
        if h in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(h)
        authority_rank = SOURCE_AUTHORITY_RANK.get(rec.source_group)
        rec.metadata = {
            **rec.metadata,
            "content_hash": h,
            "authority_rank": authority_rank if authority_rank is not None else 99,
        }
        deduped.append(rec)

    if removed:
        print(f"[Dedup] Removed {removed} duplicate records by content hash.")
    return deduped


def _render_metadata(meta: Dict[str, object]) -> str:
    if not meta:
        return ""
    pairs = []
    for key, value in meta.items():
        if isinstance(value, list):
            sval = ", ".join(str(v) for v in value)
        else:
            sval = str(value)
        pairs.append(f"{key}={sval}")
    return "; ".join(pairs)


def write_markdown(records: List[ResourceRecord], output_path: Path, mode: str = "w", include_header: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[ResourceRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.source_group].append(rec)

    order = [
        "EyeWiki (AAO)",
        "PMC Open Access",
        "Hugging Face - EYE-lit Complete",
        "Hugging Face - MedRAG Textbooks",
        "AAO Preferred Practice Patterns",
        "StatPearls (NCBI Bookshelf)",
        "Merck Manual Professional",
        "Wikipedia (Ophthalmology)",
    ]

    with output_path.open(mode, encoding="utf-8") as f:
        if include_header:
            f.write("# External Ophthalmic Resources\n\n")
            f.write(
                "Auto-fetched corpus for backward-compatible ingestion. "
                "Each section below is normalized to markdown with source metadata.\n\n"
            )

        for group in order:
            items = grouped.get(group, [])
            if not items:
                continue

            f.write(f"# {group}\n\n")
            for rec in items:
                f.write(f"## {rec.title}\n\n")
                f.write(f"*Source:* {rec.source_name}  \n")
                f.write(f"*URL:* {rec.url}  \n")
                meta_line = _render_metadata(rec.metadata)
                if meta_line:
                    f.write(f"*Metadata:* {meta_line}  \n")
                f.write("\n")
                f.write(rec.content.strip())
                f.write("\n\n")

    print(f"[Output] Wrote markdown corpus: {output_path}")


def write_manifest(records: List[ResourceRecord], output_path: Path, args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    for rec in records:
        counts[rec.source_group] += 1

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_records": len(records),
        "counts_by_source_group": dict(sorted(counts.items())),
        "config": {
            "sources": args.sources,
            "max_eye_lit": args.max_eye_lit,
            "max_medrag": args.max_medrag,
            "medrag_include_general": args.medrag_include_general,
            "max_aao": args.max_aao,
            "max_statpearls": args.max_statpearls,
            "max_merck": args.max_merck,
            "aao_pdf_pages": args.aao_pdf_pages,
            "sleep": args.sleep,
        },
        "output_md": str(args.output_md),
    }

    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Output] Wrote manifest: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch additional ophthalmic resources and normalize to *_clean.md",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="eye-lit,medrag,aao,statpearls,merck,wikipedia,eyewiki,pmc",
        help="Comma-separated list: eye-lit,medrag,aao,statpearls,merck,wikipedia,eyewiki,pmc",
    )
    parser.add_argument("--max-eye-lit", type=int, default=999999)
    parser.add_argument("--max-medrag", type=int, default=30000)
    parser.add_argument("--medrag-include-general", dest="medrag_include_general", action="store_true", default=True)
    parser.add_argument("--medrag-no-general", dest="medrag_include_general", action="store_false")
    parser.add_argument("--max-aao", type=int, default=50000)
    parser.add_argument("--max-statpearls", type=int, default=8000)
    parser.add_argument("--max-merck", type=int, default=90000)
    parser.add_argument("--max-wikipedia", type=int, default=4000)
    parser.add_argument("--max-eyewiki", type=int, default=900000)
    parser.add_argument("--max-pmc", type=int, default=5)
    parser.add_argument("--aao-pdf-pages", type=int, default=50000)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD_DEFAULT)
    parser.add_argument("--output-manifest", type=Path, default=OUTPUT_MANIFEST_DEFAULT)
    parser.add_argument("--append", action="store_true", help="Append to existing markdown instead of overwriting")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    enabled = {s.strip().lower() for s in args.sources.split(",") if s.strip()}

    all_records: List[ResourceRecord] = []
    seen_hashes: Set[str] = set()
    header_written = args.append

    def process_source(source_id: str, fetch_func, **kwargs):
        nonlocal header_written
        if source_id not in enabled:
            return
        
        recs = fetch_func(**kwargs)
        if not recs:
            return
            
        new_deduped = deduplicate_records(recs, seen_hashes)
        if not new_deduped:
            return
            
        mode = "a" if header_written else "w"
        write_markdown(new_deduped, args.output_md, mode=mode, include_header=not header_written)
        header_written = True
        all_records.extend(new_deduped)

    # Fetch and write incrementally in the preferred grouping order
    process_source("eyewiki", fetch_eyewiki, max_docs=args.max_eyewiki, sleep_s=args.sleep)
    process_source("pmc", fetch_pmc_fulltext, max_docs=args.max_pmc, sleep_s=args.sleep)
    process_source("eye-lit", fetch_eye_lit, max_docs=args.max_eye_lit, sleep_s=args.sleep)
    process_source(
        "medrag", 
        fetch_medrag, 
        max_docs=args.max_medrag, 
        sleep_s=args.sleep, 
        include_general=args.medrag_include_general
    )
    process_source(
        "aao", 
        fetch_aao_ppp, 
        max_docs=args.max_aao, 
        pdf_pages=args.aao_pdf_pages, 
        sleep_s=args.sleep
    )
    process_source("statpearls", fetch_statpearls, max_chapters=args.max_statpearls, sleep_s=args.sleep)
    process_source("merck", fetch_merck, max_docs=args.max_merck, sleep_s=args.sleep)
    process_source("wikipedia", fetch_wikipedia, max_docs=args.max_wikipedia, sleep_s=args.sleep)

    write_manifest(all_records, args.output_manifest, args)

    print("\n" + "=" * 72)
    print("External resource fetch complete")
    print("=" * 72)
    print(f"Records written: {len(all_records)}")
    print(f"Markdown: {args.output_md}")
    print(f"Manifest: {args.output_manifest}")
    print("\nNext steps:")
    print("  1) python scripts/chunk_data.py")
    print("  2) python scripts/ingest_db.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
