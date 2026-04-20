"""
fetch_articles.py — Fetch ~5000 Ophthalmic Articles from Public APIs
─────────────────────────────────────────────────────────────────────
Sources:
  1. PubMed (NCBI E-Utilities)     — peer-reviewed abstracts
  2. EuropePMC                     — open-access articles
  3. Semantic Scholar              — cross-publisher scholarly papers
  4. MedlinePlus Connect           — consumer health summaries

Output:
  data/processed/pubmed_articles_clean.md
  (H1 = topic category, H2 = article title — backward-compatible with chunk_data.py)

Usage:
  python scripts/fetch_articles.py                    # full fetch (~5000 articles)
  python scripts/fetch_articles.py --max-per-query 10 # quick test
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlencode, quote_plus
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ── Output Path ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
OUTPUT_MD = ROOT / "data" / "processed" / "pubmed_articles_clean.md"

# ── Rate Limiting ────────────────────────────────────────────────────────────
REQUEST_DELAY = 0.4   # PubMed/EuropePMC (NCBI: max 3/sec)
S2_DELAY      = 1.05  # Semantic Scholar (1 req/sec for search endpoint)


# ══════════════════════════════════════════════════════════════════════════════
# QUERY CATEGORIES — 20 India-Relevant Ophthalmic Topics
# ══════════════════════════════════════════════════════════════════════════════

PUBMED_QUERIES = {
    # ── Retinal Diseases ─────────────────────────────────────────────────────
    "Diabetic Retinopathy": (
        "(diabetic retinopathy[Title/Abstract]) AND "
        "(screening OR epidemiology OR anti-VEGF OR laser OR India)"
    ),
    "Age-Related Macular Degeneration": (
        "(age-related macular degeneration[Title/Abstract]) AND "
        "(treatment OR anti-VEGF OR geographic atrophy OR drusen OR imaging)"
    ),
    "Retinal Detachment and Vitreoretinal": (
        "(retinal detachment[Title/Abstract] OR vitrectomy[Title/Abstract] OR "
        "macular hole[Title/Abstract] OR epiretinal membrane[Title/Abstract]) AND "
        "(outcomes OR surgery OR management)"
    ),
    "Retinal Vascular Occlusions": (
        "(retinal vein occlusion[Title/Abstract] OR retinal artery occlusion[Title/Abstract] "
        "OR central retinal vein[Title/Abstract]) AND "
        "(treatment OR thrombolysis OR anti-VEGF)"
    ),
    "Retinopathy of Prematurity": (
        "(retinopathy of prematurity[Title/Abstract]) AND "
        "(screening OR anti-VEGF OR laser OR India OR neonatal)"
    ),

    # ── Anterior Segment ─────────────────────────────────────────────────────
    "Cataract Surgery": (
        "(cataract[Title/Abstract]) AND "
        "(phacoemulsification OR intraocular lens OR SICS OR outcomes OR complications)"
    ),
    "Corneal Diseases": (
        "(corneal ulcer[Title/Abstract] OR fungal keratitis[Title/Abstract] OR "
        "Acanthamoeba keratitis[Title/Abstract] OR keratoconus[Title/Abstract] OR "
        "corneal transplant[Title/Abstract]) AND "
        "(treatment OR India OR tropical OR outcomes)"
    ),
    "Refractive Surgery": (
        "(LASIK[Title/Abstract] OR PRK[Title/Abstract] OR SMILE[Title/Abstract] OR "
        "refractive surgery[Title/Abstract]) AND "
        "(outcomes OR complications OR corneal biomechanics)"
    ),

    # ── Glaucoma ─────────────────────────────────────────────────────────────
    "Glaucoma": (
        "(glaucoma[Title/Abstract]) AND "
        "(diagnosis OR intraocular pressure OR trabeculectomy OR angle closure OR India)"
    ),
    "Glaucoma Imaging and Diagnostics": (
        "(glaucoma[Title/Abstract]) AND "
        "(OCT OR visual field OR RNFL OR optic nerve head OR progression)"
    ),

    # ── Neuro-Ophthalmology ──────────────────────────────────────────────────
    "Neuro-Ophthalmology": (
        "(optic neuritis[Title/Abstract] OR Graves ophthalmopathy[Title/Abstract] OR "
        "papilledema[Title/Abstract] OR ischemic optic neuropathy[Title/Abstract] OR "
        "white dot syndrome[Title/Abstract] OR multiple sclerosis eye[Title/Abstract])"
    ),

    # ── Ocular Surface ──────────────────────────────────────────────────────
    "Ocular Surface Disease": (
        "(dry eye[Title/Abstract] OR Sjogren syndrome eye[Title/Abstract] OR "
        "blepharitis[Title/Abstract] OR pterygium[Title/Abstract] OR "
        "meibomian gland[Title/Abstract]) AND "
        "(treatment OR management OR pathophysiology)"
    ),

    # ── Uveitis and Inflammation ─────────────────────────────────────────────
    "Uveitis and Ocular Inflammation": (
        "(uveitis[Title/Abstract] OR scleritis[Title/Abstract] OR "
        "ocular inflammation[Title/Abstract] OR endophthalmitis[Title/Abstract]) AND "
        "(treatment OR diagnosis OR etiology)"
    ),

    # ── Pediatric ────────────────────────────────────────────────────────────
    "Pediatric Ophthalmology": (
        "(strabismus[Title/Abstract] OR amblyopia[Title/Abstract] OR "
        "congenital cataract[Title/Abstract] OR pediatric glaucoma[Title/Abstract] OR "
        "retinoblastoma[Title/Abstract]) AND "
        "(management OR screening OR outcomes)"
    ),

    # ── Ocular Oncology ──────────────────────────────────────────────────────
    "Ocular Oncology": (
        "(retinoblastoma[Title/Abstract] OR ocular melanoma[Title/Abstract] OR "
        "choroidal melanoma[Title/Abstract] OR ocular lymphoma[Title/Abstract]) AND "
        "(treatment OR prognosis OR diagnosis)"
    ),

    # ── Oculoplastics and Orbit ──────────────────────────────────────────────
    "Oculoplastics and Orbit": (
        "(orbital surgery[Title/Abstract] OR ptosis[Title/Abstract] OR "
        "dacryocystorhinostomy[Title/Abstract] OR orbital fracture[Title/Abstract] OR "
        "thyroid eye disease[Title/Abstract]) AND "
        "(management OR outcomes OR surgical)"
    ),

    # ── Ophthalmic Imaging ───────────────────────────────────────────────────
    "Ophthalmic Imaging": (
        "(optical coherence tomography[Title/Abstract] OR OCT angiography[Title/Abstract] OR "
        "fluorescein angiography[Title/Abstract] OR fundus photography[Title/Abstract]) AND "
        "(ophthalmology OR retina OR diagnosis)"
    ),

    # ── Ophthalmic Pharmacology ──────────────────────────────────────────────
    "Ophthalmic Pharmacology": (
        "(intravitreal injection[Title/Abstract] OR anti-VEGF[Title/Abstract] OR "
        "ranibizumab[Title/Abstract] OR bevacizumab[Title/Abstract] OR "
        "aflibercept[Title/Abstract]) AND "
        "(ophthalmology OR retina OR efficacy)"
    ),

    # ── Community and Public Health ──────────────────────────────────────────
    "Community Eye Health": (
        "(blindness prevention[Title/Abstract] OR eye health India[Title/Abstract] OR "
        "trachoma[Title/Abstract] OR vitamin A deficiency eye[Title/Abstract] OR "
        "vision screening[Title/Abstract]) AND "
        "(India OR developing countries OR community)"
    ),

    # ── Ocular Genetics ─────────────────────────────────────────────────────
    "Ocular Genetics": (
        "(retinitis pigmentosa[Title/Abstract] OR Leber congenital amaurosis[Title/Abstract] OR "
        "Stargardt disease[Title/Abstract] OR inherited retinal[Title/Abstract] OR "
        "gene therapy retina[Title/Abstract]) AND "
        "(genetics OR mutation OR therapy)"
    ),

    # ── Leukocoria and White Pupil ───────────────────────────────────────────
    "Leukocoria and White Pupil": (
        "(leukocoria[Title/Abstract] OR white pupillary reflex[Title/Abstract] OR "
        "white pupil[Title/Abstract] OR abnormal red reflex[Title/Abstract] OR "
        "absent red reflex[Title/Abstract]) AND "
        "(differential diagnosis OR retinoblastoma OR cataract OR evaluation)"
    ),

    # ── Anterior Segment Signs and Symptoms ──────────────────────────────────
    "Anterior Segment Signs": (
        "(corneal opacity[Title/Abstract] OR corneal scar[Title/Abstract] OR "
        "band keratopathy[Title/Abstract] OR corneal dystrophy[Title/Abstract] OR "
        "anterior segment anomaly[Title/Abstract] OR iris lesion[Title/Abstract] OR "
        "pupil abnormality[Title/Abstract]) AND "
        "(diagnosis OR differential OR management)"
    ),

    # ── Cherry-Red Spot and Retinal Vascular Signs ───────────────────────────
    "Cherry-Red Spot Differential": (
        "(cherry red spot[Title/Abstract] OR cherry-red spot[Title/Abstract] OR "
        "central retinal artery occlusion[Title/Abstract] OR Tay-Sachs[Title/Abstract] OR "
        "Niemann-Pick[Title/Abstract] OR macular ischemia[Title/Abstract]) AND "
        "(fundoscopy OR ophthalmoscopy OR differential diagnosis OR retinal)"
    ),

    # ── Corneal Opacity and Scarring ─────────────────────────────────────────
    "Corneal Opacity and Scarring": (
        "(corneal opacity[Title/Abstract] OR corneal scar[Title/Abstract] OR "
        "leucoma[Title/Abstract] OR nebula cornea[Title/Abstract] OR "
        "corneal vascularization[Title/Abstract]) AND "
        "(etiology OR management OR keratoplasty OR visual rehabilitation)"
    ),

    # ── Pupillary Abnormalities ──────────────────────────────────────────────
    "Pupillary Abnormalities": (
        "(pupillary abnormality[Title/Abstract] OR anisocoria[Title/Abstract] OR "
        "Marcus Gunn pupil[Title/Abstract] OR Horner syndrome[Title/Abstract] OR "
        "relative afferent pupillary defect[Title/Abstract] OR "
        "Argyll Robertson pupil[Title/Abstract] OR Adie pupil[Title/Abstract]) AND "
        "(diagnosis OR evaluation OR neuro-ophthalmology)"
    ),

    # ── Ocular Emergency Signs ───────────────────────────────────────────────
    "Ocular Emergency Signs": (
        "(ocular emergency[Title/Abstract] OR eye emergency[Title/Abstract] OR "
        "acute vision loss[Title/Abstract] OR chemical eye injury[Title/Abstract] OR "
        "globe rupture[Title/Abstract] OR hyphema[Title/Abstract] OR "
        "acute angle closure[Title/Abstract]) AND "
        "(management OR triage OR emergency department)"
    ),

    # ── Ocular Manifestations of Systemic Disease ────────────────────────────
    "Systemic Disease Ocular Manifestations": (
        "(ocular manifestation[Title/Abstract]) AND "
        "(diabetes OR hypertension OR lupus OR sarcoidosis OR "
        "tuberculosis OR HIV OR leukemia OR sickle cell) AND "
        "(ophthalmology OR retina OR fundus)"
    ),
}


# ── Semantic Scholar Queries (broader, cross-publisher) ──────────────────────
SEMANTIC_SCHOLAR_QUERIES = [
    "diabetic retinopathy artificial intelligence screening India",
    "deep learning fundus image classification ophthalmology",
    "OCT image segmentation retinal disease",
    "glaucoma optic nerve head detection machine learning",
    "cataract surgery intraocular lens calculation outcomes",
    "fungal keratitis tropical developing countries treatment",
    "retinopathy of prematurity telemedicine India",
    "anti-VEGF intravitreal injection diabetic macular edema",
    "corneal cross-linking keratoconus outcome",
    "dry eye disease meibomian gland dysfunction",
    "uveitis immunosuppressive therapy biologics",
    "orbital decompression thyroid eye disease",
    "retinal gene therapy inherited dystrophy",
    "ocular surface squamous neoplasia treatment",
    "optic neuritis multiple sclerosis MRI",
    "macular hole vitrectomy internal limiting membrane",
    "angle closure glaucoma laser iridotomy India",
    "pterygium excision conjunctival autograft recurrence",
    "amblyopia treatment patching atropine penalization",
    "retinoblastoma chemotherapy enucleation outcomes",
    "trachoma mass drug administration India elimination",
    "endophthalmitis post cataract surgery prophylaxis",
    "central serous chorioretinopathy photodynamic therapy",
    "Vogt-Koyanagi-Harada disease diagnosis treatment",
    "scleral buckling versus vitrectomy retinal detachment",
    "blepharospasm botulinum toxin treatment",
    "congenital nasolacrimal duct obstruction probing",
    "optic disc drusen diagnosis management",
    "ocular trauma penetrating injury epidemiology India",
    "Behcet disease ocular manifestations treatment",
    "leukocoria white pupillary reflex differential diagnosis retinoblastoma",
    "leukocoria screening pediatric cataract congenital",
    "cherry red spot central retinal artery occlusion fundoscopy",
    "cherry red spot Tay-Sachs Niemann-Pick metabolic storage disease",
    "corneal opacity scar leucoma visual rehabilitation",
    "corneal opacity congenital pediatric differential diagnosis",
    "pupillary abnormality anisocoria Horner syndrome evaluation",
    "relative afferent pupillary defect Marcus Gunn diagnosis",
    "ocular emergency triage acute vision loss management",
    "chemical eye injury irrigation emergency treatment",
    "ocular manifestations systemic disease diabetes hypertension",
    "anterior segment examination slit lamp findings",
    "band keratopathy corneal dystrophy anterior segment",
    "iris lesion nodule anterior segment differential",
]

# ── EuropePMC Queries (diverse, India-focused) ───────────────────────────────
EUROPEPMC_QUERIES = [
    "ophthalmology India clinical study",
    "diabetic retinopathy India screening",
    "fungal keratitis tropical climate",
    "retinopathy of prematurity developing countries",
    "cataract surgery outcomes India",
    "glaucoma angle closure prevalence India",
    "corneal ulcer bacterial India",
    "trachoma elimination India",
    "vitamin A deficiency xerophthalmia India",
    "strabismus amblyopia screening India",
    "retinoblastoma India outcome",
    "ocular tuberculosis uveitis India",
    "vernal keratoconjunctivitis treatment India",
    "pterygium surgery recurrence tropical",
    "endophthalmitis prevention cataract India",
    "refractive error myopia school children India",
    "optic neuritis NMO India",
    "central retinal vein occlusion anti-VEGF",
    "artificial intelligence retinal screening India",
    "teleophthalmology rural India",
    "leukocoria white pupil retinoblastoma India",
    "corneal opacity visual impairment India",
    "cherry red spot retinal artery occlusion case report",
    "pupillary abnormality neuro-ophthalmology evaluation",
    "ocular emergency department triage India",
    "systemic disease ocular manifestation India",
]


# ── HTTP Helper ──────────────────────────────────────────────────────────────
def _fetch_url(url: str, retries: int = 3) -> Optional[bytes]:
    """Fetch URL with retries and rate limiting."""
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "OphthalmicRAG/1.0"})
            with urlopen(req, timeout=30) as resp:
                return resp.read()
        except (URLError, HTTPError) as e:
            print(f"    [Retry {attempt+1}/{retries}] {e}")
            time.sleep(2 ** attempt)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: PubMed (NCBI E-Utilities)
# ══════════════════════════════════════════════════════════════════════════════

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def pubmed_search(query: str, max_results: int = 250) -> List[str]:
    """Search PubMed and return list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    url = f"{ESEARCH_URL}?{urlencode(params)}"
    data = _fetch_url(url)
    if not data:
        return []
    try:
        result = json.loads(data)
        return result.get("esearchresult", {}).get("idlist", [])
    except json.JSONDecodeError:
        return []


def pubmed_fetch_abstracts(pmids: List[str]) -> List[Dict]:
    """Fetch article metadata + abstracts for a batch of PMIDs."""
    if not pmids:
        return []

    articles = []
    # Batch in groups of 200 (NCBI limit)
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i+200]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
        }
        url = f"{EFETCH_URL}?{urlencode(params)}"
        data = _fetch_url(url)
        if not data:
            continue

        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            continue

        for article_elem in root.findall(".//PubmedArticle"):
            art = _parse_pubmed_article(article_elem)
            if art and len(art.get("abstract", "")) > 100:
                articles.append(art)

        time.sleep(REQUEST_DELAY)

    return articles


def _parse_pubmed_article(elem) -> Optional[Dict]:
    """Parse a single PubmedArticle XML element."""
    try:
        medline = elem.find("MedlineCitation")
        if medline is None:
            return None

        pmid_elem = medline.find("PMID")
        pmid = pmid_elem.text if pmid_elem is not None else "unknown"

        article = medline.find("Article")
        if article is None:
            return None

        # Title
        title_elem = article.find("ArticleTitle")
        title = _extract_text(title_elem) if title_elem is not None else "Untitled"
        title = _clean_text(title)

        # Abstract
        abstract_elem = article.find("Abstract")
        abstract_parts = []
        if abstract_elem is not None:
            for text_elem in abstract_elem.findall("AbstractText"):
                label = text_elem.get("Label", "")
                text = _extract_text(text_elem)
                if label and text:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)

        abstract = "\n\n".join(abstract_parts)
        if not abstract:
            return None

        # Journal
        journal_elem = article.find("Journal/Title")
        journal = journal_elem.text if journal_elem is not None else "Unknown Journal"

        # Year
        year = "Unknown"
        pub_date = article.find("Journal/JournalIssue/PubDate")
        if pub_date is not None:
            year_elem = pub_date.find("Year")
            if year_elem is not None:
                year = year_elem.text

        # Authors
        authors = []
        author_list = article.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author")[:3]:
                last = author.find("LastName")
                if last is not None and last.text:
                    authors.append(last.text)
        author_str = ", ".join(authors) + (" et al." if len(authors) >= 3 else "")

        # MeSH terms
        mesh_terms = []
        mesh_list = medline.find("MeshHeadingList")
        if mesh_list is not None:
            for mesh in mesh_list.findall("MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "authors": author_str,
            "mesh_terms": mesh_terms,
            "source_api": "pubmed",
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: EuropePMC (open-access articles with pagination)
# ══════════════════════════════════════════════════════════════════════════════

EUROPEPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def europepmc_search(query: str, max_results: int = 100) -> List[Dict]:
    """Search EuropePMC for open-access ophthalmology articles with pagination."""
    articles = []
    cursor_mark = "*"
    fetched = 0
    page_size = min(max_results, 100)

    while fetched < max_results:
        params = {
            "query": f"{query} AND OPEN_ACCESS:y",
            "format": "json",
            "pageSize": page_size,
            "resultType": "core",
            "cursorMark": cursor_mark,
        }
        url = f"{EUROPEPMC_URL}?{urlencode(params)}"
        data = _fetch_url(url)
        if not data:
            break

        try:
            result = json.loads(data)
            items = result.get("resultList", {}).get("result", [])
            if not items:
                break

            next_cursor = result.get("nextCursorMark", "")

            for item in items:
                pmid = item.get("pmid", "")
                title = _clean_text(item.get("title", ""))
                abstract = _clean_text(item.get("abstractText", ""))

                if not abstract or len(abstract) < 100:
                    continue

                journal = item.get("journalTitle", "Unknown Journal")
                year = str(item.get("pubYear", "Unknown"))
                auth = item.get("authorString", "")

                articles.append({
                    "pmid": pmid or f"EPMC_{item.get('id', 'unknown')}",
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "authors": auth[:80],
                    "mesh_terms": [],
                    "source_api": "europepmc",
                })

            fetched += len(items)

            if not next_cursor or next_cursor == cursor_mark:
                break
            cursor_mark = next_cursor

        except (json.JSONDecodeError, KeyError):
            break

        time.sleep(REQUEST_DELAY)

    return articles[:max_results]


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: Semantic Scholar (cross-publisher, no API key needed)
# ══════════════════════════════════════════════════════════════════════════════

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def semantic_scholar_search(query: str, max_results: int = 100) -> List[Dict]:
    """Search Semantic Scholar for ophthalmology papers."""
    articles = []
    offset = 0
    limit = min(max_results, 100)

    while offset < max_results:
        params = {
            "query": query,
            "offset": offset,
            "limit": limit,
            "fields": "title,abstract,year,authors,journal,externalIds",
        }
        url = f"{S2_SEARCH_URL}?{urlencode(params)}"
        data = _fetch_url(url)
        if not data:
            break

        try:
            result = json.loads(data)
            papers = result.get("data", [])
            if not papers:
                break

            for paper in papers:
                title = _clean_text(paper.get("title", ""))
                abstract = _clean_text(paper.get("abstract", ""))

                if not abstract or len(abstract) < 100:
                    continue

                # Extract PMID or use S2 paperId
                ext_ids = paper.get("externalIds", {}) or {}
                pmid = ext_ids.get("PubMed", "") or ext_ids.get("DOI", "") or paper.get("paperId", "unknown")

                # Authors
                author_list = paper.get("authors", []) or []
                auth_names = [a.get("name", "") for a in author_list[:3] if a.get("name")]
                auth_str = ", ".join(auth_names) + (" et al." if len(auth_names) >= 3 else "")

                # Journal
                journal_info = paper.get("journal", {}) or {}
                journal = journal_info.get("name", "Unknown Journal") if isinstance(journal_info, dict) else "Unknown Journal"

                year = str(paper.get("year", "Unknown"))

                articles.append({
                    "pmid": f"S2_{pmid}" if not str(pmid).isdigit() else pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "authors": auth_str,
                    "mesh_terms": [],
                    "source_api": "semantic_scholar",
                })

            offset += len(papers)
            if result.get("total", 0) <= offset:
                break

        except (json.JSONDecodeError, KeyError):
            break

        time.sleep(S2_DELAY)

    return articles[:max_results]


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4: MedlinePlus Connect (consumer health summaries)
# ══════════════════════════════════════════════════════════════════════════════

MEDLINEPLUS_TOPICS = [
    "glaucoma", "cataracts", "macular-degeneration", "diabetic-eye-problems",
    "dry-eye", "pink-eye", "retinal-detachment", "amblyopia", "strabismus",
    "astigmatism", "corneal-conditions", "eye-infections", "color-blindness",
    "eye-injuries", "floaters", "presbyopia", "uveitis", "blepharitis",
    "keratoconus", "optic-nerve-disorders",
]


def fetch_medlineplus() -> List[Dict]:
    """Fetch MedlinePlus consumer health topic summaries."""
    articles = []
    for topic in MEDLINEPLUS_TOPICS:
        api_url = f"https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term={topic}+eye&retmax=3"
        data = _fetch_url(api_url)
        if not data:
            time.sleep(REQUEST_DELAY)
            continue

        try:
            root = ET.fromstring(data)
            for doc in root.findall(".//document"):
                title_elem = doc.find(".//content[@name='title']")
                summary_elem = doc.find(".//content[@name='FullSummary']")

                title = _extract_text(title_elem) if title_elem is not None else topic.replace("-", " ").title()
                summary = _extract_text(summary_elem) if summary_elem is not None else ""

                if not summary or len(summary) < 50:
                    continue

                summary = re.sub(r"<[^>]+>", "", summary)
                summary = re.sub(r"\s+", " ", summary).strip()

                articles.append({
                    "pmid": f"MEDLINEPLUS_{topic}",
                    "title": _clean_text(title),
                    "abstract": summary,
                    "journal": "MedlinePlus Health Information",
                    "year": "2024",
                    "authors": "U.S. National Library of Medicine",
                    "mesh_terms": [],
                    "source_api": "medlineplus",
                })
        except ET.ParseError:
            pass

        time.sleep(REQUEST_DELAY)

    return articles


# ══════════════════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _extract_text(elem) -> str:
    """Extract all text from an XML element, including mixed content."""
    if elem is None:
        return ""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_extract_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _clean_text(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("#", "").replace("```", "")
    return text


def _sanitize_heading(text: str) -> str:
    """Make text safe for use as a markdown heading."""
    text = _clean_text(text)
    text = text.strip(".-:; ")
    return text if text else "Untitled"


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_markdown(
    articles_by_category: Dict[str, List[Dict]],
    output_path: Path,
):
    """Write all articles to a single clean markdown file compatible with chunk_data.py."""
    lines = []
    total = 0

    for category, articles in articles_by_category.items():
        if not articles:
            continue

        # H1 = topic category (maps to 'chapter' in chunk metadata)
        lines.append(f"# PubMed — {category}\n")

        for art in articles:
            total += 1
            pmid = art["pmid"]
            title = _sanitize_heading(art["title"])
            abstract = art["abstract"]
            journal = art.get("journal", "")
            year = art.get("year", "")
            authors = art.get("authors", "")
            source = art.get("source_api", "unknown")

            # H2 = article title (maps to 'section' in chunk metadata)
            lines.append(f"## {title} (PMID: {pmid})\n")

            # Citation line
            if authors and journal and year:
                lines.append(f"*{authors}. {journal}, {year}. [Source: {source}]*\n")

            # MeSH terms as keywords
            mesh = art.get("mesh_terms", [])
            if mesh:
                lines.append(f"**Keywords**: {', '.join(mesh[:8])}\n")

            # Abstract body
            lines.append(f"{abstract}\n")
            lines.append("")  # blank line separator

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✓ Wrote {total} articles → {output_path}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def _match_category(query: str) -> str:
    """Match a free-text query to the closest predefined category."""
    query_lower = query.lower()
    # Priority matches
    keyword_map = {
        "diabetic": "Diabetic Retinopathy",
        "glaucoma": "Glaucoma",
        "cataract": "Cataract Surgery",
        "cornea": "Corneal Diseases",
        "keratitis": "Corneal Diseases",
        "retinopathy of prematurity": "Retinopathy of Prematurity",
        "macular degeneration": "Age-Related Macular Degeneration",
        "optic neurit": "Neuro-Ophthalmology",
        "dry eye": "Ocular Surface Disease",
        "uveitis": "Uveitis and Ocular Inflammation",
        "strabismus": "Pediatric Ophthalmology",
        "amblyopia": "Pediatric Ophthalmology",
        "retinoblastoma": "Ocular Oncology",
        "orbital": "Oculoplastics and Orbit",
        "thyroid": "Oculoplastics and Orbit",
        "OCT": "Ophthalmic Imaging",
        "fundus": "Ophthalmic Imaging",
        "anti-VEGF": "Ophthalmic Pharmacology",
        "intravitreal": "Ophthalmic Pharmacology",
        "blind": "Community Eye Health",
        "trachoma": "Community Eye Health",
        "vitamin A": "Community Eye Health",
        "retinitis pigmentosa": "Ocular Genetics",
        "gene therapy": "Ocular Genetics",
        "retinal vein": "Retinal Vascular Occlusions",
        "retinal artery": "Retinal Vascular Occlusions",
        "retinal detachment": "Retinal Detachment and Vitreoretinal",
        "vitrectomy": "Retinal Detachment and Vitreoretinal",
        "LASIK": "Refractive Surgery",
        "refractive": "Refractive Surgery",
        "pterygium": "Ocular Surface Disease",
        "endophthalmitis": "Uveitis and Ocular Inflammation",
        "macular hole": "Retinal Detachment and Vitreoretinal",
        "Vogt-Koyanagi": "Uveitis and Ocular Inflammation",
        "Behcet": "Uveitis and Ocular Inflammation",
        "scleral buckling": "Retinal Detachment and Vitreoretinal",
        "ptosis": "Oculoplastics and Orbit",
        "nasolacrimal": "Oculoplastics and Orbit",
        "ocular trauma": "Community Eye Health",
        "deep learning": "Ophthalmic Imaging",
        "artificial intelligence": "Ophthalmic Imaging",
        "machine learning": "Ophthalmic Imaging",
        "teleophthalmology": "Community Eye Health",
        "angle closure": "Glaucoma",
        "central serous": "Retinal Detachment and Vitreoretinal",
        "optic disc": "Neuro-Ophthalmology",
        "blepharospasm": "Oculoplastics and Orbit",
        "keratoconus": "Corneal Diseases",
        "cross-linking": "Corneal Diseases",
        "leukocoria": "Leukocoria and White Pupil",
        "white pupil": "Leukocoria and White Pupil",
        "white reflex": "Leukocoria and White Pupil",
        "red reflex": "Leukocoria and White Pupil",
        "cherry red spot": "Cherry-Red Spot Differential",
        "cherry-red spot": "Cherry-Red Spot Differential",
        "central retinal artery": "Cherry-Red Spot Differential",
        "Tay-Sachs": "Cherry-Red Spot Differential",
        "Niemann-Pick": "Cherry-Red Spot Differential",
        "corneal opacity": "Corneal Opacity and Scarring",
        "corneal scar": "Corneal Opacity and Scarring",
        "leucoma": "Corneal Opacity and Scarring",
        "band keratopathy": "Anterior Segment Signs",
        "corneal dystrophy": "Anterior Segment Signs",
        "anterior segment": "Anterior Segment Signs",
        "iris lesion": "Anterior Segment Signs",
        "anisocoria": "Pupillary Abnormalities",
        "Horner": "Pupillary Abnormalities",
        "Marcus Gunn": "Pupillary Abnormalities",
        "pupillary defect": "Pupillary Abnormalities",
        "Argyll Robertson": "Pupillary Abnormalities",
        "Adie pupil": "Pupillary Abnormalities",
        "ocular emergency": "Ocular Emergency Signs",
        "eye emergency": "Ocular Emergency Signs",
        "chemical injury": "Ocular Emergency Signs",
        "globe rupture": "Ocular Emergency Signs",
        "hyphema": "Ocular Emergency Signs",
        "acute vision loss": "Ocular Emergency Signs",
        "ocular manifestation": "Systemic Disease Ocular Manifestations",
    }
    for kw, cat in keyword_map.items():
        if kw.lower() in query_lower:
            return cat
    return "General Ophthalmology"


def main():
    parser = argparse.ArgumentParser(description="Fetch ophthalmic articles for RAG knowledge base")
    parser.add_argument("--max-per-query", type=int, default=3000,
                        help="Max articles per PubMed query category (default: 3000)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_MD),
                        help="Output markdown path")
    parser.add_argument("--skip-europepmc", action="store_true",
                        help="Skip EuropePMC fetching")
    parser.add_argument("--skip-semantic-scholar", action="store_true",
                        help="Skip Semantic Scholar fetching")
    parser.add_argument("--skip-medlineplus", action="store_true",
                        help="Skip MedlinePlus fetching")
    args = parser.parse_args()

    seen_pmids: Set[str] = set()
    articles_by_category: Dict[str, List[Dict]] = {}
    source_counts = {"pubmed": 0, "europepmc": 0, "semantic_scholar": 0, "medlineplus": 0}

    def _add_articles(arts: List[Dict], category: str, source_name: str) -> int:
        """Add deduplicated articles to a category. Returns count added."""
        new = [a for a in arts if a["pmid"] not in seen_pmids]
        for a in new:
            seen_pmids.add(a["pmid"])
        if new:
            articles_by_category.setdefault(category, []).extend(new)
            source_counts[source_name] += len(new)
        return len(new)

    # ── 1. PubMed ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  PHASE 1: PubMed E-Utilities (20 categories)")
    print("=" * 70)

    for category, query in PUBMED_QUERIES.items():
        print(f"\n[PubMed] Searching: {category}...")
        pmids = pubmed_search(query, max_results=args.max_per_query)
        new_pmids = [p for p in pmids if p not in seen_pmids]
        seen_pmids.update(new_pmids)
        print(f"  Found {len(pmids)} PMIDs ({len(new_pmids)} new)")

        if new_pmids:
            articles = pubmed_fetch_abstracts(new_pmids)
            for a in articles:
                seen_pmids.add(a["pmid"])
            articles_by_category.setdefault(category, []).extend(articles)
            source_counts["pubmed"] += len(articles)
            print(f"  Fetched {len(articles)} abstracts")
        time.sleep(REQUEST_DELAY)

    pubmed_total = source_counts["pubmed"]
    print(f"\n[PubMed] Total: {pubmed_total} articles")

    # ── 2. EuropePMC ─────────────────────────────────────────────────────────
    if not args.skip_europepmc:
        print(f"\n{'=' * 70}")
        print("  PHASE 2: EuropePMC (20 queries, paginated)")
        print("=" * 70)

        for eq in EUROPEPMC_QUERIES:
            print(f"\n[EuropePMC] Searching: {eq}...")
            epmc_arts = europepmc_search(eq, max_results=50)
            category = _match_category(eq)
            added = _add_articles(epmc_arts, category, "europepmc")
            if added:
                print(f"  Added {added} new articles → {category}")
            time.sleep(REQUEST_DELAY)

        print(f"\n[EuropePMC] Total: {source_counts['europepmc']} articles")

    # ── 3. Semantic Scholar ──────────────────────────────────────────────────
    if not args.skip_semantic_scholar:
        print(f"\n{'=' * 70}")
        print("  PHASE 3: Semantic Scholar (30 queries)")
        print("=" * 70)

        s2_per_query = min(args.max_per_query, 100)  # S2 max per page is 100
        for sq in SEMANTIC_SCHOLAR_QUERIES:
            print(f"\n[S2] Searching: {sq}...")
            s2_arts = semantic_scholar_search(sq, max_results=s2_per_query)
            category = _match_category(sq)
            added = _add_articles(s2_arts, category, "semantic_scholar")
            if added:
                print(f"  Added {added} new articles → {category}")
            time.sleep(S2_DELAY)

        print(f"\n[Semantic Scholar] Total: {source_counts['semantic_scholar']} articles")

    # ── 4. MedlinePlus ───────────────────────────────────────────────────────
    if not args.skip_medlineplus:
        print(f"\n{'=' * 70}")
        print("  PHASE 4: MedlinePlus Health Topics")
        print("=" * 70)

        mlp_articles = fetch_medlineplus()
        added = _add_articles(mlp_articles, "General Ophthalmology", "medlineplus")
        print(f"  Added {added} MedlinePlus articles")

    # ── 5. Write Markdown ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  WRITING MARKDOWN")
    print("=" * 70)

    total = write_markdown(articles_by_category, Path(args.output))

    # Summary
    print(f"\n{'=' * 70}")
    print("  FETCH SUMMARY")
    print("=" * 70)
    print(f"\n  By Source:")
    for src, cnt in source_counts.items():
        print(f"    {src:20s}: {cnt:>5d}")
    print(f"\n  By Category:")
    for cat, arts in sorted(articles_by_category.items()):
        print(f"    {cat:45s}: {len(arts):>5d}")
    print(f"  {'─' * 52}")
    print(f"  {'TOTAL':45s}: {total:>5d} unique articles")
    print(f"\n  Output: {args.output}")
    print(f"\n  Next steps:")
    print(f"    python scripts/chunk_data.py {args.output}")
    print(f"    python scripts/ingest_db.py")


if __name__ == "__main__":
    main()
