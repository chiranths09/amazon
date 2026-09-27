"""
normalization.py — Text normalization for business names and addresses.

Design principles:
  - Country is treated as an OPEN STRING (never hardcoded to a fixed set).
  - Abbreviation expansion is data-driven via a nested dict:
      ABBREV_DICT["_base"] = language-agnostic expansions
      ABBREV_DICT[<country_string>] = country-specific overrides / additions
    Any unseen country (e.g., "France") automatically falls back to _base.
  - Expansions are applied longest-first to avoid partial-match bugs
    (e.g., "St" should not fire inside "Street").
"""

import re
import unicodedata
import logging
from functools import lru_cache
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abbreviation dictionaries — data-driven, country-aware, extensible
# ---------------------------------------------------------------------------
# Keys are lower-cased tokens; values are the expanded form (also lower-case).
# Add new entries here to improve coverage without touching any other code.

ABBREV_DICT: dict[str, dict[str, str]] = {
    # ----- language-agnostic / English base -----
    "_base": {
        # Legal entity suffixes
        "corp": "corporation",
        "co": "company",
        "inc": "incorporated",
        "ltd": "limited",
        "llc": "limited liability company",
        "llp": "limited liability partnership",
        "lp": "limited partnership",
        "plc": "public limited company",
        "pvt": "private",
        "pte": "private",
        "pty": "proprietary",
        # Connectors
        "&": "and",
        "n": "and",          # "n" between words, e.g. "fish n chips"
        # Street / address
        "rd": "road",
        "st": "street",
        "ave": "avenue",
        "blvd": "boulevard",
        "dr": "drive",
        "ln": "lane",
        "ct": "court",
        "pl": "place",
        "pkwy": "parkway",
        "hwy": "highway",
        "fwy": "freeway",
        "sq": "square",
        "bldg": "building",
        "ste": "suite",
        "apt": "apartment",
        "dept": "department",
        "fl": "floor",
        "no": "number",
        "num": "number",
        # Directions
        "n": "north",
        "s": "south",
        "e": "east",
        "w": "west",
        "ne": "northeast",
        "nw": "northwest",
        "se": "southeast",
        "sw": "southwest",
        # US-specific common
        "intl": "international",
        "natl": "national",
        "mgmt": "management",
        "mfg": "manufacturing",
        "svc": "service",
        "svcs": "services",
        "grp": "group",
        "assoc": "associates",
        "assn": "association",
        "univ": "university",
        "tech": "technology",
        "sys": "systems",
        "soln": "solution",
        "solns": "solutions",
    },
    # ----- India-specific additions / overrides -----
    "India": {
        "pvt": "private",
        "ltd": "limited",
        "co": "company",
        "bros": "brothers",
        "engg": "engineering",
        "mfrs": "manufacturers",
        "distt": "district",
        "dist": "district",
        "vill": "village",
        "nagar": "nagar",   # keep as-is (common proper noun)
        "indl": "industrial",
        "est": "estate",
        "mkt": "market",
        "sec": "sector",
        "ph": "phase",
    },
    # ----- France-specific additions -----
    # Country appears only in test; ensure graceful expansion
    "France": {
        "sarl": "societe a responsabilite limitee",
        "sas": "societe par actions simplifiee",
        "sa": "societe anonyme",
        "sci": "societe civile immobiliere",
        "r": "rue",
        "av": "avenue",
        "bd": "boulevard",
        "imp": "impasse",
        "cite": "cite",
    },
    # Add more countries here without changing any other code:
    # "Germany": { "gmbh": "gesellschaft mit beschraenkter haftung", ... }
}

# Pre-build merged expansion tables per country (base + country-specific).
# Keys are sorted longest-first so we try longer tokens first during expansion.
_EXPANSION_CACHE: dict[str, list[tuple[str, str]]] = {}


def _get_expansions(country: Optional[str]) -> list[tuple[str, str]]:
    """
    Return ordered (token, expansion) pairs for a given country string.
    Falls back to base expansions if country is unknown or None.
    Longest tokens are tried first to avoid partial replacements.
    """
    key = country if country else "_base"
    if key not in _EXPANSION_CACHE:
        merged = dict(ABBREV_DICT["_base"])  # start with base
        if country and country in ABBREV_DICT:
            merged.update(ABBREV_DICT[country])  # override/add country-specific
        # Sort longest token first
        ordered = sorted(merged.items(), key=lambda x: -len(x[0]))
        _EXPANSION_CACHE[key] = ordered
    return _EXPANSION_CACHE[key]


# Regex to split text into word tokens (alphanumeric sequences)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str, country: Optional[str] = None) -> str:
    """
    Normalize a single business name or address string.

    Steps:
      1. Unicode NFKD decomposition → ASCII-compatible form
      2. Lowercase
      3. Replace punctuation with spaces (keep alphanumeric + space)
      4. Expand abbreviations (longest-first, word-boundary-aware)
      5. Collapse whitespace

    Parameters
    ----------
    text : str
        Raw text value (name or address).
    country : str, optional
        Country string from the same record. Used to select the right
        abbreviation set. Any value is accepted (open-string — no fixed set).

    Returns
    -------
    str  Normalized text.
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Unicode normalization (handles accented chars, ligatures, etc.)
    text = unicodedata.normalize("NFKD", text)
    # Keep only ASCII printable characters (drops Devanagari, etc.)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Step 2: Lowercase
    text = text.lower()

    # Step 3: Replace non-alphanumeric with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Step 4: Token-level abbreviation expansion
    expansions = _get_expansions(country)
    tokens = text.split()
    expanded = []
    for tok in tokens:
        replaced = tok
        for abbr, full in expansions:
            if tok == abbr:
                replaced = full
                break
        expanded.append(replaced)
    text = " ".join(expanded)

    # Step 5: Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns to a source DataFrame.

    Adds:
      norm_name    — normalized business_name
      norm_address — normalized business_address
      norm_combined — name + " " + address (useful for single-field TF-IDF)

    The 'country' column is used per-row for abbreviation expansion but is
    NOT transformed — it stays as the original string.

    Parameters
    ----------
    df : pd.DataFrame with columns [entity_id, business_name, business_address, country]

    Returns
    -------
    df with three new columns added in-place.
    """
    logger.info(f"Normalizing {len(df):,} records ...")

    # Vectorized apply; country is passed per-row for the open-string treatment
    df["norm_name"] = [
        normalize_text(name, country)
        for name, country in zip(df["business_name"], df["country"])
    ]
    df["norm_address"] = [
        normalize_text(addr, country)
        for addr, country in zip(df["business_address"], df["country"])
    ]
    df["norm_combined"] = df["norm_name"] + " " + df["norm_address"]

    logger.info("  Normalization complete.")
    return df
