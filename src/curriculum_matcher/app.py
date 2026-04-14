# -*- coding: utf-8 -*-
"""
Enhanced Curriculum Matcher v3.1.2
- Profile toggle (fast=MiniLM; accurate=MiniLM recall + MPNet rerank)
- Precomputed catalog embeddings for precise model
- Input-embedding caching per row
- Robust year parsing (e.g., "2019-2021")
- human_match_* outputs restored
"""

import pandas as pd
import numpy as np
import re
import threading
import os
import sys
import time
from datetime import datetime
import json
import argparse
import unicodedata
from pathlib import Path

from .llm_rerank import (
    build_catalog_lookup,
    build_prompt_record,
    build_responses_api_body,
    extract_json_object,
    validate_and_repair_rerank_response,
)
from .match_layers import choose_match_strategy, should_try_repairs
from .openai_batch import (
    create_batch,
    download_file_content,
    retrieve_batch,
    upload_batch_file,
)
from .qa import build_human_match_review

tk = None
filedialog = None
messagebox = None
ttk = None


# =========================
#   Core Matcher (V3.1.2)
# =========================
class EnhancedCurriculumMatcherV312:
    """
    Two-stage retrieval (BM25 + semantic) with profile toggle and robust scoring.
    """

    def __init__(
        self,
        log_callback=print,
        weights=None,
        retrieval_experiment=None,
        rerank_experiment=None,
        cross_encoder_model=None,
    ):
        self.log = log_callback
        self.model_fast = None
        self.model_precise = None
        self.cross_encoder = None
        self.bm25 = None
        self.bm25_char_ngrams = None
        self.catalog_df = None
        self.profile = "fast"  # overridden by GUI/CLI
        retrieval_experiment = (
            retrieval_experiment
            if retrieval_experiment is not None
            else os.getenv("CURRICULUM_MATCHER_RETRIEVAL_EXPERIMENT", "")
        ).strip().lower()
        # D3 established char n-gram retrieval as the accepted stage-1 baseline.
        self.retrieval_experiment = retrieval_experiment or "char_ngram"
        rerank_experiment = (
            rerank_experiment
            if rerank_experiment is not None
            else os.getenv("CURRICULUM_MATCHER_RERANK_EXPERIMENT", "")
        ).strip().lower()
        self.rerank_experiment = rerank_experiment or "default"
        self.cross_encoder_model_name = (
            cross_encoder_model
            if cross_encoder_model is not None
            else os.getenv(
                "CURRICULUM_MATCHER_CROSS_ENCODER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            )
        ).strip()
        self.char_ngram_size = 3

        # Column names
        self.INPUT_PRODUCT_NAME = "product_name_raw"
        self.INPUT_PUBLISHER = "publisher_raw"
        self.CATALOG_ID = "product_identifier"
        self.CATALOG_PRODUCT_NAME = "product_name"
        self.CATALOG_PUBLISHER = "publisher"
        self.CATALOG_PUBLISHER_PRIOR = "publisher_prior"
        self.CATALOG_YEAR = "copyright_year"
        self.INPUT_GRADE = "grade"
        self.CATALOG_GRADES = "intended_grades"
        self.CATALOG_SERIES = "series"
        self.CATALOG_SUBJECT = "subject_level1"

        # Alias maps
        self.publisher_aliases = {
            # Keep Pearson distinct so current Pearson Education rows can still
            # match directly; legacy Savvas/Pearson ties are already handled
            # via catalog publisher_prior.
            "benchmark education": "benchmark education company",
            "holt mcdougal": "hmh",
            "holt mcdougall": "hmh",
            "holt rinehart and winston": "hmh",
            "houghton mifflin": "hmh",
            "houghton mifflin harcourt": "hmh",
            "pearson education": "pearson education",
            "pearson prentice hall": "pearson education",
            "prentice hall": "pearson education",
            "pearson scott foresman": "pearson education",
            "scott foresman": "pearson education",
            "pearson school": "pearson education",
            "mheducation": "mcgraw hill",
            "mgh": "mcgraw hill",
            "mcgraw-hill": "mcgraw hill",
            "mcgraw hill education": "mcgraw hill",
            "mcgraw hill llc": "mcgraw hill",
            "mcgraw hill school education": "mcgraw hill",
            "mcgraw hill school education llc": "mcgraw hill",
            "mcgraw hill wright group": "mcgraw hill",
            "glencoe mcgraw hill": "mcgraw hill",
            "savvas": "savvas learning company",
            "savvas learning": "savvas learning company",
            "ngl cengage": "national geographic learning cengage",
            "natgeo cengage": "national geographic learning cengage",
        }
        self.subject_aliases = {
            "ela": "english language arts",
            "lang arts": "english language arts",
            "maths": "mathematics",
        }
        self.product_title_alias_patterns = [
            (
                re.compile(r"\bcaaspp\b"),
                "california assessment of student performance and progress caaspp",
            ),
            (
                re.compile(r"\balternate\s+elpac\b"),
                "alternate english language proficiency assessments for california elpac",
            ),
            (
                re.compile(r"\belpac\b"),
                "english language proficiency assessments for california elpac",
            ),
            (
                re.compile(r"\bnaep\b"),
                "national assessment of educational progress naep",
            ),
            (
                re.compile(r"\belpa\s*21\b"),
                "english language proficiency assessment for the 21st century elpa21",
            ),
            (
                re.compile(r"\bnwea\s+map\b"),
                "northwest evaluation association map suite nwea map",
            ),
            (
                re.compile(r"\bstaar\b"),
                "state of texas assessments of academic readiness staar",
            ),
            (
                re.compile(r"\btelpas\b"),
                "texas english language proficiency assessment system telpas",
            ),
            (
                re.compile(r"\bteks\b"),
                "texas essential knowledge and skills teks",
            ),
            (
                re.compile(r"\bwida\s+access\b"),
                "access for ells wida access",
            ),
            (
                re.compile(r"\bdra\b"),
                "developmental reading assessment dra",
            ),
            (
                re.compile(r"\bga\s+ed\b"),
                "georgia edition",
            ),
            (
                re.compile(r"\bca\s+studies\b"),
                "california studies",
            ),
            (
                re.compile(r"\bsc\s+1st\s+edition\b"),
                "south carolina first edition",
            ),
            (
                re.compile(r"\bflorida'?s?\s+b\.?e\.?s\.?t\.?\b"),
                "florida best",
            ),
            (
                re.compile(r"\bts\s*gold\b"),
                "teaching strategies gold ts gold",
            ),
            (
                re.compile(r"\btsg\b"),
                "teaching strategies gold tsg",
            ),
        ]

        # Scoring weights (you can tune via UI)
        self.weights = (
            weights
            if weights
            else {
                "name_semantic": 0.40,
                "name_fuzzy": 0.15,
                "publisher": 0.25,
                "grade": 0.15,
                "year": 0.05,
            }
        )

        # Lazy import placeholders
        self._rapidfuzz = None
        self._SentenceTransformer = None
        self._CrossEncoder = None
        self._cosine_similarity = None
        self._BM25Okapi = None
        self.low_confidence_threshold = 0.55
        self.repair_replacement_margin = 0.03
        self.strategy_repair_replacement_margins = {
            "title_plus_publisher": 0.12,
        }
    # ---------- Lazy imports ----------
    def _ensure_libs(self):
        if self._rapidfuzz is None:
            import rapidfuzz as _rf

            self._rapidfuzz = _rf
        if self._cosine_similarity is None:
            from sklearn.metrics.pairwise import cosine_similarity as _cs

            self._cosine_similarity = _cs
        if self._SentenceTransformer is None:
            from sentence_transformers import SentenceTransformer as _ST

            self._SentenceTransformer = _ST
        if self._CrossEncoder is None:
            from sentence_transformers import CrossEncoder as _CE

            self._CrossEncoder = _CE
        if self._BM25Okapi is None:
            try:
                from rank_bm25 import BM25Okapi as _BM25
            except Exception as e:
                raise RuntimeError(
                    "rank_bm25 is required. Install with: pip install rank-bm25"
                ) from e
            self._BM25Okapi = _BM25

    # ---------- Normalization helpers ----------
    def _normalize(self, text, for_semantic=False):
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        s = str(text).lower()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")

        s = s.replace("&", " and ").replace("pre-k", "prek")
        s = re.sub(r"\bgr\b", "grade", s)

        for k, v in self.subject_aliases.items():
            s = re.sub(rf"\b{k}\b", v, s)

        if for_semantic:
            s = re.sub(r"[^\w\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
        else:
            s = re.sub(r"[^\w\s]", "", s)
            s = re.sub(r"\s+", "", s).strip()
        return s

    def _normalize_product_title(self, text, for_semantic=False):
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        s = str(text).lower()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
        for pattern, replacement in self.product_title_alias_patterns:
            updated = pattern.sub(replacement, s)
            if updated != s:
                s = updated
                break
        return self._normalize(s, for_semantic=for_semantic)

    def _build_char_ngram_tokens(self, text, n=None):
        normalized = self._normalize_product_title(text, for_semantic=True)
        if not normalized:
            return []
        size = n or self.char_ngram_size
        padded = f"{'_' * (size - 1)}{normalized.replace(' ', '_')}{'_' * (size - 1)}"
        return [padded[idx : idx + size] for idx in range(len(padded) - size + 1)]

    def _publisher_canonical(self, text):
        s = self._normalize(text, for_semantic=True)
        s = re.sub(r"\b(inc|llc|ltd|co|corp|company)\b\.?", "", s).strip()
        s = re.sub(r"\s+", " ", s).strip()
        for k, v in self.publisher_aliases.items():
            if s == k:
                return v
        return s

    def _build_cross_encoder_query_text(self, record):
        title = self._normalize_product_title(record.get(self.INPUT_PRODUCT_NAME), True)
        publisher = self._publisher_canonical(record.get(self.INPUT_PUBLISHER))
        grade = self._normalize(record.get(self.INPUT_GRADE), for_semantic=True)
        return " | ".join([part for part in [title, publisher, grade] if part])

    def _build_cross_encoder_candidate_text(self, candidate):
        title = self._normalize_product_title(candidate.get(self.CATALOG_PRODUCT_NAME), True)
        series = self._normalize_product_title(candidate.get(self.CATALOG_SERIES), True)
        publisher = self._publisher_canonical(candidate.get(self.CATALOG_PUBLISHER))
        publisher_prior = self._publisher_canonical(
            candidate.get(self.CATALOG_PUBLISHER_PRIOR)
        )
        grades = self._normalize(candidate.get(self.CATALOG_GRADES), for_semantic=True)
        year = self._normalize(candidate.get(self.CATALOG_YEAR), for_semantic=True)
        return " | ".join(
            [
                part
                for part in [title, series, publisher, publisher_prior, grades, year]
                if part
            ]
        )

    def _rerank_with_cross_encoder(self, record, results, candidate_lookup):
        if self.rerank_experiment != "cross_encoder" or not results:
            return results

        query_text = self._build_cross_encoder_query_text(record)
        if not query_text:
            return results

        pairs = []
        for result in results:
            candidate = candidate_lookup.get(result["catalog_id"])
            candidate_text = self._build_cross_encoder_candidate_text(candidate)
            pairs.append((query_text, candidate_text))

        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        reranked = []
        for result, score in zip(results, scores):
            updated = dict(result)
            updated["cross_encoder_score"] = float(score)
            reranked.append(updated)
        reranked.sort(
            key=lambda item: (item["cross_encoder_score"], item["final_score"]),
            reverse=True,
        )
        return reranked

    # ---------- Year helpers ----------
    YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

    def _extract_first_year(self, value):
        """Return the first 4-digit year found; else None."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, (int, np.integer)):
            y = int(value)
            return y if 1800 <= y <= 2100 else None
        m = self.YEAR_RE.search(str(value))
        return int(m.group(0)) if m else None

    def _extract_year(self, text):
        return self._extract_first_year(text)

    def _parse_grade_range(self, grade_str):
        if grade_str is None or (isinstance(grade_str, float) and np.isnan(grade_str)):
            return None, None
        s = str(grade_str).lower().strip().replace("pre-k", "prek")
        if s in {"p", "pk", "prek", "k", "kindergarten"}:
            return 0, 0
        nums = re.findall(r"\d+", s)
        if not nums:
            return None, None
        nums = [int(n) for n in nums]
        return min(nums), max(nums)

    # ---------- Models ----------
    def load_models(self, profile="fast"):
        self._ensure_libs()
        self.profile = profile
        if profile == "fast":
            fast_model = "all-MiniLM-L6-v2"
            precise_model = "all-MiniLM-L6-v2"
        else:
            fast_model = "all-MiniLM-L6-v2"
            precise_model = "all-mpnet-base-v2"

        if self.model_fast is None:
            self.log(f"Loading recall model: {fast_model} ...")
            self.model_fast = self._SentenceTransformer(fast_model)
        if self.model_precise is None or (
            profile == "fast" and self.model_precise != self.model_fast
        ):
            if profile == "fast":
                self.model_precise = self.model_fast
            else:
                self.log(f"Loading precise model: {precise_model} ...")
                self.model_precise = self._SentenceTransformer(precise_model)
        if self.rerank_experiment == "cross_encoder" and self.cross_encoder is None:
            self.log(f"Loading cross-encoder reranker: {self.cross_encoder_model_name} ...")
            self.cross_encoder = self._CrossEncoder(self.cross_encoder_model_name)
        self.log(f"Models loaded (profile={self.profile}).")

    def prepare_catalog(self, catalog_df):
        """
        Pre-clean, build search text, encode recall embeddings, BM25,
        and precise embeddings (if MPNet).
        """
        self.catalog_df = catalog_df.copy()

        def build_search_text(row):
            name = self._normalize_product_title(row.get(self.CATALOG_PRODUCT_NAME), True)
            series = self._normalize_product_title(row.get(self.CATALOG_SERIES), True)
            year = str(row.get(self.CATALOG_YEAR) or "")
            return " ".join([p for p in [name, series, year] if p])

        self.log("Pre-cleaning catalog and building search_text...")
        self.catalog_df["search_text"] = self.catalog_df.apply(
            build_search_text, axis=1
        )

        self.log("Encoding catalog embeddings for recall (MiniLM)...")
        self.catalog_df["embedding_fast"] = list(
            self.model_fast.encode(
                self.catalog_df["search_text"].tolist(),
                batch_size=64,
                show_progress_bar=False,
            )
        )

        self.log("Building BM25 index...")
        tokenized = [s.split() for s in self.catalog_df["search_text"]]
        self.bm25 = self._BM25Okapi(tokenized)
        self.bm25_char_ngrams = None
        if self.retrieval_experiment == "char_ngram":
            self.log("Building character n-gram BM25 index...")
            self.catalog_df["search_text_char_ngrams"] = self.catalog_df[
                "search_text"
            ].apply(self._build_char_ngram_tokens)
            self.bm25_char_ngrams = self._BM25Okapi(
                self.catalog_df["search_text_char_ngrams"].tolist()
            )

        # Precompute precise embeddings if using MPNet
        if self.model_precise is not self.model_fast:
            self.log("Encoding catalog embeddings for precise re-ranking (MPNet)...")
            self.catalog_df["embedding_precise"] = list(
                self.model_precise.encode(
                    self.catalog_df["search_text"].tolist(),
                    batch_size=32,
                    show_progress_bar=False,
                )
            )
        else:
            if "embedding_precise" in self.catalog_df.columns:
                del self.catalog_df["embedding_precise"]

    # ---------- Scoring components ----------
    def _name_scores(self, input_sem_vec, cand_sem_vec, input_text, cand_text):
        sem = float(self._cosine_similarity(input_sem_vec, [cand_sem_vec])[0][0])
        fuzz = (
            self._rapidfuzz.fuzz.token_set_ratio(
                self._normalize_product_title(input_text),
                self._normalize_product_title(cand_text),
            )
            / 100.0
        )
        return sem, fuzz

    def _publisher_score(self, input_pub, cand_pub, cand_pub_prior=None):
        if not input_pub and not cand_pub:
            return 0.0
        inp = self._publisher_canonical(input_pub)
        cand = self._publisher_canonical(cand_pub)
        prior = self._publisher_canonical(cand_pub_prior) if cand_pub_prior else ""
        score_current = self._rapidfuzz.fuzz.WRatio(inp, cand) / 100.0 if cand else 0.0
        score_prior = self._rapidfuzz.fuzz.WRatio(inp, prior) / 100.0 if prior else 0.0
        return max(score_current, score_prior)

    def _grade_score(self, input_grade, cand_grade_str):
        in_low, in_high = self._parse_grade_range(input_grade)
        c_low, c_high = self._parse_grade_range(cand_grade_str)
        if in_low is None or c_low is None:
            return 0.0
        if in_low > c_high:
            dist = in_low - c_high
        elif c_low > in_high:
            dist = c_low - in_high
        else:
            dist = 0
        return {0: 1.0, 1: 0.8, 2: 0.4}.get(dist, 0.0)

    def _year_score(self, input_year_val, cand_year_val):
        y1 = self._extract_first_year(input_year_val)
        y2 = self._extract_first_year(cand_year_val)
        if y1 is None or y2 is None:
            return 0.0
        return 1.0 if y1 == y2 else 0.0

    def _record_with_updates(self, record, **updates):
        clone = dict(record)
        clone.update(updates)
        return clone

    def _build_repair_variants(self, record):
        title = record.get(self.INPUT_PRODUCT_NAME)
        publisher = record.get(self.INPUT_PUBLISHER)
        variants = {}
        if title or publisher:
            variants["title_publisher_swapped"] = self._record_with_updates(
                record,
                product_name_raw="" if pd.isna(publisher) else publisher,
                publisher_raw="" if pd.isna(title) else title,
            )
            if publisher:
                variants["publisher_as_title"] = self._record_with_updates(
                    record,
                    product_name_raw=publisher,
                )
            if title and publisher:
                variants["title_plus_publisher"] = self._record_with_updates(
                    record,
                    product_name_raw=f"{title} {publisher}".strip(),
                )
        return variants

    def _score_candidate(self, record, cand, input_fast_vec, input_precise_vec=None):
        use_precise = input_precise_vec is not None and "embedding_precise" in cand
        cand_sem_vec = cand["embedding_precise"] if use_precise else cand["embedding_fast"]
        input_sem_vec = input_precise_vec if use_precise else input_fast_vec

        sem, fuzz = self._name_scores(
            input_sem_vec,
            cand_sem_vec,
            record.get(self.INPUT_PRODUCT_NAME),
            cand["search_text"],
        )
        pub = self._publisher_score(
            record.get(self.INPUT_PUBLISHER),
            cand.get(self.CATALOG_PUBLISHER),
            cand.get(self.CATALOG_PUBLISHER_PRIOR),
        )
        grade = self._grade_score(
            record.get(self.INPUT_GRADE), cand.get(self.CATALOG_GRADES)
        )
        year = self._year_score(
            record.get(self.INPUT_PRODUCT_NAME) or record.get(self.INPUT_PUBLISHER),
            cand.get(self.CATALOG_YEAR),
        )
        final = (
            self.weights["name_semantic"] * sem
            + self.weights["name_fuzzy"] * fuzz
            + self.weights["publisher"] * pub
            + self.weights["grade"] * grade
            + self.weights["year"] * year
        )
        return {
            "catalog_id": cand.get(self.CATALOG_ID),
            "final_score": float(final),
            "name_semantic": float(sem),
            "name_fuzzy": float(fuzz),
            "publisher": float(pub),
            "grade": float(grade),
            "year": float(year),
        }

    def _blend_stage1_scores(self, bm25_scores, sem_scores_fast, char_ngram_scores=None):
        if self.retrieval_experiment == "char_ngram" and char_ngram_scores is not None:
            return 0.25 * bm25_scores + 0.5 * sem_scores_fast + 0.25 * char_ngram_scores
        return 0.5 * bm25_scores + 0.5 * sem_scores_fast

    # ---------- Matching ----------
    def _match_record_once(self, record, topn_stage1=60, topn_final=3):
        if self.catalog_df is None or self.bm25 is None:
            raise RuntimeError("Catalog not prepared. Call prepare_catalog() first.")

        search_text = self._normalize_product_title(
            record.get(self.INPUT_PRODUCT_NAME), True
        )
        if not search_text:
            return []

        # Stage 1 recall: BM25 + MiniLM semantic
        bm25_scores = self.bm25.get_scores(search_text.split())
        input_fast_vec = self.model_fast.encode([search_text])  # cached per row
        cat_fast = np.vstack(self.catalog_df["embedding_fast"].values)
        sem_scores_fast = self._cosine_similarity(input_fast_vec, cat_fast)[0]
        char_ngram_scores = None
        if self.bm25_char_ngrams is not None:
            char_ngram_scores = self.bm25_char_ngrams.get_scores(
                self._build_char_ngram_tokens(search_text)
            )
        combined = self._blend_stage1_scores(
            bm25_scores, sem_scores_fast, char_ngram_scores
        )

        recall_k = 40 if self.profile == "fast" else 60
        top_idx = np.argsort(combined)[::-1][:recall_k]

        # Stage 2 rerank: precise model if available
        use_precise = "embedding_precise" in self.catalog_df.columns
        if use_precise:
            input_precise_vec = self.model_precise.encode([search_text])

        results = []
        candidate_lookup = {}

        for idx in top_idx:
            cand = self.catalog_df.iloc[idx]
            sem, fuzz = self._name_scores(
                input_precise_vec if use_precise else input_fast_vec,
                cand["embedding_precise"] if use_precise else cand["embedding_fast"],
                record.get(self.INPUT_PRODUCT_NAME),
                cand["search_text"],
            )

            # hard negative cutoff
            if sem < 0.70 and fuzz < 0.55:
                continue

            scored_candidate = self._score_candidate(
                record,
                cand,
                input_fast_vec,
                input_precise_vec=input_precise_vec if use_precise else None,
            )
            candidate_lookup[scored_candidate["catalog_id"]] = cand
            results.append(scored_candidate)

        results.sort(key=lambda x: x["final_score"], reverse=True)
        results = self._rerank_with_cross_encoder(record, results, candidate_lookup)
        return results[:topn_final]

    def match_record_with_repairs(
        self,
        record,
        topn_stage1=60,
        topn_final=3,
        confidence_threshold=None,
        replacement_margin=None,
    ):
        confidence_threshold = (
            self.low_confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        replacement_margin = (
            self.repair_replacement_margin
            if replacement_margin is None
            else replacement_margin
        )

        primary_matches = self._match_record_once(
            record, topn_stage1=topn_stage1, topn_final=topn_final
        )
        fallback_results = {}
        if should_try_repairs(primary_matches, confidence_threshold):
            for strategy_name, variant_record in self._build_repair_variants(record).items():
                fallback_results[strategy_name] = self._match_record_once(
                    variant_record, topn_stage1=topn_stage1, topn_final=topn_final
                )

        result = choose_match_strategy(
            "primary",
            primary_matches,
            fallback_results,
            confidence_threshold=confidence_threshold,
            replacement_margin=replacement_margin,
            strategy_replacement_margins=self.strategy_repair_replacement_margins,
        )
        result["repair_attempted"] = bool(fallback_results)
        return result

    def match_record(self, record, topn_stage1=60, topn_final=3):
        return self.match_record_with_repairs(
            record, topn_stage1=topn_stage1, topn_final=topn_final
        )["matches"]


# =========================
#          GUI
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = PROJECT_ROOT / "matcher_settings.json"


class TeeLogStream:
    def __init__(self, original_stream, callback):
        self.original_stream = original_stream
        self.callback = callback
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, text):
        if not text:
            return 0
        with self._lock:
            self.original_stream.write(text)
            self.original_stream.flush()
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                cleaned = line.rstrip()
                if cleaned:
                    self.callback(cleaned)
        return len(text)

    def flush(self):
        with self._lock:
            self.original_stream.flush()
            cleaned = self._buffer.rstrip()
            self._buffer = ""
            if cleaned:
                self.callback(cleaned)

    def isatty(self):
        return getattr(self.original_stream, "isatty", lambda: False)()

    @property
    def encoding(self):
        return getattr(self.original_stream, "encoding", "utf-8")


class MatcherApp:
    def __init__(self, headless=False, profile="fast"):
        self.headless = headless
        self.root = None
        self.matcher = EnhancedCurriculumMatcherV312(
            log_callback=self.log if not headless else print
        )
        self.profile = profile
        self.worker_buttons = []
        self.default_benchmark_file = (
            PROJECT_ROOT / "benchmarks/gold/historical_07122025_representative_1000.csv"
        )
        self.default_rerank_records = (
            PROJECT_ROOT
            / "benchmarks/outputs/historical_07122025_representative_1000_fast_top10_records.csv"
        )
        self.default_rerank_output_dir = PROJECT_ROOT / "benchmarks/outputs"
        self.default_batch_runs_dir = self.default_rerank_output_dir / "openai_batch_runs"
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._streams_redirected = False

        if not self.headless:
            self._ensure_tkinter()
            self.root = tk.Tk()
            self._setup_gui()
            self._redirect_process_output()

    def _ensure_tkinter(self):
        global tk, filedialog, messagebox, ttk
        if tk is None:
            import tkinter as _tk
            from tkinter import filedialog as _filedialog
            from tkinter import messagebox as _messagebox
            from tkinter import ttk as _ttk

            tk = _tk
            filedialog = _filedialog
            messagebox = _messagebox
            ttk = _ttk

    # -------- logging helpers --------
    def log(self, message):
        if self.root:
            self.root.after(0, lambda: self._log_message(message))
        else:
            print(message)

    def _log_message(self, message):
        rendered = f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
        for widget_name in ("matcher_log_text", "rerank_log_text"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.insert(tk.END, rendered)
                widget.see(tk.END)

    def _redirect_process_output(self):
        if self.headless or self._streams_redirected:
            return
        sys.stdout = TeeLogStream(self._original_stdout, self.log)
        sys.stderr = TeeLogStream(self._original_stderr, self.log)
        self._streams_redirected = True

    def _restore_process_output(self):
        if not self._streams_redirected:
            return
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._streams_redirected = False

    def _set_status_text(self, message):
        if self.root:
            self.root.after(0, lambda: self.status_label.config(text=message))

    def _set_progress_value(self, value, message=None):
        def update_gui():
            self.progress_var.set(max(0.0, min(100.0, float(value))))
            if message:
                self.status_label.config(text=message)

        if self.root:
            self.root.after(0, update_gui)

    def _set_summary_text(self, message):
        if self.root and hasattr(self, "batch_summary_var"):
            self.root.after(0, lambda: self.batch_summary_var.set(message))

    def _set_worker_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"

        def update_gui():
            for button in self.worker_buttons:
                button.config(state=state)

        if self.root:
            self.root.after(0, update_gui)

    def run(self):
        if self.headless:
            self._run_headless_mode()
        else:
            self.root.mainloop()

    # -------- UI setup --------
    def _setup_gui(self):
        self.root.title("Enhanced Curriculum Matcher (V3.1.2)")
        self.root.geometry("1080x900")

        # --- UI Variables ---
        self.input_file = tk.StringVar()
        self.catalog_file = tk.StringVar()
        self.output_dir = tk.StringVar()

        self.weight_name_sem_var = tk.StringVar()
        self.weight_name_fuzzy_var = tk.StringVar()
        self.weight_pub_var = tk.StringVar()
        self.weight_grade_var = tk.StringVar()
        self.weight_year_var = tk.StringVar()
        self.profile_var = tk.StringVar(value=self.profile)
        self.batch_records_file = tk.StringVar(
            value=str(self.default_rerank_records)
            if self.default_rerank_records.exists()
            else ""
        )
        self.batch_output_dir = tk.StringVar(value=str(self.default_rerank_output_dir))
        self.batch_runs_dir = tk.StringVar(value=str(self.default_batch_runs_dir))
        self.batch_benchmark_file = tk.StringVar(value=str(self.default_benchmark_file))
        self.batch_prompt_jsonl = tk.StringVar()
        self.batch_prompt_summary_json = tk.StringVar()
        self.batch_request_jsonl = tk.StringVar()
        self.batch_run_name = tk.StringVar(value="historical-top10-gpt54mini-medium-batch")
        self.batch_model = tk.StringVar(value="gpt-5.4-mini")
        self.batch_reasoning_effort = tk.StringVar(value="medium")
        self.batch_shortlist_size = tk.StringVar(value="10")
        self.batch_poll_interval = tk.StringVar(value="30")
        self.batch_summary_var = tk.StringVar(
            value="Batch summary will appear here after a run."
        )

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=0)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=0)

        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        matcher_tab = ttk.Frame(notebook, padding="8")
        rerank_tab = ttk.Frame(notebook, padding="8")
        notebook.add(matcher_tab, text="Matcher")
        notebook.add(rerank_tab, text="LLM Batch Rerank")

        self._setup_matcher_tab(matcher_tab)
        self._setup_rerank_tab(rerank_tab)

        status_frame = ttk.LabelFrame(main_frame, text="Status")
        status_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        self.status_label = ttk.Label(
            status_frame, text="Select files for matcher or rerank workflow."
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=5)

        for variable in (
            self.batch_records_file,
            self.batch_output_dir,
            self.batch_shortlist_size,
        ):
            variable.trace_add("write", self._refresh_rerank_paths)
        self.batch_model.trace_add("write", self._refresh_run_name)
        self.batch_reasoning_effort.trace_add("write", self._refresh_run_name)
        self.batch_shortlist_size.trace_add("write", self._refresh_run_name)
        self.catalog_file.trace_add("write", self._refresh_rerank_ready_state)
        self.batch_records_file.trace_add("write", self._refresh_rerank_ready_state)
        self.batch_output_dir.trace_add("write", self._refresh_rerank_ready_state)
        for variable in (
            self.input_file,
            self.catalog_file,
            self.output_dir,
            self.batch_records_file,
            self.batch_output_dir,
            self.batch_runs_dir,
            self.batch_run_name,
            self.batch_model,
            self.batch_reasoning_effort,
            self.batch_shortlist_size,
            self.batch_poll_interval,
        ):
            variable.trace_add("write", self._on_path_or_setting_changed)

        self._load_settings()
        self._refresh_run_name()
        self._refresh_rerank_paths()
        self._refresh_rerank_ready_state()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_matcher_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)

        file_frame = ttk.LabelFrame(parent, text="1. Select Files & Directory")
        file_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self._create_file_selector(file_frame, "Input Data File:", self.input_file)
        self._create_file_selector(
            file_frame, "Product Catalog File:", self.catalog_file
        )
        self._create_directory_selector(
            file_frame, "Output Directory:", self.output_dir
        )

        weights_frame = ttk.LabelFrame(parent, text="2. Configure Weights")
        weights_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self._create_weight_entry(
            weights_frame, "Name (semantic) Weight:", self.weight_name_sem_var
        )
        self._create_weight_entry(
            weights_frame, "Name (fuzzy) Weight:", self.weight_name_fuzzy_var
        )
        self._create_weight_entry(
            weights_frame, "Publisher Weight:", self.weight_pub_var
        )
        self._create_weight_entry(weights_frame, "Grade Weight:", self.weight_grade_var)
        self._create_weight_entry(weights_frame, "Year Weight:", self.weight_year_var)

        row = ttk.Frame(weights_frame)
        row.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(row, text="Profile:", width=25).pack(side=tk.LEFT)
        self.profile_combo = ttk.Combobox(
            row,
            textvariable=self.profile_var,
            values=["fast", "accurate"],
            width=12,
            state="readonly",
        )
        self.profile_combo.pack(side=tk.LEFT, padx=5)

        self.weights_sum_label = ttk.Label(
            weights_frame, text="Sum: 1.0", font=("Segoe UI", 9)
        )
        self.weights_sum_label.pack(pady=(0, 5))

        actions_frame = ttk.LabelFrame(parent, text="3. Run Matcher")
        actions_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.process_button = ttk.Button(
            actions_frame,
            text="Start Processing",
            command=self._start_processing_thread,
            state="disabled",
        )
        self.process_button.pack(anchor="w", padx=10, pady=10)
        self.worker_buttons.append(self.process_button)

        self.matcher_log_text = self._create_log_pane(parent, row=3)

    def _setup_rerank_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(5, weight=1)

        input_frame = ttk.LabelFrame(parent, text="1. Inputs")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self._create_file_selector(
            input_frame,
            "Matcher Records CSV:",
            self.batch_records_file,
        )
        self._create_file_selector(
            input_frame,
            "Product Catalog File:",
            self.catalog_file,
        )
        self._create_directory_selector(
            input_frame,
            "Artifact Output Directory:",
            self.batch_output_dir,
        )
        self._create_directory_selector(
            input_frame,
            "Batch Run Output Directory:",
            self.batch_runs_dir,
        )

        config_frame = ttk.LabelFrame(parent, text="2. Batch Settings")
        config_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self._create_labeled_entry(
            config_frame,
            "Run Name:",
            self.batch_run_name,
            readonly=False,
        )
        self._create_combo_row(
            config_frame,
            "Model:",
            self.batch_model,
            ["gpt-5.4-mini", "gpt-5.4", "gpt-5.2"],
        )
        self._create_combo_row(
            config_frame,
            "Reasoning Effort:",
            self.batch_reasoning_effort,
            ["low", "medium", "high"],
        )
        self._create_combo_row(
            config_frame,
            "Shortlist Size:",
            self.batch_shortlist_size,
            ["3", "10", "20"],
        )
        self._create_labeled_entry(
            config_frame,
            "Poll Interval (seconds):",
            self.batch_poll_interval,
            readonly=False,
        )

        artifact_frame = ttk.LabelFrame(parent, text="3. Derived Artifacts")
        artifact_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self._create_labeled_entry(
            artifact_frame,
            "Prompt Pack JSONL:",
            self.batch_prompt_jsonl,
        )
        self._create_labeled_entry(
            artifact_frame,
            "Prompt Summary JSON:",
            self.batch_prompt_summary_json,
        )
        self._create_labeled_entry(
            artifact_frame,
            "Batch Request JSONL:",
            self.batch_request_jsonl,
        )

        actions_frame = ttk.LabelFrame(parent, text="4. Actions")
        actions_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.build_prompt_button = ttk.Button(
            actions_frame,
            text="Build Prompt Pack",
            command=self._start_build_prompt_pack_thread,
        )
        self.build_prompt_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.prepare_requests_button = ttk.Button(
            actions_frame,
            text="Prepare Batch Request JSONL",
            command=self._start_prepare_batch_requests_thread,
        )
        self.prepare_requests_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.run_batch_button = ttk.Button(
            actions_frame,
            text="Run Full Batch Pipeline",
            command=self._start_run_batch_pipeline_thread,
        )
        self.run_batch_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.worker_buttons.extend(
            [
                self.build_prompt_button,
                self.prepare_requests_button,
                self.run_batch_button,
            ]
        )

        summary_frame = ttk.LabelFrame(parent, text="5. Batch Summary")
        summary_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        summary_label = ttk.Label(
            summary_frame,
            textvariable=self.batch_summary_var,
            justify=tk.LEFT,
        )
        summary_label.pack(fill=tk.X, padx=10, pady=10)

        self.rerank_log_text = self._create_log_pane(parent, row=5)

    def _create_log_pane(self, parent, row):
        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.grid(row=row, column=0, sticky="nsew", padx=5, pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        log_text = tk.Text(log_frame, height=18, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
        log_text.configure(yscrollcommand=scrollbar.set)
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return log_text

    def _create_weight_entry(self, parent, label_text, string_var):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)
        entry = ttk.Entry(frame, textvariable=string_var, width=15)
        entry.pack(side=tk.LEFT, padx=5)
        string_var.trace_add("write", self._validate_weights)

    def _validate_weights(self, *args):
        total = 0.0
        try:
            total += float(self.weight_name_sem_var.get() or 0)
            total += float(self.weight_name_fuzzy_var.get() or 0)
            total += float(self.weight_pub_var.get() or 0)
            total += float(self.weight_grade_var.get() or 0)
            total += float(self.weight_year_var.get() or 0)
            self.weights_sum_label.config(text=f"Sum: {total:.2f}")
            if abs(total - 1.0) > 0.01:
                self.weights_sum_label.config(foreground="orange")
            else:
                self.weights_sum_label.config(foreground="green")
        except Exception:
            self.weights_sum_label.config(text="Invalid number", foreground="red")

    def _load_settings(self):
        defaults = {
            "name_semantic": 0.40,
            "name_fuzzy": 0.15,
            "publisher": 0.25,
            "grade": 0.15,
            "year": 0.05,
            "profile": self.profile,
            "input_file": "",
            "catalog_file": "",
            "output_dir": "",
            "batch_records_file": str(self.default_rerank_records)
            if self.default_rerank_records.exists()
            else "",
            "batch_output_dir": str(self.default_rerank_output_dir),
            "batch_runs_dir": str(self.default_batch_runs_dir),
            "batch_benchmark_file": str(self.default_benchmark_file),
            "batch_run_name": "historical-top10-gpt54mini-medium-batch",
            "batch_model": "gpt-5.4-mini",
            "batch_reasoning_effort": "medium",
            "batch_shortlist_size": "10",
            "batch_poll_interval": "30",
        }
        try:
            if SETTINGS_FILE.exists():
                with SETTINGS_FILE.open("r") as f:
                    settings = json.load(f)
            else:
                settings = defaults
        except Exception:
            settings = defaults

        self.weight_name_sem_var.set(
            str(settings.get("name_semantic", defaults["name_semantic"]))
        )
        self.weight_name_fuzzy_var.set(
            str(settings.get("name_fuzzy", defaults["name_fuzzy"]))
        )
        self.weight_pub_var.set(str(settings.get("publisher", defaults["publisher"])))
        self.weight_grade_var.set(str(settings.get("grade", defaults["grade"])))
        self.weight_year_var.set(str(settings.get("year", defaults["year"])))
        self.profile_var.set(str(settings.get("profile", defaults["profile"])))
        self.input_file.set(str(settings.get("input_file", defaults["input_file"])))
        self.catalog_file.set(str(settings.get("catalog_file", defaults["catalog_file"])))
        self.output_dir.set(str(settings.get("output_dir", defaults["output_dir"])))
        self.batch_records_file.set(
            str(settings.get("batch_records_file", defaults["batch_records_file"]))
        )
        self.batch_output_dir.set(
            str(settings.get("batch_output_dir", defaults["batch_output_dir"]))
        )
        self.batch_runs_dir.set(
            str(settings.get("batch_runs_dir", defaults["batch_runs_dir"]))
        )
        self.batch_benchmark_file.set(
            str(settings.get("batch_benchmark_file", defaults["batch_benchmark_file"]))
        )
        self.batch_run_name.set(
            str(settings.get("batch_run_name", defaults["batch_run_name"]))
        )
        self.batch_model.set(str(settings.get("batch_model", defaults["batch_model"])))
        self.batch_reasoning_effort.set(
            str(
                settings.get(
                    "batch_reasoning_effort", defaults["batch_reasoning_effort"]
                )
            )
        )
        self.batch_shortlist_size.set(
            str(settings.get("batch_shortlist_size", defaults["batch_shortlist_size"]))
        )
        self.batch_poll_interval.set(
            str(settings.get("batch_poll_interval", defaults["batch_poll_interval"]))
        )
        self._validate_weights()
        self._check_ready()

    def _save_settings(self):
        try:
            settings = {
                "name_semantic": float(self.weight_name_sem_var.get() or 0),
                "name_fuzzy": float(self.weight_name_fuzzy_var.get() or 0),
                "publisher": float(self.weight_pub_var.get() or 0),
                "grade": float(self.weight_grade_var.get() or 0),
                "year": float(self.weight_year_var.get() or 0),
                "profile": self.profile_var.get(),
                "input_file": self.input_file.get().strip(),
                "catalog_file": self.catalog_file.get().strip(),
                "output_dir": self.output_dir.get().strip(),
                "batch_records_file": self.batch_records_file.get().strip(),
                "batch_output_dir": self.batch_output_dir.get().strip(),
                "batch_runs_dir": self.batch_runs_dir.get().strip(),
                "batch_benchmark_file": self.batch_benchmark_file.get().strip(),
                "batch_run_name": self.batch_run_name.get().strip(),
                "batch_model": self.batch_model.get().strip(),
                "batch_reasoning_effort": self.batch_reasoning_effort.get().strip(),
                "batch_shortlist_size": self.batch_shortlist_size.get().strip(),
                "batch_poll_interval": self.batch_poll_interval.get().strip(),
            }
            with SETTINGS_FILE.open("w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.log(f"Error saving settings: {e}")

    def _on_closing(self):
        self._save_settings()
        self._restore_process_output()
        if self.root:
            self.root.destroy()

    def _create_file_selector(self, parent, label_text, string_var, editable=True):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)
        entry = ttk.Entry(
            frame,
            textvariable=string_var,
            state="normal" if editable else "readonly",
        )
        entry.pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(
            frame, text="Browse...", command=lambda: self._select_file(string_var)
        ).pack(side=tk.LEFT, padx=5)

    def _create_directory_selector(
        self, parent, label_text, string_var, editable=True
    ):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)
        entry = ttk.Entry(
            frame,
            textvariable=string_var,
            state="normal" if editable else "readonly",
        )
        entry.pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(
            frame, text="Browse...", command=lambda: self._select_directory(string_var)
        ).pack(side=tk.LEFT, padx=5)

    def _create_labeled_entry(self, parent, label_text, string_var, readonly=True):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)
        ttk.Entry(
            frame,
            textvariable=string_var,
            state="readonly" if readonly else "normal",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _create_combo_row(self, parent, label_text, variable, values):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)
        combo = ttk.Combobox(
            frame,
            textvariable=variable,
            values=values,
            width=20,
            state="readonly",
        )
        combo.pack(side=tk.LEFT, padx=5)

    def _select_file(self, string_var):
        current_value = string_var.get().strip()
        initialdir = None
        initialfile = None
        if current_value:
            current_path = Path(current_value)
            if current_path.is_file():
                initialdir = str(current_path.parent)
                initialfile = current_path.name
            elif current_path.parent.exists():
                initialdir = str(current_path.parent)
                initialfile = current_path.name

        path = filedialog.askopenfilename(
            title="Select File",
            initialdir=initialdir,
            initialfile=initialfile,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            string_var.set(path)
            self._save_settings()
            self._check_ready()

    def _select_directory(self, string_var):
        current_value = string_var.get().strip()
        initialdir = current_value if current_value and Path(current_value).exists() else None
        path = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=initialdir,
        )
        if path:
            string_var.set(path)
            self._save_settings()
            self._check_ready()

    def _check_ready(self):
        if all([self.input_file.get(), self.catalog_file.get(), self.output_dir.get()]):
            self.process_button.config(state="normal")
            self.status_label.config(text="Ready to process.")
        elif hasattr(self, "process_button"):
            self.process_button.config(state="disabled")

    def _on_path_or_setting_changed(self, *args):
        self._check_ready()
        self._refresh_rerank_ready_state()
        if hasattr(self, "matcher_log_text") or hasattr(self, "rerank_log_text"):
            self._save_settings()

    def _refresh_rerank_ready_state(self, *args):
        rerank_ready = all(
            [
                self.batch_records_file.get().strip(),
                self.catalog_file.get().strip(),
                self.batch_output_dir.get().strip(),
            ]
        )
        state = "normal" if rerank_ready else "disabled"
        for button in (
            getattr(self, "build_prompt_button", None),
            getattr(self, "prepare_requests_button", None),
            getattr(self, "run_batch_button", None),
        ):
            if button:
                button.config(state=state)

    def _refresh_rerank_paths(self, *args):
        output_dir = Path(
            self.batch_output_dir.get().strip() or self.default_rerank_output_dir
        )
        records_path = Path(self.batch_records_file.get().strip() or "records.csv")
        base_name = records_path.stem
        if base_name.endswith("_records"):
            base_name = base_name[: -len("_records")]
        if not base_name:
            base_name = "llm_rerank"
        self.batch_prompt_jsonl.set(str(output_dir / f"{base_name}_llm_rerank_prompts.jsonl"))
        self.batch_prompt_summary_json.set(
            str(output_dir / f"{base_name}_llm_rerank_prompts_summary.json")
        )
        self.batch_request_jsonl.set(
            str(output_dir / f"{base_name}_llm_rerank_batch_requests.jsonl")
        )
        if not self.batch_runs_dir.get().strip():
            self.batch_runs_dir.set(str(output_dir / "openai_batch_runs"))

    def _refresh_run_name(self, *args):
        model_tag = re.sub(r"[^a-z0-9]+", "", self.batch_model.get().lower()) or "model"
        effort_tag = (
            re.sub(r"[^a-z0-9]+", "", self.batch_reasoning_effort.get().lower())
            or "default"
        )
        shortlist_size = self.batch_shortlist_size.get().strip() or "10"
        if (
            self.batch_run_name.get().strip().startswith("historical-")
            or not self.batch_run_name.get().strip()
        ):
            self.batch_run_name.set(
                f"historical-top{shortlist_size}-{model_tag}-{effort_tag}-batch"
            )

    def _update_progress(self, current, total):
        def update_gui():
            if total > 0:
                self.progress_var.set((current / total) * 100)
            self.status_label.config(text=f"Processing: {current}/{total}")

        if self.root:
            self.root.after(0, update_gui)

    # -------- human_match scorer (restored) --------
    def _score_human_match(self, row):
        """
        Compute component + final scores for the human-provided match id on this row.
        Returns a dict with human_match_* fields, or {} if no valid human id.
        """
        human_id = row.get(self.matcher.CATALOG_ID)
        if pd.isna(human_id):
            return {}

        # Find the same row in the matcher’s catalog (has embeddings)
        cand_rows = self.matcher.catalog_df[
            self.matcher.catalog_df[self.matcher.CATALOG_ID].astype(str)
            == str(human_id)
        ]
        if cand_rows.empty:
            return {}

        cand = cand_rows.iloc[0]

        search_text = self.matcher._normalize(
            row.get(self.matcher.INPUT_PRODUCT_NAME), True
        )
        if not search_text:
            return {}

        input_fast_vec = self.matcher.model_fast.encode([search_text])
        use_precise = "embedding_precise" in self.matcher.catalog_df.columns
        if use_precise:
            input_precise_vec = self.matcher.model_precise.encode([search_text])

        scored = self.matcher._score_candidate(
            row,
            cand,
            input_fast_vec,
            input_precise_vec=input_precise_vec if use_precise else None,
        )
        sem = scored["name_semantic"]
        fuzz = scored["name_fuzzy"]
        pub = scored["publisher"]
        grade = scored["grade"]
        year = scored["year"]

        # Back-compat single name score
        human_name_score = 0.7 * sem + 0.3 * fuzz

        return {
            "human_match_final_score": round(float(scored["final_score"]), 4),
            "human_match_name_score": round(float(human_name_score), 4),
            "human_match_publisher_score": round(float(pub), 4),
            "human_match_grade_score": round(float(grade), 4),
            # optional audits
            "human_match_name_semantic": round(float(sem), 4),
            "human_match_name_fuzzy": round(float(fuzz), 4),
            "human_match_year_score": round(float(year), 4),
        }

    def _assess_human_match(self, row, ai_result, llm_reviewer=None):
        scored = self._score_human_match(row)
        review = build_human_match_review(
            human_match_id="" if pd.isna(row.get(self.matcher.CATALOG_ID)) else str(row.get(self.matcher.CATALOG_ID)),
            human_scores=scored,
            ai_result=ai_result,
            row_context=dict(row),
        )
        if llm_reviewer and review and (
            review.get("human_match_challenge_flag")
            or ai_result.get("used_fallback")
        ):
            llm_context = {
                "row": dict(row),
                "human_scores": scored,
                "qa_review": review,
                "ai_result": ai_result,
            }
            try:
                llm_response = llm_reviewer.review_match(llm_context)
            except Exception as exc:
                llm_response = {"decision": "error", "reasoning": str(exc), "confidence": 0.0}
            review.update(
                {
                    "human_match_llm_review_requested": True,
                    "human_match_llm_decision": llm_response.get("decision", ""),
                    "human_match_llm_reasoning": llm_response.get("reasoning", ""),
                    "human_match_llm_confidence": llm_response.get("confidence", ""),
                }
            )
        elif review:
            review["human_match_llm_review_requested"] = False
        scored.update(review)
        return scored

    # -------- processing --------
    def _start_processing_thread(self):
        self._set_worker_buttons_enabled(False)
        self._save_settings()
        try:
            weights = {
                "name_semantic": float(self.weight_name_sem_var.get() or 0),
                "name_fuzzy": float(self.weight_name_fuzzy_var.get() or 0),
                "publisher": float(self.weight_pub_var.get() or 0),
                "grade": float(self.weight_grade_var.get() or 0),
                "year": float(self.weight_year_var.get() or 0),
            }
        except ValueError:
            messagebox.showerror(
                "Invalid Weight", "Please ensure all weights are valid numbers."
            )
            self.process_button.config(state="normal")
            return

        thread = threading.Thread(
            target=self._run_processing_logic,
            args=(
                self.input_file.get(),
                self.catalog_file.get(),
                self.output_dir.get(),
                weights,
                self.profile_var.get(),
            ),
            daemon=True,
        )
        thread.start()

    def _start_build_prompt_pack_thread(self):
        self._set_worker_buttons_enabled(False)
        thread = threading.Thread(target=self._build_prompt_pack_workflow, daemon=True)
        thread.start()

    def _start_prepare_batch_requests_thread(self):
        self._set_worker_buttons_enabled(False)
        thread = threading.Thread(
            target=self._prepare_batch_requests_workflow, daemon=True
        )
        thread.start()

    def _start_run_batch_pipeline_thread(self):
        self._set_worker_buttons_enabled(False)
        thread = threading.Thread(target=self._run_batch_pipeline_workflow, daemon=True)
        thread.start()

    def _run_headless_mode(self):
        input_file_path = "your_input_data.csv"
        catalog_file_path = "product_catalog.csv"
        output_dir_path = "results"
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        self._run_processing_logic(
            input_file_path, catalog_file_path, output_dir_path, None, self.profile
        )

    def _run_processing_logic(
        self, input_path, catalog_path, output_dir, weights, profile
    ):
        try:
            log = self.log
            progress = (
                self._update_progress if not self.headless else (lambda c, t: None)
            )

            if weights:
                self.matcher.weights = weights
                log(f"Using custom weights: {weights}")
            else:
                log(f"Using default weights: {self.matcher.weights}")
            self.matcher.profile = profile
            log(f"Profile selected: {profile}")

            log(f"Using Input File: {os.path.basename(input_path)}")
            log("Loading data files...")
            input_df = pd.read_csv(input_path, encoding="latin-1")
            catalog_df = pd.read_csv(catalog_path, encoding="latin-1")
            log(
                f"Loaded {len(input_df)} input records and {len(catalog_df)} catalog records."
            )

            self.matcher.load_models(profile=profile)
            self.matcher.prepare_catalog(catalog_df)

            log("Starting processing...")
            all_results = []
            # Lookup without embeddings (for catalog details)
            catalog_lookup = catalog_df.set_index(self.matcher.CATALOG_ID)

            recall_k = 40 if profile == "fast" else 60

            for idx, row in input_df.iterrows():
                progress(idx + 1, len(input_df))
                result_row = row.to_dict()

                # top-N matches
                match_result = self.matcher.match_record_with_repairs(
                    row, topn_stage1=recall_k, topn_final=3
                )
                matches = match_result["matches"]

                # human_match_* block (restored)
                hm = self._assess_human_match(row, match_result)
                result_row.update(hm)
                result_row["match_selected_strategy"] = match_result["selected_strategy"]
                result_row["match_used_fallback"] = match_result["used_fallback"]
                result_row["match_primary_top1_score"] = match_result["primary_top1_score"]
                result_row["match_selected_top1_score"] = match_result["selected_top1_score"]
                result_row["match_variant_scores"] = json.dumps(
                    match_result["variant_scores"], sort_keys=True
                )

                # fill top-3 results
                for i in range(3):
                    if i < len(matches):
                        m = matches[i]
                        match_id = m["catalog_id"]
                        result_row[f"match_{i+1}_id"] = match_id
                        result_row[f"match_{i+1}_final_score"] = round(
                            m["final_score"], 4
                        )
                        result_row[f"match_{i+1}_name_semantic"] = round(
                            m["name_semantic"], 4
                        )
                        result_row[f"match_{i+1}_name_fuzzy"] = round(
                            m["name_fuzzy"], 4
                        )
                        result_row[f"match_{i+1}_publisher_score"] = round(
                            m["publisher"], 4
                        )
                        result_row[f"match_{i+1}_grade_score"] = round(m["grade"], 4)
                        result_row[f"match_{i+1}_year_score"] = round(m["year"], 4)

                        # catalog details
                        # tolerate id type mismatch by trying both raw and str
                        try:
                            c = catalog_lookup.loc[match_id]
                        except KeyError:
                            c = catalog_lookup.loc[str(match_id)]

                        result_row[f"match_{i+1}_catalog_product_name"] = c.get(
                            self.matcher.CATALOG_PRODUCT_NAME
                        )
                        result_row[f"match_{i+1}_catalog_series"] = c.get(
                            self.matcher.CATALOG_SERIES
                        )
                        result_row[f"match_{i+1}_catalog_supplier_name"] = c.get(
                            self.matcher.CATALOG_PUBLISHER
                        )
                        result_row[f"match_{i+1}_catalog_copyright_year"] = c.get(
                            self.matcher.CATALOG_YEAR
                        )
                        result_row[f"match_{i+1}_catalog_intended_grades"] = c.get(
                            self.matcher.CATALOG_GRADES
                        )
                    else:
                        result_row[f"match_{i+1}_id"] = ""

                all_results.append(result_row)

            results_df = pd.DataFrame(all_results)
            log(f"Processing complete. Saving {len(results_df)} rows of results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"Matcher_Results_{timestamp}.csv"
            output_path = os.path.join(output_dir, base_filename)
            os.makedirs(output_dir, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            log(f"Results saved successfully to: {output_path}")
            if not self.headless:
                messagebox.showinfo("Success", "Processing complete!")

        except Exception as e:
            log(f"FATAL ERROR: {e}")
            if not self.headless:
                messagebox.showerror("Error", f"An error occurred:\n\n{e}")
        finally:
            if not self.headless:
                self._set_worker_buttons_enabled(True)

    def _parse_shortlist_size(self):
        try:
            value = int(self.batch_shortlist_size.get().strip() or "10")
        except ValueError as exc:
            raise ValueError("Shortlist size must be an integer.") from exc
        if value <= 0:
            raise ValueError("Shortlist size must be positive.")
        return value

    def _parse_poll_interval(self):
        try:
            value = int(self.batch_poll_interval.get().strip() or "30")
        except ValueError as exc:
            raise ValueError("Poll interval must be an integer number of seconds.") from exc
        if value <= 0:
            raise ValueError("Poll interval must be positive.")
        return value

    def _ensure_parent_dir(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _ensure_run_artifacts_do_not_exist(self, batch_output_dir, run_name):
        batch_output_dir = Path(batch_output_dir)
        expected_paths = [
            batch_output_dir / f"{run_name}-batch-output.jsonl",
            batch_output_dir / f"{run_name}-batch-errors.jsonl",
            batch_output_dir / f"{run_name}-scored.csv",
            batch_output_dir / f"{run_name}-scored-summary.json",
            batch_output_dir / f"{run_name}-pipeline-summary.json",
        ]
        existing_paths = [path for path in expected_paths if path.exists()]
        if existing_paths:
            joined = ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(
                "Run output artifacts already exist for this run name. "
                f"Choose a new run name or remove the existing files first: {joined}"
            )

    def _ensure_topn_records(self, records_path, topn_final):
        records_path = Path(records_path)
        if records_path.exists():
            self.log(f"Using matcher records CSV: {records_path}")
            return records_path

        benchmark_file = Path(self.batch_benchmark_file.get().strip())
        catalog_file = Path(self.catalog_file.get().strip())
        if not benchmark_file.exists():
            raise FileNotFoundError(
                f"Matcher records file not found and benchmark source is missing: {benchmark_file}"
            )
        if not catalog_file.exists():
            raise FileNotFoundError(
                f"Catalog file is required to regenerate matcher records: {catalog_file}"
            )

        summary_path = records_path.with_name(records_path.stem.replace("_records", "_summary") + ".json")
        self.log(
            f"Matcher records file not found. Regenerating top-{topn_final} baseline from benchmark..."
        )
        self._set_progress_value(
            5,
            f"Regenerating top-{topn_final} matcher records from benchmark...",
        )
        from .evaluation import evaluate_matcher_run

        evaluate_matcher_run(
            str(benchmark_file),
            str(catalog_file),
            profile="fast",
            topn_final=topn_final,
            output_json=str(summary_path),
            output_csv=str(records_path),
        )
        self.log(f"Regenerated matcher records: {records_path}")
        self.log(f"Regenerated matcher summary: {summary_path}")
        return records_path

    def _build_prompt_pack(self, *, only_errors=True):
        topn_final = self._parse_shortlist_size()
        records_path = self._ensure_topn_records(self.batch_records_file.get().strip(), topn_final)
        catalog_path = Path(self.catalog_file.get().strip())
        prompt_jsonl_path = Path(self.batch_prompt_jsonl.get().strip())
        prompt_summary_path = Path(self.batch_prompt_summary_json.get().strip())
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

        self._ensure_parent_dir(prompt_jsonl_path)
        self._ensure_parent_dir(prompt_summary_path)
        self._set_progress_value(10, "Building rerank prompt pack...")
        self.log(
            f"Building rerank prompt pack from {records_path.name} with top-{topn_final} candidates."
        )

        records_df = pd.read_csv(records_path)
        catalog_df = pd.read_csv(catalog_path, encoding="latin-1")
        catalog_lookup = build_catalog_lookup(catalog_df)

        if only_errors:
            records_df = records_df.loc[
                (~records_df["top1_correct"].fillna(False))
                | (
                    records_df["predicted_match_id"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    == ""
                )
            ].copy()
        prompts = []
        total_rows = len(records_df)
        for row_number, (_, row) in enumerate(records_df.iterrows(), start=1):
            prompt_record = build_prompt_record(
                row.to_dict(),
                catalog_df,
                topn_final=topn_final,
                catalog_lookup=catalog_lookup,
            )
            if not prompt_record["candidate_ids"]:
                continue
            prompts.append(prompt_record)
            if row_number == 1 or row_number % 50 == 0 or row_number == total_rows:
                self._set_progress_value(
                    10 + (row_number / max(total_rows, 1)) * 25,
                    f"Building rerank prompt pack: {row_number}/{total_rows}",
                )

        with prompt_jsonl_path.open("w", encoding="utf-8") as handle:
            for prompt in prompts:
                handle.write(json.dumps(prompt, ensure_ascii=True) + "\n")

        candidate_distribution = (
            pd.Series([len(prompt["candidate_ids"]) for prompt in prompts])
            .value_counts()
            .sort_index()
            .to_dict()
            if prompts
            else {}
        )
        summary = {
            "row_count": int(len(prompts)),
            "source_records_csv": str(records_path),
            "catalog_file": str(catalog_path),
            "topn_final": topn_final,
            "only_errors": bool(only_errors),
            "candidate_count_distribution": candidate_distribution,
        }
        prompt_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.log(f"Prompt pack saved: {prompt_jsonl_path}")
        self.log(f"Prompt summary saved: {prompt_summary_path}")
        self.log(
            f"Prompt pack row count: {summary['row_count']} | candidate distribution: {candidate_distribution}"
        )
        return summary

    def _prepare_batch_requests(self):
        prompt_jsonl_path = Path(self.batch_prompt_jsonl.get().strip())
        request_jsonl_path = Path(self.batch_request_jsonl.get().strip())
        if not prompt_jsonl_path.exists():
            raise FileNotFoundError(
                f"Prompt pack JSONL not found. Build it first: {prompt_jsonl_path}"
            )
        self._ensure_parent_dir(request_jsonl_path)
        self._set_progress_value(40, "Preparing Batch request JSONL...")

        row_count = 0
        with prompt_jsonl_path.open("r", encoding="utf-8") as src, request_jsonl_path.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                prompt_record = json.loads(line)
                request_body = build_responses_api_body(
                    prompt_record,
                    model=self.batch_model.get().strip(),
                    reasoning_effort=self.batch_reasoning_effort.get().strip() or None,
                )
                batch_line = {
                    "custom_id": prompt_record["selection_identifier"],
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": request_body,
                }
                dst.write(json.dumps(batch_line, ensure_ascii=True) + "\n")
                row_count += 1

        file_size_kb = round(request_jsonl_path.stat().st_size / 1024.0, 1)
        self.log(
            f"Batch request JSONL saved: {request_jsonl_path} ({row_count} requests, {file_size_kb} KB)"
        )
        return {
            "row_count": row_count,
            "prompt_jsonl": str(prompt_jsonl_path),
            "output_jsonl": str(request_jsonl_path),
            "model": self.batch_model.get().strip(),
            "reasoning_effort": self.batch_reasoning_effort.get().strip() or None,
        }

    def _load_prompt_lookup(self, path):
        lookup = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                lookup[row["selection_identifier"]] = row
        return lookup

    def _extract_output_text(self, response_body):
        output = response_body.get("output", [])
        texts = []
        for item in output:
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()

    def _score_batch_output(self, *, prompt_jsonl, batch_output_jsonl, run_name, output_dir):
        prompt_lookup = self._load_prompt_lookup(prompt_jsonl)
        rows = []
        repaired_selected_id_count = 0
        invalid_selected_id_count = 0
        with Path(batch_output_jsonl).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                batch_row = json.loads(line)
                selection_identifier = batch_row.get("custom_id", "")
                prompt_row = prompt_lookup.get(selection_identifier)
                if not prompt_row:
                    continue

                response_body = batch_row.get("response", {}).get("body", {})
                response_text = self._extract_output_text(response_body)
                parsed = extract_json_object(response_text)
                validated = validate_and_repair_rerank_response(
                    parsed,
                    candidate_ids=prompt_row["candidate_ids"],
                )
                selected_candidate_id = validated["selected_candidate_id"]
                expected_match_id = prompt_row.get("expected_match_id", "")
                repaired_selected_id_count += int(
                    validated["repaired_selected_candidate_id"]
                )
                invalid_selected_id_count += int(validated["invalid_selected_candidate_id"])
                candidate_ids = prompt_row.get("candidate_ids", [])
                shortlist_size = len(candidate_ids)
                baseline_topn_key = (
                    f"top{shortlist_size}_correct" if shortlist_size else "top3_correct"
                )

                rows.append(
                    {
                        "selection_identifier": selection_identifier,
                        "expected_match_id": expected_match_id,
                        "baseline_predicted_match_id": prompt_row.get(
                            "predicted_match_id", ""
                        ),
                        "llm_selected_candidate_id": selected_candidate_id,
                        "llm_selected_candidate_id_original": validated[
                            "selected_candidate_id_original"
                        ],
                        "llm_selected_candidate_id_repaired": validated[
                            "repaired_selected_candidate_id"
                        ],
                        "llm_selected_candidate_id_invalid": validated[
                            "invalid_selected_candidate_id"
                        ],
                        "llm_decision": validated["decision"],
                        "llm_confidence": validated["confidence"],
                        "llm_primary_reason": validated["primary_reason"],
                        "llm_secondary_tags": "|".join(validated["secondary_tags"]),
                        "llm_reasoning": validated["reasoning"],
                        "baseline_top1_correct": bool(prompt_row.get("top1_correct")),
                        "baseline_shortlist_contains_gold": bool(
                            prompt_row.get(baseline_topn_key, prompt_row.get("top3_correct"))
                        ),
                        "llm_top1_correct": bool(
                            selected_candidate_id
                            and expected_match_id
                            and selected_candidate_id == expected_match_id
                        ),
                        "llm_abstained": validated["decision"] == "abstain",
                    }
                )

        df = pd.DataFrame(rows)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        score_csv = output_dir / f"{run_name}-scored.csv"
        score_json = output_dir / f"{run_name}-scored-summary.json"
        df.to_csv(score_csv, index=False)

        summary = {
            "row_count": int(len(df)),
            "llm_selection_rate": round(float((~df["llm_abstained"]).mean()), 4)
            if not df.empty
            else 0.0,
            "llm_top1_accuracy_on_all_rows": round(
                float(df["llm_top1_correct"].mean()), 4
            )
            if not df.empty
            else 0.0,
            "llm_top1_accuracy_on_selected_rows": (
                round(float(df.loc[~df["llm_abstained"], "llm_top1_correct"].mean()), 4)
                if not df.empty and (~df["llm_abstained"]).any()
                else 0.0
            ),
            "primary_reason_counts": (
                df["llm_primary_reason"]
                .fillna("missing")
                .value_counts()
                .sort_index()
                .to_dict()
                if not df.empty
                else {}
            ),
            "repaired_selected_id_count": int(repaired_selected_id_count),
            "invalid_selected_id_count": int(invalid_selected_id_count),
        }
        score_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.log(f"Scored CSV saved: {score_csv}")
        self.log(f"Scored summary saved: {score_json}")
        if repaired_selected_id_count or invalid_selected_id_count:
            self.log(
                "Scoring notes: "
                f"repaired_selected_id_count={repaired_selected_id_count} "
                f"invalid_selected_id_count={invalid_selected_id_count}"
            )
        return summary, score_csv, score_json

    def _format_batch_status(self, batch):
        status = batch.get("status", "unknown")
        request_counts = batch.get("request_counts") or {}
        if request_counts:
            return (
                f"{status} | total={request_counts.get('total', 0)} "
                f"completed={request_counts.get('completed', 0)} "
                f"failed={request_counts.get('failed', 0)}"
            )
        return status

    def _update_batch_summary(self, summary):
        primary_reasons = summary.get("primary_reason_counts", {})
        top_reason = ""
        if primary_reasons:
            reason, count = max(primary_reasons.items(), key=lambda item: item[1])
            top_reason = f" | top reason: {reason} ({count})"
        message = (
            f"Reviewed rows: {summary.get('row_count', 0)} | "
            f"selection rate: {summary.get('llm_selection_rate', 0.0):.4f} | "
            f"top-1 all rows: {summary.get('llm_top1_accuracy_on_all_rows', 0.0):.4f} | "
            f"top-1 selected rows: {summary.get('llm_top1_accuracy_on_selected_rows', 0.0):.4f}"
            f"{top_reason}"
        )
        self._set_summary_text(message)

    def _build_prompt_pack_workflow(self):
        try:
            summary = self._build_prompt_pack(only_errors=True)
            self._set_progress_value(35, "Prompt pack ready.")
            self._set_summary_text(
                f"Prompt pack ready: {summary['row_count']} review rows using top-{summary['topn_final']} shortlist."
            )
        except Exception as exc:
            self.log(f"Prompt-pack build failed: {exc}")
            self._set_status_text(f"Prompt-pack build failed: {exc}")
        finally:
            self._set_worker_buttons_enabled(True)

    def _prepare_batch_requests_workflow(self):
        try:
            if not Path(self.batch_prompt_jsonl.get().strip()).exists():
                self._build_prompt_pack(only_errors=True)
            info = self._prepare_batch_requests()
            self._set_progress_value(50, "Batch request JSONL ready.")
            self._set_summary_text(
                f"Batch request file ready: {info['row_count']} requests for {info['model']}."
            )
        except Exception as exc:
            self.log(f"Batch request preparation failed: {exc}")
            self._set_status_text(f"Batch request preparation failed: {exc}")
        finally:
            self._set_worker_buttons_enabled(True)

    def _run_batch_pipeline_workflow(self):
        try:
            shortlist_size = self._parse_shortlist_size()
            poll_interval = self._parse_poll_interval()
            run_name = self.batch_run_name.get().strip()
            if not run_name:
                raise ValueError("Run name is required.")

            prompt_summary = self._build_prompt_pack(only_errors=True)
            request_summary = self._prepare_batch_requests()

            request_jsonl = Path(request_summary["output_jsonl"])
            batch_output_dir = Path(self.batch_runs_dir.get().strip())
            batch_output_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_run_artifacts_do_not_exist(batch_output_dir, run_name)

            self._set_progress_value(55, "Uploading Batch request file...")
            self.log(f"Uploading Batch request file: {request_jsonl}")
            uploaded = upload_batch_file(request_jsonl)
            self.log(f"Upload complete. File id: {uploaded['id']}")

            self._set_progress_value(62, "Creating OpenAI Batch job...")
            metadata = {
                "run_name": run_name,
                "shortlist_size": str(shortlist_size),
                "prompt_rows": str(prompt_summary["row_count"]),
                "model": self.batch_model.get().strip(),
                "reasoning_effort": self.batch_reasoning_effort.get().strip() or "",
            }
            batch = create_batch(
                input_file_id=uploaded["id"],
                endpoint="/v1/responses",
                completion_window="24h",
                metadata=metadata,
            )
            batch_id = batch["id"]
            self.log(f"Batch created. Batch id: {batch_id}")

            terminal_statuses = {"completed", "failed", "expired", "cancelled", "cancelling"}
            poll_count = 0
            started_at = time.time()
            while batch.get("status") not in terminal_statuses:
                poll_count += 1
                status_text = self._format_batch_status(batch)
                elapsed = int(time.time() - started_at)
                self.log(
                    f"Polling Batch status every {poll_interval}s | poll #{poll_count} | elapsed={elapsed}s | {status_text}"
                )
                self._set_progress_value(
                    68 + ((poll_count - 1) % 5) * 4,
                    f"Batch running: {status_text}",
                )
                time.sleep(poll_interval)
                batch = retrieve_batch(batch_id)

            final_status = self._format_batch_status(batch)
            self.log(f"Batch finished with status: {final_status}")
            self._set_progress_value(88, f"Batch finished: {final_status}")

            pipeline_summary = {
                "uploaded_file_id": uploaded["id"],
                "batch_id": batch_id,
                "batch_status": batch.get("status"),
                "input_jsonl": str(request_jsonl),
                "prompt_jsonl": self.batch_prompt_jsonl.get().strip(),
                "run_name": run_name,
                "model": self.batch_model.get().strip(),
                "reasoning_effort": self.batch_reasoning_effort.get().strip() or "",
                "shortlist_size": shortlist_size,
            }

            output_file_id = batch.get("output_file_id")
            error_file_id = batch.get("error_file_id")
            output_download_path = None

            if output_file_id:
                output_download_path = batch_output_dir / f"{run_name}-batch-output.jsonl"
                output_bytes = download_file_content(output_file_id)
                output_download_path.write_bytes(output_bytes)
                pipeline_summary["output_download"] = {
                    "file_id": output_file_id,
                    "output_path": str(output_download_path),
                    "byte_count": len(output_bytes),
                }
                self.log(
                    f"Downloaded batch output: {output_download_path} ({len(output_bytes)} bytes)"
                )

            if error_file_id:
                error_download_path = batch_output_dir / f"{run_name}-batch-errors.jsonl"
                error_bytes = download_file_content(error_file_id)
                error_download_path.write_bytes(error_bytes)
                pipeline_summary["error_download"] = {
                    "file_id": error_file_id,
                    "output_path": str(error_download_path),
                    "byte_count": len(error_bytes),
                }
                self.log(
                    f"Downloaded batch errors: {error_download_path} ({len(error_bytes)} bytes)"
                )

            if batch.get("status") == "completed" and output_download_path:
                self._set_progress_value(94, "Scoring Batch output...")
                summary, score_csv, score_json = self._score_batch_output(
                    prompt_jsonl=self.batch_prompt_jsonl.get().strip(),
                    batch_output_jsonl=output_download_path,
                    run_name=run_name,
                    output_dir=batch_output_dir,
                )
                pipeline_summary["score_summary"] = summary
                pipeline_summary["score_csv"] = str(score_csv)
                pipeline_summary["score_json"] = str(score_json)
                self._update_batch_summary(summary)
            else:
                self._set_summary_text(f"Batch finished with status: {batch.get('status')}")

            pipeline_summary_path = batch_output_dir / f"{run_name}-pipeline-summary.json"
            pipeline_summary_path.write_text(
                json.dumps(pipeline_summary, indent=2), encoding="utf-8"
            )
            self.log(f"Pipeline summary saved: {pipeline_summary_path}")
            self._set_progress_value(100, f"Batch pipeline complete: {batch.get('status')}")

        except Exception as exc:
            self.log(f"Batch pipeline failed: {exc}")
            self._set_status_text(f"Batch pipeline failed: {exc}")
            self._set_summary_text(f"Batch pipeline failed: {exc}")
        finally:
            self._set_worker_buttons_enabled(True)


# =========================
#           Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Matcher V3.1.2 (GUI + Profiles + Robust Year + human_match)"
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument(
        "--profile",
        choices=["fast", "accurate"],
        default="fast",
        help="fast=MiniLM-only, accurate=MPNet re-rank",
    )
    args = parser.parse_args()
    app = MatcherApp(headless=args.headless, profile=args.profile)
    app.run()


if __name__ == "__main__":
    main()
