from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import rapidfuzz
import re
import unicodedata
import pandas as pd
import numpy as np
from collections import defaultdict
import string


class EnhancedCurriculumMatcherV2:
    """Refactored matcher with pre-cleaning, two-stage retrieval, and improved scoring."""

    def __init__(self, log_callback=print, weights=None):
        self.log_callback = log_callback
        self.model_fast = None  # Stage 1 model
        self.model_precise = None  # Stage 2 model
        self.bm25 = None
        self.catalog_df = None

        # Column names (compatible with your existing data)
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

        # Alias maps for cleaning
        self.publisher_aliases = {
            "pearson": "savvas",
            "holt mcdougal": "hmh",
            "houghton mifflin harcourt": "hmh",
            "mheducation": "mcgraw hill",
        }
        self.subject_aliases = {
            "ela": "english language arts",
            "maths": "mathematics",
            "lang arts": "english language arts",
        }

        self.weights = (
            weights
            if weights
            else {
                "name_semantic": 0.4,
                "name_fuzzy": 0.15,
                "publisher": 0.25,
                "grade": 0.15,
                "year": 0.05,
            }
        )

    def _log(self, msg):
        self.log_callback(msg)

    # --------------------------
    # Pre-cleaning and Normalization
    # --------------------------
    def _normalize(self, text, for_semantic=False):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Unicode normalize
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        # Remove punctuation except spaces
        if for_semantic:
            text = re.sub(r"[^\w\s]", " ", text)
            text = re.sub(r"\s+", " ", text)
        else:
            text = re.sub(r"[^\w\s]", "", text)
        # Expand grade forms
        text = re.sub(r"\bgr\b", "grade", text)
        text = text.replace("pre-k", "prek")
        # Map known subjects
        for k, v in self.subject_aliases.items():
            if k in text:
                text = text.replace(k, v)
        # Map publishers if exact match
        if not for_semantic:
            for k, v in self.publisher_aliases.items():
                if text.strip() == k:
                    text = v
        return text.strip()

    def _extract_year(self, text):
        m = re.search(r"(19|20)\d{2}", str(text))
        return int(m.group()) if m else None

    def _parse_grade_range(self, grade_str):
        """Returns low, high grade as ints (0=K/PreK)"""
        if pd.isna(grade_str):
            return None, None
        mapping = {"p": 0, "k": 0, "pk": 0, "prek": 0}
        g = str(grade_str).lower().strip()
        if g in mapping:
            return 0, 0
        nums = re.findall(r"\d+", g)
        if not nums:
            return None, None
        nums = [int(n) for n in nums]
        return min(nums), max(nums)

    # --------------------------
    # Stage 1: Catalog Preparation
    # --------------------------
    def load_models(self):
        if self.model_fast is None:
            self._log("Loading fast model for recall...")
            self.model_fast = SentenceTransformer("all-MiniLM-L6-v2")
        if self.model_precise is None:
            self._log("Loading precise model for re-ranking...")
            self.model_precise = SentenceTransformer("all-mpnet-base-v2")

    def prepare_catalog(self, catalog_df):
        """Precompute embeddings & BM25 index."""
        self.catalog_df = catalog_df.copy()
        self._log("Pre-cleaning catalog entries...")
        self.catalog_df["search_text"] = self.catalog_df.apply(
            lambda row: " ".join(
                [
                    self._normalize(row[self.CATALOG_PRODUCT_NAME], True),
                    self._normalize(row[self.CATALOG_SERIES], True),
                    str(row[self.CATALOG_YEAR] or ""),
                ]
            ),
            axis=1,
        )

        self._log("Encoding catalog with fast model...")
        self.catalog_df["embedding_fast"] = list(
            self.model_fast.encode(
                self.catalog_df["search_text"].tolist(),
                batch_size=32,
                show_progress_bar=True,
            )
        )

        self._log("Building BM25 index...")
        tokenized = [s.split() for s in self.catalog_df["search_text"]]
        self.bm25 = BM25Okapi(tokenized)

        self._log("Catalog ready.")

    # --------------------------
    # Stage 2: Candidate Retrieval & Scoring
    # --------------------------
    def _get_name_scores(self, input_text, candidate_row):
        norm_input_sem = self._normalize(input_text, True)
        norm_cand_sem = candidate_row["search_text"]

        sem_score = cosine_similarity(
            self.model_precise.encode([norm_input_sem]),
            self.model_precise.encode([norm_cand_sem]),
        )[0][0]

        fuzz_score = (
            rapidfuzz.fuzz.token_set_ratio(
                self._normalize(input_text), self._normalize(norm_cand_sem)
            )
            / 100.0
        )

        return sem_score, fuzz_score

    def _get_publisher_score(self, input_pub, cand_pub):
        return (
            rapidfuzz.fuzz.WRatio(self._normalize(input_pub), self._normalize(cand_pub))
            / 100.0
        )

    def _get_grade_score(self, input_grade, cand_grade_str):
        in_low, in_high = self._parse_grade_range(input_grade)
        c_low, c_high = self._parse_grade_range(cand_grade_str)
        if in_low is None or c_low is None:
            return 0
        # Steeper decay
        if in_low > c_high:
            dist = in_low - c_high
        elif c_low > in_high:
            dist = c_low - in_high
        else:
            dist = 0
        return {0: 1.0, 1: 0.8, 2: 0.4}.get(dist, 0.0)

    def _get_year_score(self, input_year, cand_year):
        if not input_year or not cand_year:
            return 0
        return 1.0 if input_year == cand_year else 0.0

    def match_record(self, record, topn_stage1=50, topn_final=5):
        # Stage 1: BM25 + fast semantic to get candidates
        search_text = self._normalize(record.get(self.INPUT_PRODUCT_NAME), True)
        bm25_scores = self.bm25.get_scores(search_text.split())
        fast_emb = self.model_fast.encode([search_text])
        sem_scores_fast = cosine_similarity(
            fast_emb, np.vstack(self.catalog_df["embedding_fast"])
        )[0]
        combined_stage1 = 0.5 * bm25_scores + 0.5 * sem_scores_fast
        top_candidates_idx = np.argsort(combined_stage1)[::-1][:topn_stage1]

        # Stage 2: precise scoring
        results = []
        input_year = self._extract_year(
            record.get(self.INPUT_PRODUCT_NAME)
        ) or self._extract_year(record.get(self.INPUT_PUBLISHER))
        for idx in top_candidates_idx:
            cand = self.catalog_df.iloc[idx]
            sem_score, fuzz_score = self._get_name_scores(
                record.get(self.INPUT_PRODUCT_NAME), cand
            )
            pub_score = self._get_publisher_score(
                record.get(self.INPUT_PUBLISHER), cand[self.CATALOG_PUBLISHER]
            )
            grade_score = self._get_grade_score(
                record.get(self.INPUT_GRADE), cand[self.CATALOG_GRADES]
            )
            year_score = self._get_year_score(input_year, cand[self.CATALOG_YEAR])

            final_score = (
                self.weights["name_semantic"] * sem_score
                + self.weights["name_fuzzy"] * fuzz_score
                + self.weights["publisher"] * pub_score
                + self.weights["grade"] * grade_score
                + self.weights["year"] * year_score
            )

            # Hard negative filter
            if sem_score < 0.7 and fuzz_score < 0.55:
                continue

            results.append(
                {
                    "catalog_id": cand[self.CATALOG_ID],
                    "final_score": final_score,
                    "name_semantic": sem_score,
                    "name_fuzzy": fuzz_score,
                    "publisher": pub_score,
                    "grade": grade_score,
                    "year": year_score,
                }
            )

        results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        return results[:topn_final]
