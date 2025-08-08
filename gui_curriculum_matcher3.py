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
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re
import threading
import os
from datetime import datetime
import json
import argparse
import unicodedata


# =========================
#   Core Matcher (V3.1.2)
# =========================
class EnhancedCurriculumMatcherV312:
    """
    Two-stage retrieval (BM25 + semantic) with profile toggle and robust scoring.
    """

    def __init__(self, log_callback=print, weights=None):
        self.log = log_callback
        self.model_fast = None
        self.model_precise = None
        self.bm25 = None
        self.catalog_df = None
        self.profile = "fast"  # overridden by GUI/CLI

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
            "pearson": "savvas",
            "holt mcdougal": "hmh",
            "houghton mifflin harcourt": "hmh",
            "mheducation": "mcgraw hill",
            "mcgraw-hill": "mcgraw hill",
            "mcgraw hill education": "mcgraw hill",
        }
        self.subject_aliases = {
            "ela": "english language arts",
            "lang arts": "english language arts",
            "maths": "mathematics",
        }

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
        self._cosine_similarity = None
        self._BM25Okapi = None

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

    def _publisher_canonical(self, text):
        s = self._normalize(text, for_semantic=True)
        s = re.sub(r"\b(inc|llc|ltd|co|corp|company)\b\.?", "", s).strip()
        for k, v in self.publisher_aliases.items():
            if s == k:
                return v
        return s

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
        self.log(f"Models loaded (profile={self.profile}).")

    def prepare_catalog(self, catalog_df):
        """
        Pre-clean, build search text, encode recall embeddings, BM25,
        and precise embeddings (if MPNet).
        """
        self.catalog_df = catalog_df.copy()

        def build_search_text(row):
            name = self._normalize(row.get(self.CATALOG_PRODUCT_NAME), True)
            series = self._normalize(row.get(self.CATALOG_SERIES), True)
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
                self._normalize(input_text), self._normalize(cand_text)
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

    # ---------- Matching ----------
    def match_record(self, record, topn_stage1=60, topn_final=3):
        if self.catalog_df is None or self.bm25 is None:
            raise RuntimeError("Catalog not prepared. Call prepare_catalog() first.")

        search_text = self._normalize(record.get(self.INPUT_PRODUCT_NAME), True)
        if not search_text:
            return []

        # Stage 1 recall: BM25 + MiniLM semantic
        bm25_scores = self.bm25.get_scores(search_text.split())
        input_fast_vec = self.model_fast.encode([search_text])  # cached per row
        cat_fast = np.vstack(self.catalog_df["embedding_fast"].values)
        sem_scores_fast = self._cosine_similarity(input_fast_vec, cat_fast)[0]
        combined = 0.5 * bm25_scores + 0.5 * sem_scores_fast

        recall_k = 40 if self.profile == "fast" else 60
        top_idx = np.argsort(combined)[::-1][:recall_k]

        # Stage 2 rerank: precise model if available
        use_precise = "embedding_precise" in self.catalog_df.columns
        if use_precise:
            input_precise_vec = self.model_precise.encode([search_text])

        results = []
        input_year_hint = record.get(self.INPUT_PRODUCT_NAME) or record.get(
            self.INPUT_PUBLISHER
        )

        for idx in top_idx:
            cand = self.catalog_df.iloc[idx]
            if use_precise:
                cand_sem_vec = cand["embedding_precise"]
                input_sem_vec = input_precise_vec
            else:
                cand_sem_vec = cand["embedding_fast"]
                input_sem_vec = input_fast_vec

            sem, fuzz = self._name_scores(
                input_sem_vec,
                cand_sem_vec,
                record.get(self.INPUT_PRODUCT_NAME),
                cand["search_text"],
            )

            # hard negative cutoff
            if sem < 0.70 and fuzz < 0.55:
                continue

            pub = self._publisher_score(
                record.get(self.INPUT_PUBLISHER),
                cand.get(self.CATALOG_PUBLISHER),
                cand.get(self.CATALOG_PUBLISHER_PRIOR),
            )
            grade = self._grade_score(
                record.get(self.INPUT_GRADE), cand.get(self.CATALOG_GRADES)
            )
            year = self._year_score(input_year_hint, cand.get(self.CATALOG_YEAR))

            final = (
                self.weights["name_semantic"] * sem
                + self.weights["name_fuzzy"] * fuzz
                + self.weights["publisher"] * pub
                + self.weights["grade"] * grade
                + self.weights["year"] * year
            )

            results.append(
                {
                    "catalog_id": cand.get(self.CATALOG_ID),
                    "final_score": float(final),
                    "name_semantic": float(sem),
                    "name_fuzzy": float(fuzz),
                    "publisher": float(pub),
                    "grade": float(grade),
                    "year": float(year),
                }
            )

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:topn_final]


# =========================
#          GUI
# =========================
SETTINGS_FILE = "matcher_settings.json"


class MatcherApp:
    def __init__(self, headless=False, profile="fast"):
        self.headless = headless
        self.root = None
        self.matcher = EnhancedCurriculumMatcherV312(
            log_callback=self.log if not headless else print
        )
        self.profile = profile

        if not self.headless:
            self.root = tk.Tk()
            self._setup_gui()

    # -------- logging helpers --------
    def log(self, message):
        if self.root:
            self.root.after(0, lambda: self._log_message(message))
        else:
            print(message)

    def _log_message(self, message):
        self.log_text.insert(
            tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
        )
        self.log_text.see(tk.END)

    def run(self):
        if self.headless:
            self._run_headless_mode()
        else:
            self.root.mainloop()

    # -------- UI setup --------
    def _setup_gui(self):
        self.root.title("Enhanced Curriculum Matcher (V3.1.2)")
        self.root.geometry("900x840")

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

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Frame 1: Files ---
        file_frame = ttk.LabelFrame(main_frame, text="1. Select Files & Directory")
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self._create_file_selector(file_frame, "Input Data File:", self.input_file)
        self._create_file_selector(
            file_frame, "Product Catalog File:", self.catalog_file
        )
        self._create_directory_selector(
            file_frame, "Output Directory:", self.output_dir
        )

        # --- Frame 2: Weights & Profile ---
        weights_frame = ttk.LabelFrame(main_frame, text="2. Configure Weights")
        weights_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
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
        self._load_settings()

        # --- Frame 3: Controls & Status ---
        self.process_button = ttk.Button(
            main_frame,
            text="3. Start Processing",
            command=self._start_processing_thread,
            state="disabled",
        )
        self.process_button.grid(row=2, column=0, columnspan=2, pady=20)

        status_frame = ttk.LabelFrame(main_frame, text="Status")
        status_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        self.status_label = ttk.Label(
            status_frame, text="Please select all files and an output directory."
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=5)

        # --- Frame 4: Log ---
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        main_frame.rowconfigure(4, weight=1)
        main_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=15, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

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
        }
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r") as f:
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
        self._validate_weights()

    def _save_settings(self):
        try:
            settings = {
                "name_semantic": float(self.weight_name_sem_var.get() or 0),
                "name_fuzzy": float(self.weight_name_fuzzy_var.get() or 0),
                "publisher": float(self.weight_pub_var.get() or 0),
                "grade": float(self.weight_grade_var.get() or 0),
                "year": float(self.weight_year_var.get() or 0),
                "profile": self.profile_var.get(),
            }
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.log(f"Error saving settings: {e}")

    def _on_closing(self):
        self._save_settings()
        if self.root:
            self.root.destroy()

    def _create_file_selector(self, parent, label_text, string_var):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=string_var, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(
            frame, text="Browse...", command=lambda: self._select_file(string_var)
        ).pack(side=tk.LEFT, padx=5)

    def _create_directory_selector(self, parent, label_text, string_var):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=string_var, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(
            frame, text="Browse...", command=lambda: self._select_directory(string_var)
        ).pack(side=tk.LEFT, padx=5)

    def _select_file(self, string_var):
        path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            string_var.set(path)
            self._check_ready()

    def _select_directory(self, string_var):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            string_var.set(path)
            self._check_ready()

    def _check_ready(self):
        if all([self.input_file.get(), self.catalog_file.get(), self.output_dir.get()]):
            self.process_button.config(state="normal")
            self.status_label.config(text="Ready to process.")

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

        cand_search_text = cand["search_text"]
        cand_vec = (
            cand["embedding_precise"]
            if use_precise and "embedding_precise" in cand
            else cand["embedding_fast"]
        )
        input_vec = input_precise_vec if use_precise else input_fast_vec

        sem = float(self.matcher._cosine_similarity(input_vec, [cand_vec])[0][0])
        fuzz = (
            self.matcher._rapidfuzz.fuzz.token_set_ratio(
                self.matcher._normalize(row.get(self.matcher.INPUT_PRODUCT_NAME)),
                self.matcher._normalize(cand_search_text),
            )
            / 100.0
        )

        pub = self.matcher._publisher_score(
            row.get(self.matcher.INPUT_PUBLISHER),
            cand.get(self.matcher.CATALOG_PUBLISHER),
            cand.get(self.matcher.CATALOG_PUBLISHER_PRIOR),
        )
        grade = self.matcher._grade_score(
            row.get(self.matcher.INPUT_GRADE), cand.get(self.matcher.CATALOG_GRADES)
        )
        year = self.matcher._year_score(
            (
                row.get(self.matcher.INPUT_PRODUCT_NAME)
                or row.get(self.matcher.INPUT_PUBLISHER)
            ),
            cand.get(self.matcher.CATALOG_YEAR),
        )

        # Back-compat single name score
        human_name_score = 0.7 * sem + 0.3 * fuzz

        w = self.matcher.weights
        final = (
            w["name_semantic"] * sem
            + w["name_fuzzy"] * fuzz
            + w["publisher"] * pub
            + w["grade"] * grade
            + w["year"] * year
        )

        return {
            "human_match_final_score": round(float(final), 4),
            "human_match_name_score": round(float(human_name_score), 4),
            "human_match_publisher_score": round(float(pub), 4),
            "human_match_grade_score": round(float(grade), 4),
            # optional audits
            "human_match_name_semantic": round(float(sem), 4),
            "human_match_name_fuzzy": round(float(fuzz), 4),
            "human_match_year_score": round(float(year), 4),
        }

    # -------- processing --------
    def _start_processing_thread(self):
        self.process_button.config(state="disabled")
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
                matches = self.matcher.match_record(
                    row, topn_stage1=recall_k, topn_final=3
                )

                # human_match_* block (restored)
                hm = self._score_human_match(row)
                result_row.update(hm)

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
                self.root.after(0, lambda: self.process_button.config(state="normal"))


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
