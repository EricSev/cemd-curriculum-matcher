import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re
import threading
import os
import time
from datetime import datetime
import json
import argparse

# Defer heavy imports until needed
sentence_transformers = None
cosine_similarity = None
rapidfuzz = None


def load_heavy_imports(log_callback=print):
    """
    Loads heavy libraries only when they are first needed.
    This allows the application to start up instantly.
    Args:
        log_callback (function, optional): A function to log messages. Defaults to print.
    """
    global sentence_transformers, cosine_similarity, rapidfuzz
    if sentence_transformers is None:
        message = "Loading required libraries (this may take a moment)..."
        log_callback(message)
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as cs
        import rapidfuzz as rf

        sentence_transformers, cosine_similarity, rapidfuzz = (
            SentenceTransformer,
            cs,
            rf,
        )
        log_callback("Libraries loaded successfully.")
    return sentence_transformers, cosine_similarity, rapidfuzz


class EnhancedCurriculumMatcher:
    """
    The core matching engine. This class contains all the logic for data normalization,
    feature scoring, and pre-computation. It is designed to be independent of the UI.
    """

    def __init__(self, log_callback=print):
        self.model = None
        self.log_callback = log_callback

        # --- CONFIGURATION CONSTANTS FOR COLUMN NAMES ---
        # This makes the code more robust and easier to maintain if column names change.
        self.INPUT_PRODUCT_NAME = "product_name_raw"
        self.INPUT_PUBLISHER = "publisher_raw"
        self.CATALOG_ID = "product_identifier"
        self.CATALOG_PRODUCT_NAME = "product_name"
        self.CATALOG_PUBLISHER = "publisher"
        self.CATALOG_PUBLISHER_PRIOR = "publisher_prior"
        self.CATALOG_GRADES = "intended_grades"
        self.CATALOG_GRADES_2 = "intended_grades2"
        self.CATALOG_YEAR = "copyright_year"
        self.INPUT_GRADE = "grade"
        self.INPUT_SUBJECT = "subject"
        self.INPUT_PROD_TYPE = "product_type_usage"
        self.CATALOG_SUBJECT = "subject_level1"
        self.CATALOG_PROD_TYPE = "product_type"
        self.CATALOG_SERIES = "series"
        self.CATALOG_SUBJECT_L2 = "subject_level2"
        # Allow-list for subject_level2 terms to prevent adding common word noise
        self.SUBJECT_LEVEL2_ALLOWLIST = ["a|g|a", "integrated math"]

        # Tunable weights for the final score calculation.
        self.weights = {"name": 0.5, "publisher": 0.3, "grade": 0.2}

    def _log(self, message):
        """Internal logging helper."""
        self.log_callback(message)

    def load_model(self):
        """Loads the sentence-transformer model, downloading it if necessary on first run."""
        if self.model is None:
            self._log("Loading sentence transformer model...")
            SentenceTransformer, _, _ = load_heavy_imports(self.log_callback)
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._log("Model loaded.")

    def _normalize(self, text):
        """Aggressive, space-insensitive normalization for fuzzy matching."""
        if pd.isna(text):
            return ""
        text = str(text).lower().replace("&", " and ")
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation but keep spaces
        text = re.sub(r"\s+", "", text)  # Remove all spaces to make it a single token
        return text

    def _normalize_for_semantic(self, text):
        """Lighter normalization for semantic models, preserving sentence structure."""
        if pd.isna(text):
            return ""
        text = str(text).lower().replace("&", " and ")
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _get_name_score(
        self, raw_name, catalog_search_name_embedding, catalog_search_name_text
    ):
        """
        Calculates a hybrid name score using both semantic and "two-way" fuzzy matching.
        This is robust to both meaning and partial/substring matches.
        """
        if not raw_name:
            return 0.0

        # Normalize inputs for their respective scoring methods
        raw_name_sem = self._normalize_for_semantic(raw_name)
        raw_name_fuzz = self._normalize(raw_name)
        catalog_name_fuzz = self._normalize(catalog_search_name_text)

        # 1. Semantic Score (understands meaning)
        raw_embedding = self.model.encode([raw_name_sem], show_progress_bar=False)
        semantic_score = cosine_similarity(
            raw_embedding, [catalog_search_name_embedding]  # type: ignore
        )[0][
            0
        ]  # type: ignore

        # 2. Fuzzy Score (catches typos and partial matches)
        # Check for overall similarity
        fuzzy_score_1 = rapidfuzz.fuzz.token_sort_ratio(
            raw_name_fuzz, catalog_name_fuzz
        )
        # Check if raw_name is a strong partial match within the catalog name (e.g., "SEPUP")
        fuzzy_score_2 = rapidfuzz.fuzz.partial_token_sort_ratio(
            raw_name_fuzz, catalog_name_fuzz
        )

        # Take the best of the two fuzzy approaches
        best_fuzzy_score = max(fuzzy_score_1, fuzzy_score_2) / 100.0

        # Return the weighted blend
        return (0.7 * semantic_score) + (0.3 * best_fuzzy_score)

    def _get_publisher_score(self, raw_pub, catalog_pub_current, catalog_pub_prior):
        """
        Calculates publisher score against both current and prior publishers
        and returns the best (max) score. This handles acquisitions.
        """
        if not raw_pub:
            return 0.0
        norm_raw_pub = self._normalize(raw_pub)

        # Score against the current publisher
        score_current = (
            rapidfuzz.fuzz.WRatio(norm_raw_pub, self._normalize(catalog_pub_current))
            if pd.notna(catalog_pub_current)
            else 0
        )
        # Score against the prior publisher
        score_prior = (
            rapidfuzz.fuzz.WRatio(norm_raw_pub, self._normalize(catalog_pub_prior))
            if pd.notna(catalog_pub_prior)
            else 0
        )

        return max(score_current, score_prior) / 100.0

    def _parse_grade_range(self, grade_str):
        """Intelligently parses grade strings like "K-5", "9-12", "3", "PK", "12-Sep"."""
        if pd.isna(grade_str):
            return None, None
        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }

        def convert_part(part_str):
            part_str = part_str.strip().lower()
            if part_str in ["p", "k", "pk"]:
                return 0
            try:
                return int(part_str)
            except ValueError:
                return month_map.get(part_str[:3], None)

        grade_str = str(grade_str)
        if "-" in grade_str:
            parts = grade_str.split("-", 1)
            val1, val2 = convert_part(parts[0]), convert_part(parts[1])
            if val1 is not None and val2 is not None:
                return min(val1, val2), max(val1, val2)
        else:
            val = convert_part(grade_str)
            if val is not None:
                return val, val
        return None, None

    def _get_grade_score(self, raw_grade, intended_grades_str):
        """Calculates a grade score based on distance, then normalizes to a 0-1 similarity."""
        student_grade = None
        if pd.notna(raw_grade):
            raw_grade_str = str(raw_grade).strip().lower()
            if raw_grade_str in ["k", "p", "pk"]:
                student_grade = 0
            else:
                try:
                    student_grade = int(float(raw_grade_str))
                except (ValueError, TypeError):
                    student_grade = None
        if student_grade is None:
            return 0.0

        ig_low, ig_high = self._parse_grade_range(intended_grades_str)
        if ig_low is None or ig_high is None:
            return 0.0

        distance = 0
        if student_grade < ig_low:
            distance = ig_low - student_grade
        elif student_grade > ig_high:
            distance = student_grade - ig_high

        # Convert distance (0 is best) to similarity (1.0 is best)
        return 1 / (1 + distance)

    def calculate_match_score(self, input_record, catalog_candidate):
        """The central orchestrator for scoring a single input/candidate pair."""
        # Get raw values
        raw_name = input_record.get(self.INPUT_PRODUCT_NAME)
        raw_pub = input_record.get(self.INPUT_PUBLISHER)

        # Get catalog values
        cat_pub_current = catalog_candidate.get(self.CATALOG_PUBLISHER)
        cat_pub_prior = catalog_candidate.get(self.CATALOG_PUBLISHER_PRIOR)

        # Calculate all component scores
        s_name = self._get_name_score(
            raw_name,
            catalog_candidate["embedding"],
            catalog_candidate["search_name_sem"],
        )
        s_pub = self._get_publisher_score(raw_pub, cat_pub_current, cat_pub_prior)
        s_grade = self._get_grade_score(
            input_record.get(self.INPUT_GRADE),
            catalog_candidate.get(self.CATALOG_GRADES),
        )

        # Combine into a final weighted score
        final_score = (
            (self.weights["name"] * s_name)
            + (self.weights["publisher"] * s_pub)
            + (self.weights["grade"] * s_grade)
        )

        # Return all scores for detailed output
        return {
            "final_score": final_score,
            "name_score": s_name,
            "publisher_score": s_pub,
            "grade_score": s_grade,
        }

    def precompute_catalog(self, catalog_df):
        """Enriches the catalog DataFrame with computed data needed for matching."""
        self._log("Pre-computing catalog data...")

        def create_search_name(row):
            """Creates an enriched name for searching by combining name, series, and key subject tags."""
            parts = {
                self._normalize_for_semantic(row[self.CATALOG_PRODUCT_NAME]),
                self._normalize_for_semantic(row[self.CATALOG_SERIES]),
            }
            subject_l2_norm = self._normalize_for_semantic(row[self.CATALOG_SUBJECT_L2])
            if subject_l2_norm in self.SUBJECT_LEVEL2_ALLOWLIST:
                parts.add(subject_l2_norm.replace("|", " "))
            return " ".join(sorted(list(p for p in parts if p)))

        catalog_df["search_name_sem"] = catalog_df.apply(create_search_name, axis=1)

        self._log("Computing catalog embeddings on enriched names...")
        catalog_df["embedding"] = list(
            self.model.encode(
                catalog_df["search_name_sem"].tolist(),
                batch_size=32,
                show_progress_bar=True,
            )
        )

        self._log("Catalog pre-computation complete.")
        return catalog_df


class MatcherApp:
    """The main application class that handles the GUI and the processing logic."""

    def __init__(self, headless=False):
        self.headless = headless
        self.root = None
        self.matcher = EnhancedCurriculumMatcher(
            log_callback=self.log if not headless else print
        )
        if not self.headless:
            self.root = tk.Tk()
            self._setup_gui()

    def log(self, message):
        """Thread-safe logging to the GUI."""
        if self.root:
            self.root.after(0, lambda: self._log_message(message))

    def _log_message(self, message):
        """Internal method to update the log text widget."""
        self.log_text.insert(
            tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
        )
        self.log_text.see(tk.END)

    def run(self):
        """Starts the application, either the GUI or the headless process."""
        if self.headless:
            self._run_headless_mode()
        else:
            self.root.mainloop()

    def _setup_gui(self):
        """Builds the entire Tkinter user interface."""
        self.root.title("Enhanced Curriculum Matcher")
        self.root.geometry("800x750")
        self.input_file = tk.StringVar()
        self.catalog_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.mode = tk.StringVar(value="QA")
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        # Mode Selection Frame
        mode_frame = ttk.LabelFrame(main_frame, text="1. Select Operating Mode")
        mode_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        ttk.Radiobutton(
            mode_frame,
            text="QA Scorer (Validate existing matches)",
            variable=self.mode,
            value="QA",
        ).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(
            mode_frame,
            text="Automated Matcher (Find new matches)",
            variable=self.mode,
            value="Matcher",
        ).pack(anchor=tk.W, padx=10, pady=2)
        # File/Directory Selection Frame
        file_frame = ttk.LabelFrame(main_frame, text="2. Select Files & Directory")
        file_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self._create_file_selector(file_frame, "Input Data File:", self.input_file)
        self._create_file_selector(
            file_frame, "Product Catalog File:", self.catalog_file
        )
        self._create_directory_selector(
            file_frame, "Output Directory:", self.output_dir
        )
        # Action Button
        self.process_button = ttk.Button(
            main_frame,
            text="Start Processing",
            command=self._start_processing_thread,
            state="disabled",
        )
        self.process_button.grid(row=2, column=0, columnspan=2, pady=20)
        # Status and Progress Frame
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
        # Log Frame
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

    def _create_file_selector(self, parent, label_text, string_var):
        """Helper to create a file selection row in the GUI."""
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
        """Helper to create a directory selection row in the GUI."""
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
        """Handles the file dialog logic."""
        path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            string_var.set(path)
            self._check_ready()

    def _select_directory(self, string_var):
        """Handles the directory dialog logic."""
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            string_var.set(path)
            self._check_ready()

    def _check_ready(self):
        """Enables the 'Start' button only when all inputs are provided."""
        if all([self.input_file.get(), self.catalog_file.get(), self.output_dir.get()]):
            self.process_button.config(state="normal")
            self.status_label.config(text="Ready to process.")

    def _update_progress(self, current, total):
        """Thread-safe method to update the progress bar."""

        def update_gui():
            if total > 0:
                self.progress_var.set((current / total) * 100)
            self.status_label.config(text=f"Processing: {current}/{total}")

        if self.root:
            self.root.after(0, update_gui)

    def _start_processing_thread(self):
        """Starts the main processing logic in a separate thread to keep the GUI responsive."""
        self.process_button.config(state="disabled")
        thread = threading.Thread(
            target=self._run_processing_logic,
            args=(
                self.input_file.get(),
                self.catalog_file.get(),
                self.output_dir.get(),
                self.mode.get(),
            ),
        )
        thread.daemon = True
        thread.start()

    def _run_headless_mode(self):
        """Entry point for running the script from the command line without a GUI."""
        # --- CONFIGURE FILE PATHS FOR COLAB/HEADLESS MODE HERE ---
        input_file_path = "/content/your_input_data.csv"
        catalog_file_path = "/content/product_catalog.csv"
        output_dir_path = "/content/results/"
        mode = "Matcher"
        print("--- RUNNING IN HEADLESS MODE ---")
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        self._run_processing_logic(
            input_file_path, catalog_file_path, output_dir_path, mode
        )

    def _run_processing_logic(self, input_path, catalog_path, output_dir, mode):
        """The main orchestration function that loads data and calls the correct processing mode."""
        try:
            log_func = self.log if not self.headless else print
            progress_func = (
                self._update_progress
                if not self.headless
                else lambda c, t: None if c % 1000 == 0 else None
            )

            log_func(f"Using Input File: {os.path.basename(input_path)}")
            log_func("Loading data files...")
            input_df = pd.read_csv(input_path, encoding="latin-1")
            catalog_df = pd.read_csv(catalog_path, encoding="latin-1")
            log_func(
                f"Loaded {len(input_df)} input records and {len(catalog_df)} catalog records."
            )

            self.matcher.load_model()
            catalog_df = self.matcher.precompute_catalog(catalog_df)

            log_func(f"Starting process in {mode} Mode...")
            if mode == "QA":
                results_df = self._run_qa_scorer(input_df, catalog_df, progress_func)
            else:
                results_df = self._run_automated_matcher(
                    input_df, catalog_df, progress_func
                )

            log_func(
                f"Processing complete. Saving {len(results_df)} rows of results..."
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{mode}_Results_{timestamp}.csv"
            output_path = os.path.join(output_dir, base_filename)
            results_df.to_csv(output_path, index=False)
            log_func(f"Results saved successfully to: {output_path}")
            if not self.headless:
                messagebox.showinfo("Success", "Processing complete!")

        except Exception as e:
            log_func(f"FATAL ERROR: {e}")
            if not self.headless:
                messagebox.showerror("Error", f"An error occurred:\n\n{e}")
        finally:
            if not self.headless:
                self.root.after(0, lambda: self.process_button.config(state="normal"))

    def _run_qa_scorer(self, input_df, catalog_df, progress_callback):
        """Runs the QA process on pre-matched data."""
        results = []
        catalog_lookup = catalog_df.set_index(self.matcher.CATALOG_ID)
        for idx, row in input_df.iterrows():
            progress_callback(idx + 1, len(input_df))
            human_match_id = row.get(self.matcher.CATALOG_ID)
            result_row = row.to_dict()
            if pd.notna(human_match_id) and human_match_id in catalog_lookup.index:
                candidate = catalog_lookup.loc[human_match_id]
                scores = self.matcher.calculate_match_score(row, candidate)
                result_row.update(scores)
            else:
                result_row["final_score"] = 0.0
            results.append(result_row)
        return pd.DataFrame(results)

    def _run_automated_matcher(self, input_df, catalog_df, progress_callback):
        """Runs the matching process to find new matches for raw data."""
        all_results = []
        catalog_lookup = catalog_df.set_index(self.matcher.CATALOG_ID)

        for idx, row in input_df.iterrows():
            progress_callback(idx + 1, len(input_df))
            # Stage 1: Candidate Generation
            primary_mask = (
                catalog_df[self.matcher.CATALOG_SUBJECT].str.lower()
                == self.matcher._normalize_for_semantic(
                    row.get(self.matcher.INPUT_SUBJECT)
                )
            ) & (
                catalog_df[self.matcher.CATALOG_PROD_TYPE].str.lower()
                == self.matcher._normalize_for_semantic(
                    row.get(self.matcher.INPUT_PROD_TYPE)
                )
            )
            primary_candidates = catalog_df[primary_mask]
            top_candidates_df = pd.DataFrame()
            if not primary_candidates.empty:
                raw_embedding = self.matcher.model.encode(
                    [
                        self.matcher._normalize_for_semantic(
                            row.get(self.matcher.INPUT_PRODUCT_NAME)
                        )
                    ],
                    show_progress_bar=False,
                )
                sims = cosine_similarity(
                    raw_embedding, np.vstack(primary_candidates["embedding"].values)
                )[  # type: ignore
                    0
                ]  # type: ignore
                top_indices_local = np.argsort(sims)[-5:][::-1]
                top_candidates_df = primary_candidates.iloc[top_indices_local]

            # Stage 2: Detailed Scoring
            scored_candidates = []
            for _, cand in top_candidates_df.iterrows():
                scores = self.matcher.calculate_match_score(row, cand)
                scores[self.matcher.CATALOG_ID] = cand[self.matcher.CATALOG_ID]
                scored_candidates.append(scores)

            scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

            # Format output row
            result_row = row.to_dict()
            for i in range(5):
                if i < len(scored_candidates):
                    match = scored_candidates[i]
                    match_id = match[self.matcher.CATALOG_ID]
                    result_row[f"match_{i+1}_id"] = match_id
                    for key, val in match.items():
                        if key != self.matcher.CATALOG_ID:
                            result_row[f"match_{i+1}_{key}"] = round(val, 4)
                    if match_id in catalog_lookup.index:
                        catalog_match_row = catalog_lookup.loc[match_id]
                        result_row[f"match_{i+1}_catalog_product_name"] = (
                            catalog_match_row.get(self.matcher.CATALOG_PRODUCT_NAME)
                        )
                        result_row[f"match_{i+1}_catalog_series"] = (
                            catalog_match_row.get(self.matcher.CATALOG_SERIES)
                        )
                        result_row[f"match_{i+1}_catalog_supplier_name"] = (
                            catalog_match_row.get(self.matcher.CATALOG_PUBLISHER)
                        )
                        result_row[f"match_{i+1}_catalog_copyright_year"] = (
                            catalog_match_row.get(self.matcher.CATALOG_YEAR)
                        )
                        grades_col = (
                            self.matcher.CATALOG_GRADES_2
                            if self.matcher.CATALOG_GRADES_2 in catalog_match_row
                            else self.matcher.CATALOG_GRADES
                        )
                        result_row[f"match_{i+1}_catalog_intended_grades"] = (
                            catalog_match_row.get(grades_col)
                        )
                else:
                    result_row[f"match_{i+1}_id"] = ""
            all_results.append(result_row)
        return pd.DataFrame(all_results)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Curriculum Matcher with optional GUI."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without GUI (for Colab/servers).",
    )
    args = parser.parse_args()

    app = MatcherApp(headless=args.headless)
    app.run()


if __name__ == "__main__":
    main()
