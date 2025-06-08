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

# Defer heavy imports until needed
sentence_transformers = None
cosine_similarity = None
rapidfuzz = None

# --- CORE LOGIC & HELPER FUNCTIONS ---


def load_heavy_imports(log_callback=None):
    """Load heavy imports only when needed"""
    global sentence_transformers, cosine_similarity, rapidfuzz
    if sentence_transformers is None:
        message = "Loading required libraries (this may take a moment)..."
        if log_callback:
            log_callback(message)
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as cs
        import rapidfuzz as rf

        sentence_transformers, cosine_similarity, rapidfuzz = (
            SentenceTransformer,
            cs,
            rf,
        )
        if log_callback:
            log_callback("Libraries loaded successfully.")
    return sentence_transformers, cosine_similarity, rapidfuzz


class EnhancedCurriculumMatcher:
    def __init__(self, log_callback=None):
        self.model = None
        self.log_callback = log_callback
        self.weights = {
            "name": 0.4,
            "publisher": 0.25,
            "year": 0.1,
            "grade": 0.1,
            "state": 0.15,
        }

    def _log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def load_model(self):
        if self.model is None:
            self._log("Loading sentence transformer model...")
            SentenceTransformer, _, _ = load_heavy_imports(self.log_callback)
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._log("Model loaded.")

    def _normalize(self, text):
        return re.sub(r"\s+", " ", str(text).strip().lower()) if pd.notna(text) else ""

    def _get_name_score(self, raw_name, catalog_name_embedding, catalog_name_text):
        if not raw_name:
            return 0.0
        raw_embedding = self.model.encode([raw_name], show_progress_bar=False)
        semantic_score = cosine_similarity(raw_embedding, [catalog_name_embedding])[0][
            0
        ]
        fuzzy_score = (
            rapidfuzz.fuzz.token_sort_ratio(raw_name, catalog_name_text) / 100.0
        )
        return (0.7 * semantic_score) + (0.3 * fuzzy_score)

    def _get_publisher_score(self, raw_pub, catalog_pub, pub_dict):
        if not raw_pub or not catalog_pub:
            return 0.0
        standardized_pub = pub_dict.get(raw_pub)
        if standardized_pub:
            return 1.0 if standardized_pub == catalog_pub else 0.0
        return rapidfuzz.fuzz.WRatio(raw_pub, catalog_pub) / 100.0

    def _get_year_score(self, raw_year_str, adoption_year_str, catalog_year):
        score = 0.0
        try:
            raw_year = int(float(raw_year_str)) if pd.notna(raw_year_str) else None
            catalog_year_val = (
                int(float(catalog_year)) if pd.notna(catalog_year) else None
            )
            if raw_year and catalog_year_val:
                year_diff = abs(raw_year - catalog_year_val)
                score = max(0.0, 1.0 - (year_diff / 10.0))
            # Adoption year penalty
            adoption_year = (
                int(float(adoption_year_str)) if pd.notna(adoption_year_str) else None
            )
            if adoption_year and catalog_year_val and catalog_year_val > adoption_year:
                score *= 0.25  # Apply heavy penalty
        except (ValueError, TypeError):
            pass
        return score

    def _get_grade_score(self, raw_grade, catalog_grade):
        def parse_grades(g_str):
            grades = set()
            if pd.isna(g_str):
                return grades
            for part in re.split(r"[,\s]+", str(g_str)):
                if "-" in part:
                    try:
                        start, end = map(int, part.split("-"))
                        grades.update(range(start, end + 1))
                    except:
                        pass
                elif part.isdigit():
                    grades.add(int(part))
            return grades

        raw_set, cat_set = parse_grades(raw_grade), parse_grades(catalog_grade)
        if not raw_set or not cat_set:
            return 0.0
        intersection = len(raw_set.intersection(cat_set))
        union = len(raw_set.union(cat_set))
        return intersection / union if union > 0 else 0.0

    def _get_state_score(self, raw_is_specific, raw_state, cat_is_specific, cat_state):
        if raw_is_specific and cat_is_specific:
            return (
                1.0 if self._normalize(raw_state) == self._normalize(cat_state) else 0.0
            )
        if raw_is_specific and not cat_is_specific:
            return 0.5
        if not raw_is_specific and cat_is_specific:
            return 0.3
        if not raw_is_specific and not cat_is_specific:
            return 1.0
        return 0.0

    def calculate_match_score(self, input_record, catalog_candidate, pub_dict):
        """Calculates the full set of scores for one input-candidate pair."""
        raw_name = self._normalize(input_record.get("product_name_raw"))
        raw_pub = self._normalize(input_record.get("publisher_raw"))
        cat_pub = self._normalize(catalog_candidate.get("supplier_name"))
        s_name = self._get_name_score(
            raw_name,
            catalog_candidate["embedding"],
            catalog_candidate["product_name_norm"],
        )
        s_pub = self._get_publisher_score(raw_pub, cat_pub, pub_dict)
        s_year = self._get_year_score(
            input_record.get("copyright_year_raw"),
            input_record.get("adoption_year"),
            catalog_candidate.get("copyright_year"),
        )
        s_grade = self._get_grade_score(
            input_record.get("grade"), catalog_candidate.get("intended_grades")
        )
        s_state = self._get_state_score(
            input_record.get("selection_state_specific_version"),
            input_record.get("state"),
            catalog_candidate.get("state_specific_version"),
            catalog_candidate.get("derived_state"),
        )
        final_score = (
            (self.weights["name"] * s_name)
            + (self.weights["publisher"] * s_pub)
            + (self.weights["year"] * s_year)
            + (self.weights["grade"] * s_grade)
            + (self.weights["state"] * s_state)
        )
        return {
            "final_score": final_score,
            "name_score": s_name,
            "publisher_score": s_pub,
            "year_score": s_year,
            "grade_score": s_grade,
            "state_score": s_state,
        }

    def precompute_catalog(self, catalog_df):
        self._log("Pre-computing catalog data...")
        catalog_df["product_name_norm"] = catalog_df["product_name"].apply(
            self._normalize
        )
        self._log("Deriving states from catalog product names...")

        def derive_state(row):
            if row["state_specific_version"]:
                match = re.search(
                    r"\b(alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|virginia|washington|west virginia|wisconsin|wyoming)\b|\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b",
                    row["product_name"],
                    re.IGNORECASE,
                )
                if match:
                    return match.group(0).upper()
            return None

        catalog_df["derived_state"] = catalog_df.apply(derive_state, axis=1)
        self._log("Computing catalog embeddings (this may take a minute)...")
        catalog_df["embedding"] = list(
            self.model.encode(
                catalog_df["product_name_norm"].tolist(),
                batch_size=32,
                show_progress_bar=False,
            )
        )
        self._log("Catalog pre-computation complete.")
        return catalog_df


# --- GUI CLASS ---


class MatcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Curriculum Matcher")
        self.root.geometry("800x750")

        self.matcher = EnhancedCurriculumMatcher(log_callback=self.log_message)
        self.input_file = tk.StringVar()
        self.catalog_file = tk.StringVar()
        self.pub_dict_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.mode = tk.StringVar(value="QA")

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

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

        file_frame = ttk.LabelFrame(main_frame, text="2. Select Files & Directory")
        file_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self._create_file_selector(file_frame, "Input Data File:", self.input_file)
        self._create_file_selector(
            file_frame, "Product Catalog File:", self.catalog_file
        )
        self._create_file_selector(
            file_frame, "Publisher Dictionary (JSON):", self.pub_dict_file
        )
        self._create_directory_selector(
            file_frame, "Output Directory:", self.output_dir
        )

        self.process_button = ttk.Button(
            main_frame,
            text="Start Processing",
            command=self.start_processing,
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
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)
        entry = ttk.Entry(frame, textvariable=string_var, state="readonly")
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        command = lambda: self.select_file(string_var)
        ttk.Button(frame, text="Browse...", command=command).pack(side=tk.LEFT, padx=5)

    def _create_directory_selector(self, parent, label_text, string_var):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)
        entry = ttk.Entry(frame, textvariable=string_var, state="readonly")
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        command = lambda: self.select_directory(string_var)
        ttk.Button(frame, text="Browse...", command=command).pack(side=tk.LEFT, padx=5)

    def select_file(self, string_var):
        if string_var == self.pub_dict_file:
            filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        else:
            filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
        if path:
            string_var.set(path)
            self.check_ready()

    def select_directory(self, string_var):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            string_var.set(path)
            self.check_ready()

    def check_ready(self):
        if all(
            [
                self.input_file.get(),
                self.catalog_file.get(),
                self.pub_dict_file.get(),
                self.output_dir.get(),
            ]
        ):
            self.process_button.config(state="normal")
            self.status_label.config(text="Ready to process.")

    def log_message(self, message):
        def update_gui():
            self.log_text.insert(
                tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
            )
            self.log_text.see(tk.END)

        if self.root:
            self.root.after(0, update_gui)

    def update_progress(self, current, total):
        def update_gui():
            if total > 0:
                self.progress_var.set((current / total) * 100)
            self.status_label.config(text=f"Processing: {current}/{total}")

        if self.root:
            self.root.after(0, update_gui)

    def start_processing(self):
        self.process_button.config(state="disabled")
        thread = threading.Thread(target=self.run_processing_logic)
        thread.daemon = True
        thread.start()

    def run_processing_logic(self):
        try:
            self.log_message("Loading all data files...")
            input_df = pd.read_csv(self.input_file.get(), encoding="latin-1")
            catalog_df = pd.read_csv(self.catalog_file.get(), encoding="latin-1")
            with open(self.pub_dict_file.get(), "r") as f:
                pub_dict = json.load(f)
            self.log_message("All files loaded.")

            # *** THIS IS THE FIX ***
            # Call load_heavy_imports with the correct GUI logging method
            load_heavy_imports(self.log_message)

            self.matcher.load_model()
            catalog_df = self.matcher.precompute_catalog(catalog_df)

            mode = self.mode.get()
            self.log_message(f"Starting process in {mode} Mode...")
            if mode == "QA":
                results_df = self._run_qa_scorer(input_df, catalog_df, pub_dict)
            else:
                results_df = self._run_automated_matcher(input_df, catalog_df, pub_dict)

            self.log_message(
                f"Processing complete. Saving {len(results_df)} rows of results..."
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{mode}_Results_{timestamp}.csv"
            output_path = os.path.join(self.output_dir.get(), base_filename)

            results_df.to_csv(output_path, index=False)
            self.log_message(f"Results saved successfully to: {output_path}")
            messagebox.showinfo("Success", "Processing complete!")

        except Exception as e:
            self.log_message(f"ERROR: {e}")
            messagebox.showerror("Error", f"An error occurred:\n\n{e}")
        finally:
            if self.root:
                self.root.after(0, lambda: self.process_button.config(state="normal"))

    def _run_qa_scorer(self, input_df, catalog_df, pub_dict):
        results = []
        catalog_lookup = catalog_df.set_index("product_identifier")
        for idx, row in input_df.iterrows():
            self.update_progress(idx + 1, len(input_df))
            human_match_id = row.get("product_identifier")
            result_row = row.to_dict()
            if pd.notna(human_match_id) and human_match_id in catalog_lookup.index:
                candidate = catalog_lookup.loc[human_match_id]
                scores = self.matcher.calculate_match_score(row, candidate, pub_dict)
                result_row.update(scores)
            else:
                result_row["final_score"] = 0.0
            results.append(result_row)
        return pd.DataFrame(results)

    def _run_automated_matcher(self, input_df, catalog_df, pub_dict):
        all_results = []
        for idx, row in input_df.iterrows():
            self.update_progress(idx + 1, len(input_df))
            primary_mask = (
                catalog_df["subject_level1"].str.lower()
                == self.matcher._normalize(row.get("subject"))
            ) & (
                catalog_df["product_type"].str.lower()
                == self.matcher._normalize(row.get("product_type_usage"))
            )
            primary_candidates = catalog_df[primary_mask]

            top_candidates_df = pd.DataFrame()
            if not primary_candidates.empty:
                raw_embedding = self.matcher.model.encode(
                    [self.matcher._normalize(row.get("product_name_raw"))],
                    show_progress_bar=False,
                )
                sims = cosine_similarity(
                    raw_embedding, np.vstack(primary_candidates["embedding"].values)
                )[0]
                top_indices_local = np.argsort(sims)[-20:][::-1]
                top_candidates_df = primary_candidates.iloc[top_indices_local]

            scored_candidates = []
            for _, cand in top_candidates_df.iterrows():
                scores = self.matcher.calculate_match_score(row, cand, pub_dict)
                scores["product_identifier"] = cand["product_identifier"]
                scored_candidates.append(scores)

            scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

            result_row = row.to_dict()
            for i in range(3):
                if i < len(scored_candidates):
                    match = scored_candidates[i]
                    result_row[f"match_{i+1}_id"] = match["product_identifier"]
                    for key, val in match.items():
                        if key != "product_identifier":
                            result_row[f"match_{i+1}_{key}"] = round(val, 4)
                else:
                    result_row[f"match_{i+1}_id"] = ""
            all_results.append(result_row)
        return pd.DataFrame(all_results)


def main():
    root = tk.Tk()
    app = MatcherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
