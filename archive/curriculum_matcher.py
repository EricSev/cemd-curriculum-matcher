#!/usr/bin/env python3
"""
Curriculum Matcher Application
Matches raw curriculum data to standardized product catalog
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from tqdm import tqdm
import threading
import os


class CurriculumMatcher:
    def __init__(self):
        self.model = None
        self.grade_order = [
            "PK",
            "TK",
            "K",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
        ]

    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

    def normalize_text(self, text):
        """Basic text normalization"""
        if pd.isna(text) or text is None:
            return ""

        # Convert to string and basic cleanup
        text = str(text).strip()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep alphanumeric, spaces, hyphens, periods
        text = re.sub(r"[^\w\s\-\.]", " ", text)
        # Remove extra spaces again
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def grade_to_position(self, grade):
        """Convert grade to numeric position"""
        try:
            return self.grade_order.index(str(grade))
        except ValueError:
            if str(grade) == "NI":
                return None
            return -1

    def parse_intended_grades(self, intended_grades_str):
        """Parse 'K-5' format to get min/max bounds"""
        if pd.isna(intended_grades_str) or not intended_grades_str:
            return None, None

        intended_grades_str = str(intended_grades_str).strip()
        if "-" in intended_grades_str:
            parts = intended_grades_str.split("-")
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()

        # Single grade
        return intended_grades_str, intended_grades_str

    def calculate_grade_score(self, raw_grade, intended_grades_range, tolerance=1):
        """Calculate grade alignment score"""
        if pd.isna(raw_grade) or str(raw_grade) == "NI":
            return 1.0, "Grade not specified (NI) - neutral", False

        if pd.isna(intended_grades_range):
            return 0.5, "No intended grades specified", True

        grade_min, grade_max = self.parse_intended_grades(intended_grades_range)
        if grade_min is None or grade_max is None:
            return 0.5, "Invalid intended grades format", True

        raw_position = self.grade_to_position(raw_grade)
        min_position = self.grade_to_position(grade_min)
        max_position = self.grade_to_position(grade_max)

        if raw_position == -1:
            return 0.5, f"Unknown grade format: {raw_grade}", True

        # Perfect match
        if min_position <= raw_position <= max_position:
            return (
                1.0,
                f"Perfect grade match ({raw_grade} within {intended_grades_range})",
                False,
            )

        # Calculate proximity
        distance_to_min = abs(raw_position - min_position)
        distance_to_max = abs(raw_position - max_position)
        nearest_distance = min(distance_to_min, distance_to_max)

        if nearest_distance <= tolerance:
            score = 0.85 - (nearest_distance * 0.1)
            explanation = f"Close match (distance: {nearest_distance})"
            flag_for_review = False
        elif nearest_distance <= 3:
            score = 0.6 - ((nearest_distance - tolerance) * 0.1)
            explanation = f"Moderate mismatch (distance: {nearest_distance})"
            flag_for_review = True
        else:
            score = 0.2
            explanation = f"Severe grade mismatch (distance: {nearest_distance})"
            flag_for_review = True

        return score, explanation, flag_for_review

    def calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using sentence transformers"""
        if not text1 or not text2:
            return 0.0

        try:
            embeddings = self.model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except:
            return 0.0

    def calculate_fuzzy_similarity(self, text1, text2):
        """Calculate fuzzy string similarity"""
        if not text1 or not text2:
            return 0.0

        return fuzz.ratio(text1.lower(), text2.lower()) / 100.0

    def calculate_copyright_score(self, raw_year, catalog_year):
        """Calculate copyright year similarity"""
        if pd.isna(raw_year) or raw_year == "":
            return 0.8, "Missing copyright year"

        if pd.isna(catalog_year) or catalog_year == "":
            return 0.5, "No catalog copyright year"

        try:
            raw_year = int(float(str(raw_year)))
            catalog_year = int(float(str(catalog_year)))

            if raw_year == catalog_year:
                return 1.0, "Exact copyright match"

            diff = abs(raw_year - catalog_year)
            if diff <= 1:
                return 0.9, f"Close copyright match (diff: {diff})"
            elif diff <= 3:
                return 0.7, f"Moderate copyright match (diff: {diff})"
            else:
                return 0.3, f"Poor copyright match (diff: {diff})"
        except:
            return 0.5, "Invalid copyright year format"

    def calculate_type_subject_score(self, raw_record, catalog_record):
        """Calculate type and subject alignment score"""
        # Type matching
        raw_type = str(raw_record.get("product_type_usage", "")).strip()
        catalog_type = str(catalog_record.get("product_type", "")).strip()

        if raw_type.lower() == catalog_type.lower():
            type_score = 1.0
        else:
            type_score = 0.0

        # Subject matching
        raw_subject = str(raw_record.get("subject", "")).strip()
        catalog_subject = str(catalog_record.get("subject_level1", "")).strip()

        if catalog_subject.lower() == "all":
            subject_score = 0.9
        elif raw_subject.lower() == catalog_subject.lower():
            subject_score = 1.0
        else:
            subject_score = 0.0

        return (type_score + subject_score) / 2

    def filter_catalog_candidates(self, raw_record, catalog_df):
        """Filter catalog to relevant candidates"""
        product_type = raw_record.get("product_type_usage", "")
        subject = raw_record.get("subject", "")

        # Filter by product type
        type_filter = catalog_df["product_type"] == product_type

        # Filter by subject (include 'All' subjects)
        subject_filter = (catalog_df["subject_level1"] == subject) | (
            catalog_df["subject_level1"] == "All"
        )

        candidates = catalog_df[type_filter & subject_filter].copy()

        # If no candidates with exact type, try with subject only
        if candidates.empty:
            candidates = catalog_df[subject_filter].copy()

        # If still no candidates, return all (let scoring handle it)
        if candidates.empty:
            candidates = catalog_df.copy()

        return candidates

    def calculate_match_score(self, raw_record, catalog_record):
        """Calculate comprehensive match score"""
        scores = {}
        explanations = []
        flags = []

        # Normalize text fields
        raw_product = self.normalize_text(raw_record.get("product_name_raw", ""))
        catalog_product = self.normalize_text(catalog_record.get("product_name", ""))
        raw_publisher = self.normalize_text(raw_record.get("publisher_raw", ""))
        catalog_publisher = self.normalize_text(catalog_record.get("publisher", ""))

        # Product name similarity (semantic + fuzzy)
        if raw_product and catalog_product:
            semantic_sim = self.calculate_semantic_similarity(
                raw_product, catalog_product
            )
            fuzzy_sim = self.calculate_fuzzy_similarity(raw_product, catalog_product)
            scores["product_name"] = (semantic_sim * 0.7) + (fuzzy_sim * 0.3)
        else:
            scores["product_name"] = 0.0
            explanations.append("Missing product name data")
            flags.append(True)

        # Publisher similarity
        if raw_publisher and catalog_publisher:
            scores["publisher"] = self.calculate_fuzzy_similarity(
                raw_publisher, catalog_publisher
            )
        elif not raw_publisher:
            scores["publisher"] = 0.8  # Small negative for missing publisher
            explanations.append("Missing publisher data")
        else:
            scores["publisher"] = 0.0

        # Copyright year
        copyright_score, copyright_explanation = self.calculate_copyright_score(
            raw_record.get("copyright_year_raw"), catalog_record.get("copyright_year")
        )
        scores["copyright"] = copyright_score
        explanations.append(copyright_explanation)

        # Type and subject alignment
        scores["type_subject"] = self.calculate_type_subject_score(
            raw_record, catalog_record
        )

        # Grade alignment
        grade_score, grade_explanation, grade_flag = self.calculate_grade_score(
            raw_record.get("grade"), catalog_record.get("intended_grades")
        )
        scores["grade"] = grade_score
        explanations.append(grade_explanation)
        flags.append(grade_flag)

        # Calculate weighted final score
        weights = {
            "product_name": 0.35,
            "publisher": 0.25,
            "type_subject": 0.15,
            "copyright": 0.15,
            "grade": 0.10,
        }

        base_score = sum(scores[field] * weight for field, weight in weights.items())

        # Apply grade-based confidence multiplier
        if grade_score >= 0.8:
            confidence_multiplier = 1.0
        elif grade_score >= 0.6:
            confidence_multiplier = 0.95
        elif grade_score >= 0.4:
            confidence_multiplier = 0.8
        else:
            confidence_multiplier = 0.6

        final_score = base_score * confidence_multiplier

        return {
            "final_score": final_score,
            "component_scores": scores,
            "explanations": explanations,
            "flags": any(flags),
        }

    def find_top_matches(self, raw_record, catalog_df):
        """Find top 3 matches for a raw record"""
        # Filter candidates
        candidates = self.filter_catalog_candidates(raw_record, catalog_df)

        if candidates.empty:
            return []

        # Calculate scores for all candidates
        results = []
        for _, catalog_record in candidates.iterrows():
            match_result = self.calculate_match_score(raw_record, catalog_record)
            results.append(
                {
                    "product_identifier": catalog_record.get("product_identifier", ""),
                    "score": match_result["final_score"],
                    "details": match_result,
                }
            )

        # Sort by score and return top 3
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:3]

    def process_matching(self, raw_df, catalog_df, progress_callback=None):
        """Process all matching with progress tracking"""
        self.load_model()

        results = []
        total_rows = len(raw_df)

        for idx, (_, raw_record) in enumerate(raw_df.iterrows()):
            # Find top matches
            top_matches = self.find_top_matches(raw_record, catalog_df)

            # Prepare result row
            result_row = raw_record.to_dict()

            # Add match columns
            for i in range(3):
                if i < len(top_matches):
                    result_row[f"match_{i+1}_id"] = top_matches[i]["product_identifier"]
                    result_row[f"match_{i+1}_score"] = round(top_matches[i]["score"], 4)
                else:
                    result_row[f"match_{i+1}_id"] = ""
                    result_row[f"match_{i+1}_score"] = 0.0

            results.append(result_row)

            # Update progress
            if progress_callback:
                progress_callback(idx + 1, total_rows)

        return pd.DataFrame(results)


class MatcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Curriculum Matcher")
        self.root.geometry("600x400")

        self.matcher = CurriculumMatcher()
        self.raw_file = None
        self.catalog_file = None

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # File selection
        ttk.Label(main_frame, text="Select Raw Input File (CSV):").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.raw_file_label = ttk.Label(
            main_frame, text="No file selected", foreground="gray"
        )
        self.raw_file_label.grid(row=1, column=0, sticky=tk.W)
        ttk.Button(main_frame, text="Browse", command=self.select_raw_file).grid(
            row=1, column=1, padx=10
        )

        ttk.Label(main_frame, text="Select Product Catalog File (CSV):").grid(
            row=2, column=0, sticky=tk.W, pady=(20, 5)
        )
        self.catalog_file_label = ttk.Label(
            main_frame, text="No file selected", foreground="gray"
        )
        self.catalog_file_label.grid(row=3, column=0, sticky=tk.W)
        ttk.Button(main_frame, text="Browse", command=self.select_catalog_file).grid(
            row=3, column=1, padx=10
        )

        # Process button
        self.process_button = ttk.Button(
            main_frame, text="Process Matching", command=self.start_processing
        )
        self.process_button.grid(row=4, column=0, columnspan=2, pady=20)
        self.process_button.config(state="disabled")

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.grid(
            row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )

        # Status label
        self.status_label = ttk.Label(main_frame, text="Select both files to begin")
        self.status_label.grid(row=6, column=0, columnspan=2, pady=10)

        # Log text area
        self.log_text = tk.Text(main_frame, height=10, width=70)
        self.log_text.grid(row=7, column=0, columnspan=2, pady=10)

        # Scrollbar for log
        scrollbar = ttk.Scrollbar(
            main_frame, orient="vertical", command=self.log_text.yview
        )
        scrollbar.grid(row=7, column=2, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def log_message(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()

    def select_raw_file(self):
        filename = filedialog.askopenfilename(
            title="Select Raw Input CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filename:
            self.raw_file = filename
            self.raw_file_label.config(
                text=os.path.basename(filename), foreground="black"
            )
            self.log_message(f"Selected raw file: {os.path.basename(filename)}")
            self.check_ready()

    def select_catalog_file(self):
        filename = filedialog.askopenfilename(
            title="Select Product Catalog CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filename:
            self.catalog_file = filename
            self.catalog_file_label.config(
                text=os.path.basename(filename), foreground="black"
            )
            self.log_message(f"Selected catalog file: {os.path.basename(filename)}")
            self.check_ready()

    def check_ready(self):
        if self.raw_file and self.catalog_file:
            self.process_button.config(state="normal")
            self.status_label.config(text="Ready to process")

    def update_progress(self, current, total):
        """Update progress bar"""
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.status_label.config(text=f"Processing: {current}/{total} records")
        self.root.update()

    def start_processing(self):
        """Start processing in a separate thread"""
        self.process_button.config(state="disabled")
        self.log_message("Starting processing...")

        # Start processing thread
        thread = threading.Thread(target=self.process_files)
        thread.daemon = True
        thread.start()

    def process_files(self):
        try:
            # Load files
            self.log_message("Loading raw input file...")
            raw_df = pd.read_csv(self.raw_file)
            self.log_message(f"Loaded {len(raw_df)} raw records")

            self.log_message("Loading catalog file...")
            catalog_df = pd.read_csv(self.catalog_file)
            self.log_message(f"Loaded {len(catalog_df)} catalog records")

            # Validate required columns
            required_raw_cols = [
                "product_type_usage",
                "subject",
                "product_name_raw",
                "grade",
            ]
            required_catalog_cols = [
                "product_identifier",
                "product_type",
                "subject_level1",
                "product_name",
            ]

            missing_raw = [
                col for col in required_raw_cols if col not in raw_df.columns
            ]
            missing_catalog = [
                col for col in required_catalog_cols if col not in catalog_df.columns
            ]

            if missing_raw or missing_catalog:
                error_msg = ""
                if missing_raw:
                    error_msg += f"Missing raw file columns: {missing_raw}\n"
                if missing_catalog:
                    error_msg += f"Missing catalog file columns: {missing_catalog}"
                raise ValueError(error_msg)

            # Process matching
            self.log_message("Starting matching process...")
            results_df = self.matcher.process_matching(
                raw_df, catalog_df, progress_callback=self.update_progress
            )

            # Save results
            self.log_message("Processing complete. Choose save location...")
            output_file = filedialog.asksaveasfilename(
                title="Save Results As",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )

            if output_file:
                results_df.to_csv(output_file, index=False)
                self.log_message(f"Results saved to: {os.path.basename(output_file)}")
                messagebox.showinfo(
                    "Success",
                    f"Processing complete!\nResults saved to: {os.path.basename(output_file)}",
                )
            else:
                self.log_message("Save cancelled by user")

        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Error", error_msg)

        finally:
            self.process_button.config(state="normal")
            self.progress_var.set(0)
            self.status_label.config(text="Ready to process")


def main():
    root = tk.Tk()
    app = MatcherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
