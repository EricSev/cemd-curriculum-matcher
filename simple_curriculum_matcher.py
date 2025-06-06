#!/usr/bin/env python3
"""
Simple Concatenated Curriculum Matcher
Fast matching using single concatenated string comparison
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re
import threading
import os
import time
from datetime import datetime

# Defer heavy imports until needed
sentence_transformers = None
cosine_similarity = None
tqdm = None


def load_heavy_imports():
    """Load heavy imports only when needed"""
    global sentence_transformers, cosine_similarity, tqdm
    if sentence_transformers is None:
        print("Loading required libraries...")
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as cs
        from tqdm import tqdm as tqdm_import

        sentence_transformers = SentenceTransformer
        cosine_similarity = cs
        tqdm = tqdm_import
        print("Libraries loaded!")

    return sentence_transformers, cosine_similarity, tqdm


class SimpleCurriculumMatcher:
    def __init__(self):
        self.model = None

    def load_model(self):
        """Load the sentence transformer model with progress feedback"""
        if self.model is None:
            print(
                "Loading sentence transformer model (this may take 30-60 seconds on first run)..."
            )

            # Load heavy imports first
            SentenceTransformer, _, _ = load_heavy_imports()

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Model loaded successfully!")
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

    def create_concatenated_string(self, record, is_raw=True):
        """Create concatenated string from record"""
        if is_raw:
            # Raw input format
            parts = [
                self.normalize_text(record.get("product_type_usage", "")),
                self.normalize_text(record.get("subject", "")),
                self.normalize_text(record.get("product_name_raw", "")),
                self.normalize_text(record.get("publisher_raw", "")),
                self.normalize_text(record.get("copyright_year_raw", "")),
                self.normalize_text(record.get("grade", "")),
            ]
        else:
            # Catalog format
            parts = [
                self.normalize_text(record.get("product_type", "")),
                self.normalize_text(record.get("subject_level1", "")),
                self.normalize_text(record.get("product_name", "")),
                self.normalize_text(record.get("publisher", "")),
                self.normalize_text(record.get("copyright_year", "")),
                self.normalize_text(record.get("intended_grades", "")),
            ]

        # Join parts with spaces, remove empty parts
        concatenated = " ".join([part for part in parts if part])
        return concatenated

    def precompute_catalog_embeddings(self, catalog_df, progress_callback=None):
        """Precompute embeddings for all catalog products"""
        print("Creating concatenated strings for catalog...")

        # Create concatenated strings for all catalog records
        catalog_strings = []
        for idx, (_, row) in enumerate(catalog_df.iterrows()):
            concat_string = self.create_concatenated_string(row, is_raw=False)
            catalog_strings.append(concat_string)

            if progress_callback and idx % 100 == 0:
                progress_callback(idx + 1, len(catalog_df), "Preparing catalog")

        print("Computing embeddings for catalog...")
        # Batch encode all catalog strings at once
        catalog_embeddings = self.model.encode(
            catalog_strings,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        )

        return catalog_embeddings, catalog_strings

    def find_top_matches(
        self, raw_record, catalog_df, catalog_embeddings, catalog_strings
    ):
        """Find top 3 matches for a raw record"""
        # Create concatenated string for raw record
        raw_string = self.create_concatenated_string(raw_record, is_raw=True)

        if not raw_string.strip():
            return []

        try:
            # Encode the raw string
            raw_embedding = self.model.encode([raw_string], convert_to_numpy=True)

            # Calculate cosine similarity against all catalog embeddings
            similarities = cosine_similarity(raw_embedding, catalog_embeddings)[0]

            # Get top 3 matches
            top_indices = np.argsort(similarities)[-3:][::-1]  # Descending order

            results = []
            for idx in top_indices:
                results.append(
                    {
                        "product_identifier": catalog_df.iloc[idx].get(
                            "product_identifier", ""
                        ),
                        "score": float(similarities[idx]),
                        "catalog_string": catalog_strings[
                            idx
                        ],  # Add concatenated string
                    }
                )

            return results

        except Exception as e:
            print(f"Error processing record: {e}")
            return []

    def process_matching(self, raw_df, catalog_df, progress_callback=None):
        """Process all matching with progress tracking"""
        start_time = time.time()

        # Load model with user feedback
        if progress_callback:
            progress_callback(0, 100, "Loading AI model")

        self.load_model()

        if progress_callback:
            progress_callback(10, 100, "Model loaded")

        # Precompute catalog embeddings once
        catalog_embeddings, catalog_strings = self.precompute_catalog_embeddings(
            catalog_df, progress_callback
        )

        print(
            f"Catalog preprocessing completed in {time.time() - start_time:.2f} seconds"
        )
        print(
            f"Processing {len(raw_df)} raw records against {len(catalog_df)} catalog records..."
        )

        results = []
        total_rows = len(raw_df)

        # Process raw records
        for idx, (_, raw_record) in enumerate(raw_df.iterrows()):
            # Find top matches
            top_matches = self.find_top_matches(
                raw_record, catalog_df, catalog_embeddings, catalog_strings
            )

            # Prepare result row
            result_row = raw_record.to_dict()

            # Add match columns (ID, score, and catalog string for each match)
            for i in range(3):
                if i < len(top_matches):
                    result_row[f"match_{i+1}_id"] = top_matches[i]["product_identifier"]
                    result_row[f"match_{i+1}_score"] = round(top_matches[i]["score"], 4)
                    result_row[f"match_{i+1}_catalog_string"] = top_matches[i][
                        "catalog_string"
                    ]
                else:
                    result_row[f"match_{i+1}_id"] = ""
                    result_row[f"match_{i+1}_score"] = 0.0
                    result_row[f"match_{i+1}_catalog_string"] = ""

            results.append(result_row)

            # Update progress
            if progress_callback:
                progress_callback(idx + 1, total_rows, "Processing matches")

            # Print progress every 1000 records
            if (idx + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (total_rows - idx - 1) / rate
                print(
                    f"Processed {idx + 1}/{total_rows} records. "
                    f"Rate: {rate:.1f} rec/sec. "
                    f"ETA: {remaining/60:.1f} minutes"
                )

        total_time = time.time() - start_time
        print(
            f"Processing completed in {total_time:.2f} seconds ({total_time/60:.1f} minutes)"
        )
        print(f"Average rate: {len(raw_df)/total_time:.1f} records per second")

        return pd.DataFrame(results)


class SimpleMatcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Concatenated Curriculum Matcher")
        self.root.geometry("1200x800")  # Doubled the size

        self.matcher = SimpleCurriculumMatcher()
        self.raw_file = None
        self.catalog_file = None

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(
            main_frame, text="Simple Concatenated Matcher", font=("Arial", 14, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Description
        desc_label = ttk.Label(
            main_frame,
            text="Fast matching using single concatenated string comparison",
            font=("Arial", 10),
        )
        desc_label.grid(row=1, column=0, columnspan=2, pady=5)

        # File selection
        ttk.Label(main_frame, text="Select Raw Input File (CSV):").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.raw_file_label = ttk.Label(
            main_frame, text="No file selected", foreground="gray"
        )
        self.raw_file_label.grid(row=3, column=0, sticky=tk.W)
        ttk.Button(main_frame, text="Browse", command=self.select_raw_file).grid(
            row=3, column=1, padx=10
        )

        ttk.Label(main_frame, text="Select Product Catalog File (CSV):").grid(
            row=4, column=0, sticky=tk.W, pady=(20, 5)
        )
        self.catalog_file_label = ttk.Label(
            main_frame, text="No file selected", foreground="gray"
        )
        self.catalog_file_label.grid(row=5, column=0, sticky=tk.W)
        ttk.Button(main_frame, text="Browse", command=self.select_catalog_file).grid(
            row=5, column=1, padx=10
        )

        # Process button
        self.process_button = ttk.Button(
            main_frame, text="Process Matching", command=self.start_processing
        )
        self.process_button.grid(row=6, column=0, columnspan=2, pady=20)
        self.process_button.config(state="disabled")

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.grid(
            row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )

        # Status label
        self.status_label = ttk.Label(main_frame, text="Select both files to begin")
        self.status_label.grid(row=8, column=0, columnspan=2, pady=10)

        # Log text area - much larger
        self.log_text = tk.Text(main_frame, height=25, width=120, font=("Consolas", 10))
        self.log_text.grid(
            row=9, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S)
        )

        # Scrollbar for log
        scrollbar = ttk.Scrollbar(
            main_frame, orient="vertical", command=self.log_text.yview
        )
        scrollbar.grid(row=9, column=2, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # Make the main frame expandable
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(9, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Instructions
        instructions = """
        Instructions:
        1. This version concatenates all fields into a single string
        2. Uses semantic similarity only (no field-specific logic)  
        3. Much faster than multi-field comparison
        4. Good for testing and quick iteration
        
        Expected processing time for 25K records: 10-30 minutes
        Note: AI model loads when you click "Process Matching" (30-60 seconds)
        
        ========================================================================
        FAST STARTUP: Heavy libraries load only when processing starts
        ========================================================================
        """
        self.log_text.insert(tk.END, instructions)

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

    def update_progress(self, current, total, stage="Processing"):
        """Update progress bar"""
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.status_label.config(text=f"{stage}: {current}/{total}")
        self.root.update()

    def start_processing(self):
        """Start processing in a separate thread"""
        self.process_button.config(state="disabled")
        self.log_message("Starting concatenated matching...")

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
            required_raw_cols = ["product_type_usage", "subject", "product_name_raw"]
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

            # Show estimated time
            estimated_minutes = (
                (len(raw_df) * len(catalog_df)) / 1000000 * 2
            )  # Rough estimate
            self.log_message(
                f"Estimated processing time: {estimated_minutes:.1f} minutes"
            )

            # Process matching
            self.log_message("Starting concatenated matching process...")
            results_df = self.matcher.process_matching(
                raw_df, catalog_df, progress_callback=self.update_progress
            )

            # Save results
            self.log_message("Processing complete. Choose save location...")

            # Generate default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"SimpleCurriculumMatcher_{timestamp}.csv"

            output_file = filedialog.asksaveasfilename(
                title="Save Results As",
                initialfile=default_filename,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )

            if output_file:
                results_df.to_csv(output_file, index=False)
                self.log_message(f"Results saved to: {os.path.basename(output_file)}")

                # Show summary statistics
                total_matches = len(results_df)
                high_confidence = len(results_df[results_df["match_1_score"] > 0.8])
                medium_confidence = len(
                    results_df[
                        (results_df["match_1_score"] > 0.6)
                        & (results_df["match_1_score"] <= 0.8)
                    ]
                )
                low_confidence = len(results_df[results_df["match_1_score"] <= 0.6])

                summary = f"""
                Processing Summary:
                - Total records processed: {total_matches}
                - High confidence matches (>0.8): {high_confidence} ({high_confidence/total_matches*100:.1f}%)
                - Medium confidence matches (0.6-0.8): {medium_confidence} ({medium_confidence/total_matches*100:.1f}%)
                - Low confidence matches (≤0.6): {low_confidence} ({low_confidence/total_matches*100:.1f}%)
                """
                self.log_message(summary)

                messagebox.showinfo(
                    "Success",
                    f"Processing complete!\n{summary}\nResults saved to: {os.path.basename(output_file)}",
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
    app = SimpleMatcherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
