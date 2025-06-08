import pandas as pd
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
from datetime import datetime

# --- CONFIGURATION ---
# These are now defaults, but can be changed by the user in the UI.
OUTPUT_JSON_FILENAME = "publisher_dictionary.json"
OUTPUT_CLEAN_MAPPINGS_FILENAME = "clean_mappings.csv"
OUTPUT_REVIEW_NEEDED_FILENAME = "review_needed.csv"

# --- TUNING PARAMETERS (for Analysis Mode) ---
CONFIDENCE_THRESHOLD = 0.90
MIN_OCCURRENCES_FOR_AMBIGUITY_CHECK = 5


class DictBuilderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Publisher Dictionary Builder")
        self.root.geometry("700x650")

        self.input_file_path = tk.StringVar()
        self.output_dir_path = tk.StringVar()
        self.analysis_mode = tk.BooleanVar(
            value=False
        )  # Default to Direct Conversion mode

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Title ---
        title_label = ttk.Label(
            main_frame, text="Publisher Dictionary Builder", font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W)

        # --- Step 1: Input File Selection ---
        ttk.Label(main_frame, text="1. Select Input Mapping File (CSV)").grid(
            row=1, column=0, sticky=tk.W, pady=(10, 2)
        )
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        input_entry = ttk.Entry(
            input_frame, textvariable=self.input_file_path, width=70, state="readonly"
        )
        input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(input_frame, text="Browse...", command=self.select_input_file).pack(
            side=tk.LEFT
        )

        # --- Step 2: Output Directory Selection ---
        ttk.Label(main_frame, text="2. Select Output Directory").grid(
            row=3, column=0, sticky=tk.W, pady=(10, 2)
        )
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
        output_entry = ttk.Entry(
            output_frame, textvariable=self.output_dir_path, width=70, state="readonly"
        )
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(output_frame, text="Browse...", command=self.select_output_dir).pack(
            side=tk.LEFT
        )

        # --- Step 3: Mode Selection ---
        mode_check = ttk.Checkbutton(
            main_frame,
            text="Create new dictionary from historical data (enables analysis)",
            variable=self.analysis_mode,
        )
        mode_check.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=15)

        # --- Step 4: Action Button ---
        self.generate_button = ttk.Button(
            main_frame,
            text="Generate Dictionary",
            command=self.start_processing,
            state="disabled",
        )
        self.generate_button.grid(row=6, column=0, columnspan=2, pady=10)

        # --- Progress & Status ---
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.status_label = ttk.Label(
            main_frame, text="Please select an input file and output directory."
        )
        self.status_label.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(5, 10))

        # --- Log Area ---
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text = tk.Text(log_frame, height=15, width=80, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(9, weight=1)

    def select_input_file(self):
        path = filedialog.askopenfilename(
            title="Select Input Mapping File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.input_file_path.set(path)
            self.check_ready()

    def select_output_dir(self):
        path = filedialog.askdirectory(title="Select Directory to Save Output Files")
        if path:
            self.output_dir_path.set(path)
            self.check_ready()

    def check_ready(self):
        if self.input_file_path.get() and self.output_dir_path.get():
            self.generate_button.config(state="normal")
            self.status_label.config(text="Ready to generate dictionary.")

    def log_message(self, message):
        def update_gui():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)

        self.root.after(0, update_gui)

    def update_progress(self, value, text):
        def update_gui():
            self.progress_var.set(value)
            self.status_label.config(text=text)

        self.root.after(0, update_gui)

    def start_processing(self):
        self.generate_button.config(state="disabled")
        thread = threading.Thread(target=self.run_processing_logic)
        thread.daemon = True
        thread.start()

    def run_processing_logic(self):
        try:
            if self.analysis_mode.get():
                self.log_message("Starting process in Analysis Mode...")
                self._run_analysis_mode()
            else:
                self.log_message("Starting process in Direct Conversion Mode...")
                self._run_direct_conversion_mode()
        except Exception as e:
            self.log_message(f"ERROR: {e}")
            messagebox.showerror(
                "Error", f"An error occurred during processing:\n\n{e}"
            )
            self.update_progress(100, "Failed.")
        finally:
            self.root.after(0, lambda: self.generate_button.config(state="normal"))

    def _load_and_validate_data(self):
        input_file = self.input_file_path.get()
        raw_col, standard_col = "publisher_raw", "supplier_name"

        try:
            df = pd.read_csv(input_file, encoding="latin-1")
            if raw_col not in df.columns or standard_col not in df.columns:
                raise ValueError(
                    f"Input file must contain '{raw_col}' and '{standard_col}' columns."
                )
            self.log_message(f"Successfully loaded {len(df)} records.")
            return df[[raw_col, standard_col]]
        except Exception as e:
            raise Exception(f"Failed to load or validate input file: {e}")

    def _normalize_dataframe(self, df):
        df_norm = df.dropna().copy()
        df_norm.columns = ["raw_norm", "standard_norm"]
        suffixes = [r"\binc\b", r"\bllc\b", r"\bltd\b", r"\bcorp\b"]
        for col in ["raw_norm", "standard_norm"]:
            df_norm[col] = df_norm[col].str.lower().str.strip()
            for suffix in suffixes:
                df_norm[col] = (
                    df_norm[col].str.replace(suffix, "", regex=True).str.strip()
                )
        self.log_message("Normalization complete.")
        return df_norm

    def _run_direct_conversion_mode(self):
        self.update_progress(10, "Loading data...")
        df = self._load_and_validate_data()

        self.update_progress(30, "Normalizing mappings...")
        df_norm = self._normalize_dataframe(df)

        self.update_progress(60, "Converting to dictionary...")
        # Since the input is finalized, we drop duplicates on the raw name to ensure a valid 1-to-1 map
        final_mappings = df_norm.drop_duplicates(subset=["raw_norm"])
        self.log_message(
            f"Removed {len(df_norm) - len(final_mappings)} duplicate raw names to create a clean map."
        )

        publisher_dict = pd.Series(
            final_mappings.standard_norm.values, index=final_mappings.raw_norm
        ).to_dict()

        self.update_progress(90, "Saving output files...")
        output_dir = self.output_dir_path.get()

        json_path = os.path.join(output_dir, OUTPUT_JSON_FILENAME)
        with open(json_path, "w") as f:
            json.dump(publisher_dict, f, indent=4)
        self.log_message(f"Saved clean dictionary to: {json_path}")

        clean_csv_path = os.path.join(output_dir, OUTPUT_CLEAN_MAPPINGS_FILENAME)
        final_mappings.to_csv(clean_csv_path, index=False)
        self.log_message(f"Saved human-readable clean mappings to: {clean_csv_path}")

        self.update_progress(100, "Direct conversion complete!")
        self.log_message("\nProcess finished successfully.")
        messagebox.showinfo(
            "Success",
            f"Dictionary conversion complete!\n\nFiles saved in:\n{output_dir}",
        )

    def _run_analysis_mode(self):
        self.update_progress(5, "Loading data...")
        df = self._load_and_validate_data()

        self.update_progress(15, "Normalizing mappings...")
        df_norm = self._normalize_dataframe(df)

        self.update_progress(40, "Analyzing mapping frequencies...")
        freq_analysis = (
            df_norm.groupby(["raw_norm", "standard_norm"])
            .size()
            .reset_index(name="count")
        )

        self.update_progress(60, "Determining best mappings...")
        idx_of_max_count = freq_analysis.groupby("raw_norm")["count"].idxmax()
        winning_mappings = freq_analysis.loc[idx_of_max_count].copy()
        winning_mappings.rename(columns={"count": "winning_count"}, inplace=True)
        total_occurrences = (
            freq_analysis.groupby("raw_norm")["count"]
            .sum()
            .reset_index(name="total_count")
        )
        final_analysis = pd.merge(winning_mappings, total_occurrences, on="raw_norm")
        final_analysis["confidence"] = (
            final_analysis["winning_count"] / final_analysis["total_count"]
        )

        self.update_progress(80, "Filtering for high-confidence mappings...")
        is_clean = (final_analysis["confidence"] >= CONFIDENCE_THRESHOLD) | (
            final_analysis["total_count"] < MIN_OCCURRENCES_FOR_AMBIGUITY_CHECK
        )
        clean_mappings = final_analysis[is_clean]
        ambiguous_mappings = final_analysis[~is_clean]
        self.log_message(
            f"Found {len(clean_mappings)} clean and {len(ambiguous_mappings)} ambiguous mappings."
        )

        self.update_progress(90, "Saving output files...")
        output_dir = self.output_dir_path.get()
        publisher_dict = pd.Series(
            clean_mappings.standard_norm.values, index=clean_mappings.raw_norm
        ).to_dict()

        json_path = os.path.join(output_dir, OUTPUT_JSON_FILENAME)
        with open(json_path, "w") as f:
            json.dump(publisher_dict, f, indent=4)
        self.log_message(f"Saved clean dictionary to: {json_path}")

        clean_csv_path = os.path.join(output_dir, OUTPUT_CLEAN_MAPPINGS_FILENAME)
        clean_mappings.to_csv(clean_csv_path, index=False)
        self.log_message(f"Saved human-readable clean mappings to: {clean_csv_path}")

        if not ambiguous_mappings.empty:
            review_csv_path = os.path.join(output_dir, OUTPUT_REVIEW_NEEDED_FILENAME)
            ambiguous_mappings.to_csv(review_csv_path, index=False)
            self.log_message(
                f"WARNING: Saved ambiguous mappings for review to: {review_csv_path}"
            )

        self.update_progress(100, "Analysis complete!")
        self.log_message("\nProcess finished successfully.")
        messagebox.showinfo(
            "Success", f"Dictionary analysis complete!\n\nFiles saved in:\n{output_dir}"
        )


def main():
    root = tk.Tk()
    app = DictBuilderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
