# Enhanced Curriculum Matcher

The Enhanced Curriculum Matcher is a powerful Python application designed to accurately match messy, real-world curriculum data against a standardized product catalog. It uses a sophisticated, multi-faceted scoring engine and provides a user-friendly graphical interface (GUI) for iterative testing, as well as a headless mode for large-scale batch processing.

This tool is ideal for data quality assurance (QA) and for automating the process of matching new, un-matched curriculum data.

## Key Features

-   **Multi-Faceted Scoring Engine:** Moves beyond simple string comparison by calculating independent scores for Product Name, Publisher, and Grade, then combines them using tunable weights for a more reliable `final_score`.
-   **Dual-Mode Operation:**
    -   **QA Scorer Mode:** Validates previously made human matches by calculating a confidence score, helping to identify potential errors in an existing dataset.
    -   **Automated Matcher Mode:** Takes new, unmatched data and finds the Top 5 best potential matches from the catalog, complete with detailed scoring for efficient human review.
-   **Robust Normalization:** Employs advanced, purpose-built data cleaning pipelines to handle real-world data issues like typos, punctuation, abbreviations, and word order variations.
-   **Context-Aware Matching:**
    -   **Publisher Acquisitions:** Intelligently checks against both current (`publisher`) and prior (`publisher_prior`) publisher fields to correctly handle acquisitions.
    -   **Title Enrichment:** Enriches the catalog's product names with `series` and `subject_level2` data to create a more powerful search signal.
-   **Graphical User Interface (GUI):** A user-friendly desktop application for easy file selection and processing on a local machine.
-   **Headless Mode:** A command-line option (`--headless`) to bypass the GUI, enabling large-scale processing on servers or in cloud environments like Google Colab where a GPU can provide a significant speed-up.
-   **Enriched QA Output:** The output CSV includes all original data columns plus detailed component scores and diagnostic "helper" columns to make the QA and refinement process fast and intuitive.

## Setup and Installation

This project uses a virtual environment to manage dependencies.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd enhanced-curriculum-matcher
    ```

2.  **Create a Virtual Environment:**
    ```bash
    # For Windows
    python -m venv venv

    # For macOS/Linux
    python3 -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    ```bash
    # For Windows (PowerShell)
    .\venv\Scripts\Activate

    # For macOS/Linux
    source venv/bin/activate
    ```
    Your terminal prompt should now show `(venv)` at the beginning.

4.  **Install Dependencies:**
    Install all required libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application can be run in two ways: with the GUI for local testing, or in headless mode for large-scale processing.

### GUI Mode (For Laptops)

This is the standard way to run the application for smaller test batches.

1.  Activate your virtual environment (see Step 3 above).
2.  Run the script from your terminal:
    ```bash
    python enhanced_curriculum_matcher.py
    ```
3.  The application window will appear.
    -   **Select Operating Mode:** Choose either "QA Scorer" or "Automated Matcher".
    -   **Select Files:** Use the "Browse..." buttons to select your **Input Data File (CSV)** and your **Product Catalog File (CSV)**.
    -   **Select Output Directory:** Choose the folder where you want the results file to be saved.
    -   **Start Processing:** Click the "Start Processing" button. Progress will be displayed in the progress bar and the log window.

### Headless Mode (For Servers or Google Colab)

This mode is designed for processing very large datasets (like your 520k records) where a GPU is beneficial and a GUI is not available.

1.  **Open the Script:** Open `enhanced_curriculum_matcher.py` in a text editor.
2.  **Configure File Paths:** Navigate to the `_run_headless_mode` function and **edit the file paths** to point to your data files in the server/Colab environment.
    ```python
    def _run_headless_mode(self):
        # --- CONFIGURE FILE PATHS FOR COLAB/HEADLESS MODE HERE ---
        input_file_path = "/content/your_large_input_file.csv"
        catalog_file_path = "/content/product_catalog.csv"
        output_dir_path = "/content/results/"
        mode = "Matcher" # or "QA"
        # ...
    ```
3.  **Run from the Command Line:** Activate your environment and run the script with the `--headless` flag.
    ```bash
    python enhanced_curriculum_matcher.py --headless
    ```
4.  The script will run without launching a UI, printing all log messages directly to the console. The output file will be saved to the specified output directory.

## Data Requirements

For the program to function correctly, your input files should contain the following key columns:

-   **Input Data File:** `product_name_raw`, `publisher_raw`, `grade`, `subject`, `product_type_usage`. For QA mode, it must also contain `product_identifier`.
-   **Product Catalog File:** `product_identifier`, `product_name`, `publisher`, `publisher_prior`, `series`, `subject_level1`, `subject_level2`, `intended_grades`.