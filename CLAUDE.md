# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered curriculum matching system that matches raw curriculum data to standardized product catalogs using semantic similarity and machine learning. The system provides three different matcher implementations optimized for different use cases.

## Development Setup

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running Applications
```bash
# Run the enhanced matcher (recommended for production)
python enhanced_curriculum_matcher.py

# Run the simple matcher (fast processing)
python simple_curriculum_matcher.py

# Run the original matcher (detailed analysis)
python curriculum_matcher.py

# Run the publisher dictionary builder
python pub_dictionary.py
```

## Architecture

### Core Components

**Matcher Implementations:**
- `enhanced_curriculum_matcher.py` - Production system with dual modes (QA Scorer + Automated Matcher), weighted scoring (name: 50%, publisher: 30%, grade: 20%)
- `simple_curriculum_matcher.py` - Fast concatenated string matcher optimized for speed and high-volume processing
- `curriculum_matcher.py` - Original comprehensive matcher with detailed field-by-field scoring and explanations

**Supporting Tools:**
- `pub_dictionary.py` - Publisher standardization dictionary builder with Analysis and Direct Conversion modes

### Data Flow Pipeline
```
Input CSV → Data Loading → Model Loading → Processing → Results CSV
```

Each matcher follows this pattern:
1. **Input Loading**: Raw curriculum data + product catalog
2. **Preprocessing**: Text normalization, grade standardization, embedding computation  
3. **Matching**: Filtering → Scoring → Ranking
4. **Output**: Results with scores, metadata, and explanations

### Key Classes

**EnhancedCurriculumMatcher** - Production matcher with configurable weights and dual operational modes
**SimpleCurriculumMatcher** - Speed-optimized matcher using concatenated strings
**CurriculumMatcher** - Original matcher with comprehensive multi-component scoring
**DictBuilderGUI** - Publisher name standardization utility

## Data Schemas

### Input Data Requirements
- `product_name_raw` - Raw product name text
- `publisher_raw` - Raw publisher name
- `grade` - Target grade level (supports P/K/PK/numeric)
- `subject` - Subject area
- `product_type_usage` - Product type classification

### Catalog Data Requirements  
- `product_identifier` - Unique catalog ID
- `product_name` - Standardized product name
- `supplier_name`/`publisher` - Standardized publisher
- `intended_grades` - Grade range (e.g., "K-5")
- `subject_level1` - Subject classification

## Processing Notes

### Heavy Dependencies
All matchers use lazy loading for ML libraries (sentence-transformers, sklearn, torch) to improve startup time. Libraries are loaded only when processing begins.

### Grade Handling
Enhanced matcher supports flexible grade parsing:
- P/PK/TK → Grade 0
- K → Kindergarten  
- Numeric grades 1-12
- Grade ranges (e.g., "K-5", "6-8")

### Performance Optimization
- Simple matcher uses batch embedding computation for speed
- Enhanced matcher includes publisher dictionary integration
- All implementations support progress tracking and threading for GUI responsiveness

## Matching Modes

### Enhanced Matcher Modes
1. **QA Scorer Mode** - Validates existing human matches against ground truth
2. **Automated Matcher Mode** - Finds new matches for unmatched curriculum items

### Publisher Dictionary Modes
1. **Analysis Mode** - Analyzes historical data to build confidence-based mappings
2. **Direct Conversion Mode** - Simple CSV to JSON dictionary conversion