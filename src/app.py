# src/app.py

import streamlit as st
from PIL import Image
import json
from pathlib import Path
import pandas as pd
import numpy as np

# --- Configuration ---
# We assume the app is run from the `src` directory, so paths are relative to it.
DATA_DIR = Path("../data")
MASTER_LOG_PATH = DATA_DIR / "master_traceability_log.json"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="VisionXplain",
    page_icon="👁️",
    layout="wide"
)

# --- Helper Function to load data ---
# Use st.cache_data to load the data once and cache it for performance.
@st.cache_data
def load_master_log():
    """Loads the master traceability log file."""
    try:
        with open(MASTER_LOG_PATH, 'r') as f:
            log_data = json.load(f)
        # Create a mapping from image_id to its record for quick access
        records = {record['image_id']: record for record in log_data['records']}
        return records
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# --- Main App UI ---
st.title("👁️ VisionXplain: Explainable Image Descriptions")

# 1. Load the data
records = load_master_log()

if records is None:
    st.error(f"**Error:** The master log file was not found at `{MASTER_LOG_PATH}`. Please ensure you have run all the previous scripts (`data_ingestion.py`, `cv_module.py`, `nlp_module.py`, `traceability_module.py`) successfully.")
else:
    # --- Feature 7: Evaluation Dashboard ---
    st.header("Project Evaluation Dashboard")

    # Convert records to a Pandas DataFrame for easy analysis
    df = pd.DataFrame.from_dict(records, orient='index')

    # --- Metric Calculations ---
    
    # 1. Traceability Completeness
    # Check if the essential CV and NLP outputs exist for each record that was processed
    processed_df = df.dropna(subset=['cv_module_output', 'nlp_module_output'])
    total_processed = len(processed_df)
    traceability_complete_count = processed_df[
        processed_df['cv_module_output'].apply(lambda x: isinstance(x, list) and len(x) > 0) &
        processed_df['nlp_module_output'].apply(lambda x: isinstance(x, dict) and x.get('generated_description'))
    ].shape[0]
    traceability_completeness = (traceability_complete_count / total_processed * 100) if total_processed > 0 else 0

    # 2. Average CV Confidence (for top prediction)
    confidences = processed_df['cv_module_output'].apply(lambda x: x[0]['confidence'] if x else None).dropna()
    avg_confidence = confidences.mean() if not confidences.empty else 0

    # 3. Explainability Coverage
    # Check how many of the processed images have a generated explanation map
    explanation_count = processed_df['explanation_output'].apply(lambda x: x is not None and x.get('explanation_file_path') is not None).sum()
    explainability_coverage = (explanation_count / total_processed * 100) if total_processed > 0 else 0

    # --- Display Metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Traceability Completeness",
            value=f"{traceability_completeness:.1f}%",
            help="Percentage of processed records with complete CV and NLP logs."
        )
    with col2:
        st.metric(
            label="Avg. CV Confidence (Top 1)",
            value=f"{avg_confidence:.2%}",
            help="The average confidence score of the top object detection across all processed images."
        )
    with col3:
        st.metric(
            label="Explainability Coverage",
            value=f"{explanation_count} / {total_processed} Images",
            help="Number of processed images for which an explanation has been generated."
        )

    st.markdown("---")
    
    # --- Individual Image Explorer ---
    st.header("Individual Image Explorer")
    st.markdown("Select an image ID from the dropdown to see the model's output, its visual explanation, and the full traceability log.")

    # Create the image selection dropdown
    available_ids = [img_id for img_id, rec in records.items() if rec.get('source_info', {}).get('local_path')]
    
    def format_image_id(img_id):
        """Safely formats the dropdown label, handling missing CV data."""
        record = records.get(img_id, {})
        cv_output = record.get('cv_module_output')
        if isinstance(cv_output, list) and cv_output:
            top_detection = cv_output[0].get('object', 'N/A')
        else:
            top_detection = 'Not Processed'
        return f"{img_id} - (Top detection: {top_detection})"

    selected_id = st.selectbox(
        "**Select an Image ID to Analyze:**",
        options=available_ids,
        format_func=format_image_id
    )

    if selected_id:
        record = records[selected_id]
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            image_path_str = record.get('source_info', {}).get('local_path')
            if image_path_str and Path(image_path_str).exists():
                st.image(Image.open(image_path_str), use_column_width=True)
            else:
                st.warning("Original image file not found.")

            st.subheader("Visual Explanation (Saliency Map)")
            explanation_path_str = record.get('explanation_output', {}).get('explanation_file_path')
            if explanation_path_str and Path(explanation_path_str).exists():
                st.image(Image.open(explanation_path_str), caption="Highlighted regions influenced the CV model's top prediction.", use_column_width=True)
            else:
                st.info("Explanation map not generated for this image.")

        with col2:
            st.subheader("Generated Description")
            st.info(record.get('nlp_module_output', {}).get('generated_description', 'Not Processed'))

            st.subheader("Traceability Log")
            st.json(record, expanded=False)

