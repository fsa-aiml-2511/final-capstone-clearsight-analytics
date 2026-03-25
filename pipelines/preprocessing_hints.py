"""
Healthcare Data Preprocessing Hints
=====================================
These are HINTS, not complete solutions. Use them as a starting point
for your data pipeline. You'll need to adapt and expand these for your
specific models.

Datasets:
- patient_encounters_2023.csv (101,767 encounters)
- clinical_codes_reference.csv (68 reference codes)
- patient_medication_feedback.csv (~180,000 drug reviews)
- retinal_labels.csv (3,662 image labels)
- retinal_scan_images/ (3,222 PNG images)
"""

# =============================================================================
# *** CLASS IMBALANCE WARNING ***
# =============================================================================
# EVERY dataset in this project has class imbalance. If you ignore it,
# your model will learn to predict the majority class and look "accurate"
# while being clinically useless.
#
# Techniques you MUST consider for every model:
#   1. class_weight='balanced' in sklearn models (easiest first step)
#   2. SMOTE (Synthetic Minority Oversampling) from imblearn
#   3. Stratified train/test splits (use stratify= in train_test_split)
#   4. Weighted loss functions in TensorFlow/Keras
#   5. Evaluation with weighted F1, precision, recall — NOT just accuracy
#
# A model that predicts the majority class for everything is WORTHLESS
# even if it gets 80%+ accuracy. Always check per-class metrics.
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# HINT 1: Loading Data with Proper Handling
# =============================================================================

def load_encounters(filepath: str) -> pd.DataFrame:
    """
    Load the patient encounters dataset.

    Key gotchas:
    - Missing values are encoded as '?' (string), not NaN
    - You need to convert '?' to NaN after loading
    - Some columns look numeric but contain '?' so they load as strings
    """
    df = pd.read_csv(filepath)

    # Replace '?' with NaN across the entire dataframe
    df.replace('?', np.nan, inplace=True)

    # These columns should be numeric but may have loaded as strings
    numeric_cols = ['weight', 'payer_code', 'medical_specialty']
    # Be careful: not all of these are truly numeric (medical_specialty is categorical)

    return df


def load_medication_reviews(filepath: str) -> pd.DataFrame:
    """
    Load patient medication feedback (drug review) data.

    Columns: patient_id, drug_name, condition, review_text, effectiveness, rating, date

    Key gotchas:
    - ~180K reviews across 3,400+ medications and 800+ conditions
    - Review text contains HTML artifacts, informal language, medical jargon
    - Patient IDs do NOT match encounter data (different patient systems)
    - 'effectiveness' is the NLP classification target (5 classes)
    - 'rating' is numeric 1-10 (corresponds to effectiveness category)
    - Class distribution may be imbalanced — check before training
    """
    df = pd.read_csv(filepath)
    return df


# =============================================================================
# HINT 2: Understanding What's Missing and Why
# =============================================================================

def analyze_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing data in healthcare is rarely random.

    In this dataset:
    - weight (97% missing): Only recorded for specific admission types
    - medical_specialty (49% missing): Not always documented in ER admissions
    - payer_code (40% missing): Insurance info sometimes unavailable
    - max_glu_serum (95% missing): Only ordered when clinically indicated
    - A1Cresult (83% missing): Only ordered when clinically indicated

    The MISSINGNESS ITSELF can be a feature. A missing glucose test might mean
    the clinician didn't think it was necessary — that's clinical information.
    """
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_pct': (df.isnull().sum().values / len(df) * 100).round(1)
    })
    return missing.sort_values('missing_pct', ascending=False)


# =============================================================================
# HINT 3: Age Encoding
# =============================================================================

def parse_age_brackets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Age is stored as text brackets: "[70-80)", "[60-70)", etc.

    Options:
    1. Convert to midpoint (e.g., "[70-80)" -> 75)
    2. Convert to ordinal encoding
    3. Keep as categorical with one-hot encoding

    For clinical interpretability, midpoint is often best.
    """
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25,
        '[30-40)': 35, '[40-50)': 45, '[50-60)': 55,
        '[60-70)': 65, '[70-80)': 75, '[80-90)': 85,
        '[90-100)': 95
    }
    df['age_numeric'] = df['age'].map(age_map)
    return df


# =============================================================================
# HINT 4: ICD-9 Diagnostic Codes
# =============================================================================

def categorize_icd9(code) -> str:
    """
    ICD-9 codes are hierarchical. The first 3 digits tell you the category.

    Common diabetes-related codes:
    - 250.xx: Diabetes mellitus
    - 276.xx: Disorders of fluid, electrolyte, and acid-base balance
    - 428.xx: Heart failure
    - 401.xx: Essential hypertension
    - 414.xx: Chronic ischemic heart disease

    Grouping codes into clinical categories is powerful feature engineering.
    """
    if pd.isna(code):
        return 'missing'

    code = str(code)

    # Diabetes codes
    if code.startswith('250'):
        return 'diabetes'
    # Circulatory
    elif code[0] in ['3', '4'] and 390 <= float(code[:3]) <= 459:
        return 'circulatory'
    # Respiratory
    elif 460 <= float(code[:3]) <= 519:
        return 'respiratory'
    # Digestive
    elif 520 <= float(code[:3]) <= 579:
        return 'digestive'
    # Injury
    elif code.startswith('V') or code.startswith('E'):
        return 'external'
    else:
        return 'other'


# =============================================================================
# HINT 5: Creating the Readmission Target — BINARY
# =============================================================================
# IMPORTANT: The raw `readmitted` column has 3 values: '<30', '>30', 'NO'
# You need to convert this to a BINARY target for your classification models.
#
# Binary mapping:
#   - readmitted = 1  if '<30' OR '>30'  (patient WAS readmitted, any timeframe)
#   - readmitted = 0  if 'NO'            (patient was NOT readmitted)
#
# This gives you approximately 46% positive (readmitted) / 54% negative (not
# readmitted) — a much more balanced split than using only <30 days.
#
# The clinical question is: "Will this patient come back to the hospital?"
# Both <30 and >30 day readmissions represent a failure in discharge planning.
# =============================================================================

def create_binary_readmission(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary readmission target from the 3-value readmitted column.

    Raw values: '<30', '>30', 'NO'
    Binary target: 1 = readmitted (both <30 AND >30), 0 = not readmitted (NO)

    This gives ~46% positive / ~54% negative — a much more balanced split
    than using only <30 day readmissions (~11%).

    For healthcare: prioritize SENSITIVITY (catching readmissions) over
    specificity. Missing a readmission is worse than a false alarm.

    Strategies to handle remaining imbalance:
    1. class_weight='balanced' in sklearn models
    2. SMOTE (oversampling minority class)
    3. Stratified train/test splits
    4. Threshold tuning on probability output
    """
    df['readmission_binary'] = (df['readmitted'] != 'NO').astype(int)

    pos = df['readmission_binary'].sum()
    neg = len(df) - pos
    print(f"Readmission binary target created:")
    print(f"  Readmitted (1): {pos} ({pos/len(df)*100:.1f}%)")
    print(f"  Not readmitted (0): {neg} ({neg/len(df)*100:.1f}%)")

    return df


# =============================================================================
# HINT 5B: Exclude Death/Hospice Discharges (CRITICAL)
# =============================================================================

def exclude_death_hospice(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL: Exclude patients who died or were discharged to hospice.
    These patients cannot be "readmitted" — including them creates data leakage
    because their outcome is guaranteed to be "no readmission" for reasons
    unrelated to clinical care quality.

    Discharge disposition IDs to exclude:
      11 = Expired
      13 = Hospice / home
      14 = Hospice / medical facility
      19 = Expired at home (Medicaid only, hospice)
      20 = Expired in a medical facility (Medicaid only, hospice)
      21 = Expired, place unknown (Medicaid only, hospice)

    Run this BEFORE creating your readmission target variable.
    """
    exclude_dispositions = [11, 13, 14, 19, 20, 21]
    n_before = len(df)
    df = df[~df['discharge_disposition_id'].isin(exclude_dispositions)]
    n_after = len(df)
    print(f"Excluded {n_before - n_after} death/hospice rows ({n_before} -> {n_after})")
    return df


# =============================================================================
# HINT 6: Medication Feature Engineering
# =============================================================================

def process_medications(df: pd.DataFrame) -> pd.DataFrame:
    """
    23 medication columns with values: "No", "Steady", "Up", "Down"

    Feature ideas:
    1. Total medications changed (count of non-"No" values)
    2. Any medication changed (binary)
    3. Medications increased vs decreased
    4. Specific high-impact medications (metformin, insulin)
    """
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                'metformin-pioglitazone']

    # Only use columns that exist in the dataframe
    med_cols = [c for c in med_cols if c in df.columns]

    # Count medications that were changed
    df['n_meds_changed'] = df[med_cols].apply(
        lambda row: sum(1 for v in row if v != 'No'), axis=1
    )

    # Any medication changed
    df['any_med_changed'] = (df['n_meds_changed'] > 0).astype(int)

    # Medications increased
    df['n_meds_increased'] = df[med_cols].apply(
        lambda row: sum(1 for v in row if v == 'Up'), axis=1
    )

    return df


# =============================================================================
# HINT 7: Clinical Complexity Score
# =============================================================================

def create_complexity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple features into a clinical complexity score.

    Higher complexity = higher readmission risk (clinically intuitive).
    This is an example of domain-driven feature engineering.
    """
    df['complexity_score'] = (
        df['num_lab_procedures'].fillna(0) / 50 +  # Normalize
        df['num_procedures'].fillna(0) / 6 +
        df['num_medications'].fillna(0) / 20 +
        df['number_diagnoses'].fillna(0) / 9 +
        df['number_emergency'].fillna(0) / 3 +
        df['number_inpatient'].fillna(0) / 5
    )
    return df


# =============================================================================
# HINT 8: Retinal Image — BINARY Classification
# =============================================================================
# IMPORTANT: Convert the 5-class diagnosis (0-4) to BINARY for your model:
#   - 0 = No DR        (class 0 only)     — ~49% of data
#   - 1 = Has DR        (classes 1-4)      — ~51% of data
#
# This gives you an approximately 50/50 split — well balanced!
# The clinical question is: "Does this patient have ANY diabetic retinopathy?"
# This is the most useful screening question — patients with DR (any grade)
# need specialist referral, while patients without DR can wait.
# =============================================================================

def create_binary_retinal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 5-class retinal diagnosis to binary: No DR vs Has DR.

    Raw values: 0 (No DR), 1 (Mild), 2 (Moderate), 3 (Severe), 4 (Proliferative)
    Binary target: 0 = No DR (class 0), 1 = Has DR (classes 1-4 combined)

    This gives approximately 50/50 split — well balanced for training.

    The clinical use case: screen patients to identify who needs a specialist
    referral (any DR) vs. who can wait (no DR).
    """
    df['dr_binary'] = (df['diagnosis'] > 0).astype(int)

    pos = df['dr_binary'].sum()
    neg = len(df) - pos
    print(f"Binary retinal target created:")
    print(f"  Has DR (1): {pos} ({pos/len(df)*100:.1f}%)")
    print(f"  No DR (0): {neg} ({neg/len(df)*100:.1f}%)")

    return df


def get_retinal_image_info():
    """
    Tips for working with the retinal images:

    1. Images are high-resolution PNGs — resize for training
       - Common sizes: 224x224 (ResNet), 299x299 (Inception), 384x384 (ViT)

    2. Class distribution is imbalanced:
       - Grade 0: 49% (No DR)
       - Grade 1: 10% (Mild)
       - Grade 2: 27% (Moderate)
       - Grade 3: 5%  (Severe)       <-- Very underrepresented
       - Grade 4: 8%  (Proliferative)

    3. Augmentation is critical for the minority classes:
       - Random rotation, flipping, color jitter
       - Consider: RandomResizedCrop, ColorJitter, RandomHorizontalFlip

    4. Transfer learning recommended:
       - ResNet50, EfficientNet, or DenseNet pre-trained on ImageNet
       - Fine-tune the last few layers on your retinal data
       - Freeze early layers (they detect generic features like edges)

    5. For clinical use, "referable" DR (grades 2-4) is the key metric
       - Sensitivity for referable cases is more important than overall accuracy
       - A missed severe case is much worse than a false alarm

    6. Match images to labels:
       - retinal_labels.csv has id_code and diagnosis columns
       - Image filenames in retinal_scan_images/ correspond to id_code
       - Some labels may not have matching images — filter these out
       - See match_retinal_images_to_labels() below for working join code
    """
    pass


def match_retinal_images_to_labels(labels_path: str, image_dir: str) -> pd.DataFrame:
    """
    Join retinal labels to actual image files.

    The labels file has 3,662 entries but only 3,222 images exist.
    ~440 labels have no matching image and must be filtered out.
    Always do an inner join on what's actually on disk.
    """
    import os

    labels = pd.read_csv(labels_path)
    existing_images = set(os.listdir(image_dir))

    # Keep only labels where the corresponding image file exists
    labels_matched = labels[labels['id_code'].apply(lambda x: x + '.png' in existing_images)]

    print(f"Labels in CSV: {len(labels)}")
    print(f"Images on disk: {len(existing_images)}")
    print(f"Matched (usable): {len(labels_matched)}")
    print(f"Labels without images (dropped): {len(labels) - len(labels_matched)}")

    return labels_matched


# =============================================================================
# HINT 9: Medication Effectiveness — 3-Category Mapping
# =============================================================================
# IMPORTANT: The raw effectiveness column has 5 categories, but you should
# map them to 3 categories for your NLP classification model:
#
#   "Highly Effective"    -> "Highly Effective"
#   "Considerably Effective" -> "Somewhat Effective"
#   "Moderately Effective"   -> "Somewhat Effective"
#   "Marginally Effective"   -> "Somewhat Effective"
#   "Ineffective"         -> "Ineffective"
#
# This simplifies the problem while preserving clinically meaningful
# distinctions: the drug clearly works, sort of works, or doesn't work.
# The 3 middle categories are hard to distinguish even for clinicians.
# =============================================================================

def create_medication_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map 5 effectiveness categories to 3 for NLP classification.

    Raw values: Highly Effective, Considerably Effective, Moderately Effective,
                Marginally Effective, Ineffective
    Mapped to:  Highly Effective, Somewhat Effective, Ineffective

    The 3 middle categories (Considerably, Moderately, Marginally) are
    collapsed into "Somewhat Effective" because:
    1. The distinctions are subjective and hard to classify from text
    2. Clinically, the actionable decision is the same: monitor and adjust
    3. This gives better-balanced classes for training
    """
    effectiveness_map = {
        'Highly Effective': 'Highly Effective',
        'Considerably Effective': 'Somewhat Effective',
        'Moderately Effective': 'Somewhat Effective',
        'Marginally Effective': 'Somewhat Effective',
        'Ineffective': 'Ineffective'
    }

    df['effectiveness_3class'] = df['effectiveness'].map(effectiveness_map)

    print("3-class effectiveness distribution:")
    print(df['effectiveness_3class'].value_counts())
    print(f"\nClass proportions:")
    print(df['effectiveness_3class'].value_counts(normalize=True).round(3))

    return df


# =============================================================================
# HINT 10: NLP Preprocessing for Drug Reviews
# =============================================================================

def preprocess_review_text(text: str) -> str:
    """
    Patient drug review text is messy. Real-world NLP!

    Dataset: ~180K drug reviews with columns:
    - review_text: the patient's written review (INPUT for your model)
    - effectiveness: 5-class target ("Highly Effective", "Considerably Effective",
      "Moderately Effective", "Marginally Effective", "Ineffective")
    - drug_name, condition: useful for analysis but careful about leakage
    - rating: numeric 1-10 (DO NOT use as a feature — it directly maps to the target!)

    Common issues in this dataset:
    - Typos and informal language: patients write like they're talking
    - Medical abbreviations: "mg", "BP", "HbA1c", drug brand names
    - Emotional language: "this drug saved my life" vs "worst experience ever"
    - Mixed sentiment: "works great but the side effects are horrible"
    - Some reviews are very short, others are multi-paragraph narratives
    - HTML artifacts may remain (&#039; for apostrophes, etc.)

    For classification, consider:
    1. TF-IDF + traditional classifier (good baseline — start here!)
    2. Word embeddings (Word2Vec, GloVe) + LSTM
    3. Pre-trained transformers (BERT, DistilBERT)
       - Bio-BERT or Clinical BERT for medical text
       - These understand medical terminology better

    Data leakage warning:
    - DO NOT use 'rating' as a feature (it maps directly to 'effectiveness')
    - Be careful with 'drug_name' — some drugs only treat conditions with
      high/low effectiveness, so the name leaks information about the target

    Simplification options:
    - Full 5-class classification (most challenging)
    - 3-class: Effective (7-10), Neutral (4-6), Ineffective (1-3)
    - Binary: Effective (rating >= 7) vs Ineffective (rating < 7)
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    # Add your preprocessing steps here
    return text


# =============================================================================
# HINT 11: Innovation Model Ideas
# =============================================================================

def innovation_model_hints():
    """
    Your Innovation Model (Model 5) — Your Team's Choice

    This is your chance to surprise us. Identify a problem in the data
    that we DIDN'T ask you to solve, and build a model for it.

    Ideas from this dataset:
    1. Length of Stay Prediction
       - Predict time_in_hospital from admission features
       - Regression or binned classification
       - Clinical value: helps bed management and discharge planning

    2. ICU Transfer Prediction
       - Use discharge_disposition to identify ICU transfers
       - Early warning system for deteriorating patients
       - High clinical impact

    3. Diabetes Subtype Clustering
       - Unsupervised learning on medication + diagnosis patterns
       - Discover patient phenotypes (e.g., insulin-dependent vs. oral-only)
       - Useful for personalized treatment planning

    4. Medication Interaction Risk Scoring
       - Use the 23 medication columns to predict adverse outcomes
       - Flag dangerous combinations
       - Combine with readmission data for validation

    5. Emergency vs. Elective Outcome Comparison
       - Compare readmission risk by admission type
       - Identify which emergency patients need extra follow-up

    Requirements:
    - Clear clinical justification for why this model matters
    - Defined success metric (you choose what to measure)
    - ROI estimate (how would this save money or improve care?)
    - Output must match model5_results_template.csv format

    Output columns: id, prediction, confidence, metric_name, metric_value
    """
    pass
