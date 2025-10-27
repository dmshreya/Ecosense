import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from rapidfuzz import process, fuzz
import os
import joblib

# --- 1. Page Configuration ---
# Set config at the very top
st.set_page_config(page_title="EcoSence | Predictor", page_icon="🌿", layout="centered")

# --- 2. Custom CSS ---
st.markdown(
    """
<style>
    body { 
        background: linear-gradient(135deg, #f8fbff 0%, #e6f0ff 100%);
        font-family: 'Segoe UI', sans-serif; 
    }
    .header { 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        background: linear-gradient(135deg, #004080 0%, #0078d7 100%); 
        padding: 20px 40px; 
        border-radius: 16px; 
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 30px;
    }
    .header h1 { 
        color: white; 
        margin: 0; 
        font-size: 28px;
        font-weight: bold;
    }
    .logo { 
        width: 55px; 
        height: 55px; 
        border-radius: 50%; 
    }
    .main-box { 
        background-color: white; 
        border-radius: 20px; 
        padding: 40px; 
        margin-top: 20px; 
        box-shadow: 0px 6px 16px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .input-label { 
        font-size: 20px !important; 
        font-weight: bold !important; 
        color: #004080 !important; 
        margin-bottom: 10px;
    }
    .stTextInput input {
        font-size: 18px !important;
        padding: 15px !important;
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
    }
    .stButton button {
        font-size: 18px !important;
        font-weight: bold;
        border-radius: 12px !important; 
        padding: 12px 30px !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    /* Style for individual correction buttons (selecting word) */
    .stButton .correction-button {
        background-color: #e6f0ff;
        color: #004080;
        border: 2px solid #0078d7;
        font-size: 16px !important;
        font-weight: normal !important;
        padding: 8px 12px !important;
    }
    .stButton .correction-button:hover {
        background-color: #0078d7;
        color: white;
    }
    /* Style for 'Yes'/'Go Back' buttons */
    .stButton .confirm-button-yes {
        background-color: #e8fbe8;
        color: #008000;
        border: 2px solid #00cc44;
        font-size: 16px !important;
        font-weight: bold !important;
        padding: 8px 12px !important;
    }
    .stButton .confirm-button-yes:hover {
        background-color: #00cc44;
        color: white;
    }
    .stButton .confirm-button-back {
        background-color: #fff;
        color: #555;
        border: 2px solid #ccc;
        font-size: 16px !important;
        font-weight: normal !important;
        padding: 8px 12px !important;
    }
    .stButton .confirm-button-back:hover {
        background-color: #f0f0f0;
        border-color: #999;
    }

    /* Style for cancel/keep buttons */
    .stButton .control-button-keep {
        background-color: #e0e0e0;
        color: #333;
        border: 2px solid #999;
        font-size: 16px !important;
        font-weight: normal !important;
    }
    .stButton .control-button-keep:hover {
        background-color: #ccc;
    }
    .stButton .control-button-cancel {
        background-color: #ffe6e6;
        color: #cc0000;
        border: 2px solid #ff3333;
        font-size: 16px !important;
        font-weight: normal !important;
    }
    .stButton .control-button-cancel:hover {
        background-color: #ff3333;
        color: white;
    }
    
    .suggestions-box {
        background-color: #f8fbff;
        border-radius: 12px;
        padding: 15px 20px;
        margin: 15px 0;
        border-left: 5px solid #0078d7;
    }
    .success-box { 
        background-color: #e8fbe8; 
        border-left: 6px solid #00cc44; 
        padding: 25px; 
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
    }
    .error-box { 
        background-color: #ffe6e6; 
        border-left: 6px solid #ff3333; 
        padding: 25px; 
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Make sure spellchecker is installed: pip install pyspellchecker ---
try:
    from spellchecker import SpellChecker

    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False

    # Define a dummy SpellChecker class if the library is not installed
    class SpellChecker:
        def __init__(self, distance=2):
            pass

        def correction(self, word):
            return word  # Just return the original word

        def word_frequency(self):  # Dummy attribute
            pass

        def update(self, words):  # Dummy method
            pass


# # --- 1. Page Configuration ---
# st.set_page_config(
#     page_title="Sustainability Predictor", page_icon="♻️", layout="centered"
# )


# --- 2. Load Assets (Simplified & Corrected Filename) ---
@st.cache_resource
def load_assets():
    # --- Use the exact filename provided by the user ---
    df_path = "Sustainable.csv"
    vocab_path = "app_vocab.json"  # Assumes you saved this from your notebook

    df = None
    vocab = None
    spell = None
    model = None

    # Load DataFrame
    try:
        df = pd.read_csv(df_path)
        print(f"Successfully loaded data from: {df_path}")
    except FileNotFoundError:
        st.error(
            f"Error: Data file not found at '{df_path}'. Make sure it's in the same folder as app.py."
        )
        return None, None, None  # Stop if data fails
    except Exception as e:
        st.error(f"Error reading CSV file '{df_path}': {e}")
        return None, None, None  # Stop if data fails

    # Load Vocabulary
    try:
        with open(vocab_path, "r") as f:
            vocab = set(json.load(f))
        st.success(f"Successfully loaded vocabulary from: {vocab_path}")
    except FileNotFoundError:
        print(
            "Warning: Vocab file not found at '{vocab_path}'. Spell check might be limited. Building vocab from data..."
        )
        # Fallback: Build vocab from DataFrame
        vocab = set()
        text_columns = [
            "Name",
            "Components Used",
            "Packaging",
        ]  # Columns to build vocab from
        for col in text_columns:
            if col in df.columns:
                for cell in df[col].fillna(""):
                    for t in re.findall(r"[a-zA-Z]+", str(cell).lower()):
                        if len(t) >= 2:
                            vocab.add(t)
        if not vocab:  # Add defaults if still empty
            vocab = {"product", "sustainable", "plastic", "bottle", "packaging"}
    except Exception as e:
        st.error(f"Error reading JSON file '{vocab_path}': {e}")
        # Continue without vocab if it fails to load

    # Initialize SpellChecker (only if library is available)
    if SPELLCHECKER_AVAILABLE:
        try:
            spell = SpellChecker(distance=2)
            # Different versions of pyspellchecker expose different APIs for
            # seeding the vocabulary. Prefer load_words, then add, then fall
            # back to a safe no-op if neither exists.
            if vocab:
                wf = getattr(spell, "word_frequency", None)
                if wf is not None and hasattr(wf, "load_words"):
                    try:
                        wf.load_words(list(vocab))
                    except Exception:
                        # load_words may fail on some implementations; try add()
                        for w in vocab:
                            try:
                                wf.add(w)
                            except Exception:
                                pass
                elif wf is not None and hasattr(wf, "add"):
                    for w in vocab:
                        try:
                            wf.add(w)
                        except Exception:
                            pass
                else:
                    # Last-resort: if word_frequency behaves like a dict-like
                    # object attempt to populate it gently.
                    try:
                        for w in vocab:
                            try:
                                wf[w] = wf.get(w, 1) if hasattr(wf, "get") else 1
                            except Exception:
                                pass
                    except Exception:
                        pass

            print("Spell checker initialized.")
        except Exception as e:
            st.error(f"Error initializing spell checker: {e}")
            spell = None  # Disable spell checking on error
    else:
        st.warning(
            "Warning: 'pyspellchecker' library not installed. Spell checking disabled. Run 'pip install pyspellchecker'"
        )
        spell = None  # Disable spell checking

    # Try to load a serialized sklearn pipeline or model if it exists
    # Look for common filenames produced by notebooks: 'pipeline.pkl', 'model.joblib', 'model.pkl'
    for candidate in ("pipeline.pkl", "model.joblib", "model.pkl", "eco_model.pkl"):
        candidate_path = os.path.join(os.getcwd(), candidate)
        if os.path.exists(candidate_path):
            try:
                model = joblib.load(candidate_path)
                st.success(f"Loaded ML model from: {candidate}")
                break
            except Exception as e:
                st.warning(f"Found model file '{candidate}' but failed to load it: {e}")

    return df, vocab, spell, model


# --- Load the data when the script runs ---
df, vocab, spell, model = load_assets()


# --- 3. Helper Functions ---


def tokenize_text(s):
    return [t for t in re.findall(r"[a-zA-Z]+", str(s).lower()) if len(t) >= 2]


@st.cache_data
def get_product_categories(_df):
    """Extract significant category-like words from product names"""
    if _df is None:
        return []

    categories = set()
    ignore_words = {
        "a",
        "an",
        "the",
        "with",
        "for",
        "in",
        "of",
        "and",
        "or",
        "new",
        "eco",
    }
    for name in _df["Name"].dropna():
        words = re.findall(r"[a-zA-Z]+", str(name).lower())
        for word in words:
            if len(word) > 3 and word not in ignore_words:
                categories.add(word)
    return list(categories)


def load_data():
    """Return a DataFrame for the app.

    Prefer the DataFrame loaded by load_assets() if available, otherwise read
    the CSV file `Sustainable_data.csv` from the current working directory.
    """
    # If load_assets already populated a global `df`, use it
    if "df" in globals() and globals().get("df") is not None:
        return globals().get("df")

    # Prefer the user's dataset path if provided
    csv_path = os.path.join(os.getcwd(), "Sustainable.csv")
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Failed to read dataset at '{csv_path}': {e}")
            return None

    st.error(f"Dataset not found at '{csv_path}'.")
    return None


def build_master_vocab(_df):
    """Build a master vocabulary set from textual columns in the dataframe.

    Returns a list of vocabulary words (lowercased).
    """
    vocab = set()
    if _df is None:
        return []
    text_columns = ["Name", "Components Used", "Packaging"]
    for col in text_columns:
        if col in _df.columns:
            for cell in _df[col].fillna(""):
                for t in re.findall(r"[a-zA-Z]+", str(cell).lower()):
                    if len(t) >= 2:
                        vocab.add(t)
    return sorted(vocab)


def resolve_column(df_obj, preferred_candidates):
    """Try to resolve a column name from a list of candidate names.

    Returns the actual column name present in df_obj or None if not found.
    Uses exact match (case-insensitive), underscore/space normalization, then
    a fuzzy-match fallback with a conservative score cutoff.
    """
    if df_obj is None:
        return None

    cols = list(df_obj.columns)
    # Build normalized map
    norm_map = {c.lower().strip(): c for c in cols}

    for cand in preferred_candidates:
        key = cand.lower().strip()
        if key in norm_map:
            return norm_map[key]
        # Try replacing spaces/underscores
        alt = key.replace(" ", "_")
        for c in cols:
            if c.lower().strip().replace(" ", "_") == alt:
                return c

    # Fuzzy fallback
    try:
        for cand in preferred_candidates:
            match = process.extractOne(cand, cols, scorer=fuzz.WRatio)
            if match and match[1] >= 75:
                return match[0]
    except Exception:
        pass

    return None


def get_prediction_details(user_text, best_match_row):
    """Produce a small human-readable explanation for a prediction.

    Returns: (level, alternative, match_name, reason)
    """
    if best_match_row is None:
        return "Unknown", "No alternative", "Unknown", "No match data available."

    # Use keys that may vary between datasets. Default to the common names,
    # but accept rows with different column headers.
    level = (
        best_match_row.get("Sustainability_Level")
        or best_match_row.get("Sustainability Level")
        or best_match_row.get("Sustainability_Score")
        or best_match_row.get("sustainability_level")
        or "Unknown"
    )
    alternative = (
        best_match_row.get("Sustainable_Alternative")
        or best_match_row.get("Sustainable Alternative")
        or best_match_row.get("Alternative")
        or best_match_row.get("Suggested Alternative")
        or "No alternative provided"
    )
    match_name = (
        best_match_row.get("Name")
        or best_match_row.get("Product Name")
        or best_match_row.get("product")
        or "Unknown"
    )

    # Compute a simple Jaccard-style similarity on tokens for an explanation
    utoks = set(re.findall(r"[a-zA-Z]+", str(user_text).lower()))
    target_text = " ".join(
        [
            str(best_match_row.get(c, ""))
            for c in ["Name", "Components Used", "Packaging"]
        ]
    )
    ttoks = set(re.findall(r"[a-zA-Z]+", str(target_text).lower()))
    common = sorted(utoks & ttoks)
    union = utoks | ttoks
    score = len(common) / len(union) if len(union) > 0 else 0.0
    if common:
        reason = f"Matched on words: {', '.join(common)} (similarity {score:.2f})."
    else:
        reason = f"Low token overlap with best match (similarity {score:.2f})."

    return level, alternative, match_name, reason


# --- Load data on script run ---
df = load_data()
if df is not None:
    MASTER_VOCAB = build_master_vocab(df)
    CATEGORIES = get_product_categories(df)
else:
    MASTER_VOCAB = ["plastic", "bottle", "sustainable"]
    CATEGORIES = ["bottle"]


def enhanced_spell_check(text, master_vocab_list, category_list):
    """
    Finds typos and category suggestions.
    Returns: (correction_map, category_suggestions, note)
    - correction_map: List of tuples [('original', ['sugg1', 'sugg2']), ...]
    - category_suggestions: List of strings ["'word' (did you mean: ...)", ...]
    - note: String "Auto-corrected | Category suggestions"
    """
    original_text = text.strip().lower()
    # Use regex to find all words, including those with apostrophes
    words = re.findall(r"[a-zA-Z']+", original_text)

    correction_map = []  # List of ('original', [list_of_suggestions])
    category_suggestions = []  # List of strings
    note_parts = []

    master_vocab_set = set(master_vocab_list)
    # Keep track of words we've already offered a correction for
    corrected_originals = set()

    for word in words:
        if word in master_vocab_set or len(word) < 3 or word in corrected_originals:
            continue

        # 1. Find multiple direct typo corrections
        # Use process.extract to get a few good suggestions
        suggestions = process.extract(
            word,
            master_vocab_list,
            scorer=fuzz.WRatio,
            limit=3,  # Get top 3 suggestions
            score_cutoff=80,  # Lowered cutoff to get more options
        )

        # suggestions is a list of tuples: [(match, score, index)]
        # We only want the matches, and only if they are different from the word
        valid_suggestions = [match for match, score, _ in suggestions if match != word]

        if valid_suggestions:
            # Found one or more suggestions
            correction_map.append((word, valid_suggestions))
            corrected_originals.add(word)
            if "Auto-corrected" not in note_parts:
                note_parts.append("Auto-corrected")
        else:
            # 2. No typo found, check for category hints
            category_matches = process.extract(
                word, category_list, scorer=fuzz.partial_ratio, limit=2, score_cutoff=80
            )

            cat_suggestions = []
            for cat_match, cat_score, _ in category_matches:
                if (
                    cat_match != word
                    and cat_match not in word
                    and word not in cat_match
                ):
                    cat_suggestions.append(cat_match)

            if cat_suggestions:
                unique_suggestions = list(set(cat_suggestions))
                category_suggestions.append(
                    f"'{word}' (did you mean: {', '.join(unique_suggestions)}?)"
                )
                if "Category suggestions" not in note_parts:
                    note_parts.append("Category suggestions")

    note = " | ".join(note_parts)

    # Return map of corrections, list of category hints, and note
    return correction_map, category_suggestions, note


# Auto-correcting spell-checker (non-interactive)
def validate_and_correct_input(user_text, vocab_set, spell_checker):
    txt = str(user_text).strip()
    if not txt or len(txt) < 3:
        return None, "Input is too short."

    tokens = tokenize_text(txt)
    if not tokens:
        return None, "No valid words found."

    # Check if any token is known, case-insensitively
    found = [t for t in tokens if t in vocab_set]
    if found:
        return txt, None  # Known words found, skip spell-check

    # Auto-correct logic (only if spellchecker is available)
    corrected_text = txt
    corrections_made = []
    if spell_checker:  # Check if spell object exists and is not None
        for t in tokens:
            if t not in vocab_set:
                suggestion = spell_checker.correction(t)
                # Check if suggestion is different and exists
                if suggestion and suggestion != t:
                    # Use regex to replace whole word only
                    corrected_text = re.sub(
                        r"\b" + re.escape(t) + r"\b",
                        suggestion,
                        corrected_text,
                        flags=re.IGNORECASE,
                    )
                    corrections_made.append(f"'{t}' -> '{suggestion}'")

    if corrections_made:
        correction_note = f"Auto-corrected: {', '.join(corrections_made)}."
        return corrected_text, correction_note

    # If no known words and no corrections, return original text with a note
    return corrected_text, "No known keywords found, but no corrections made."


# Similarity Search (Find best match based on text similarity)
def find_best_match(user_text, df):
    utoks = set(tokenize_text(user_text))
    if not utoks or df is None:  # Check if df is loaded
        return None  # Return None if no tokens or df not loaded

    scores = []
    # Combine relevant text columns for matching - ENSURE THESE EXIST
    search_cols = ["Name", "Components Used", "Packaging"]
    # Check if columns exist before creating search_text
    valid_search_cols = [col for col in search_cols if col in df.columns]
    if not valid_search_cols:
        st.error(
            "Error: Could not find 'Name', 'Components Used', or 'Packaging' columns for similarity search."
        )
        return None

    df["search_text"] = df[valid_search_cols].fillna("").agg(" ".join, axis=1)

    for idx, row in df.iterrows():
        target_tokens = set(tokenize_text(row["search_text"]))
        if not target_tokens:
            continue  # Skip empty rows

        inter = len(utoks & target_tokens)
        union = len(utoks | target_tokens)
        score = inter / union if union > 0 else 0
        if score > 0:
            scores.append((idx, score))

    # Clean up temporary column
    df.drop(columns=["search_text"], inplace=True, errors="ignore")

    if not scores:
        return None

    # Find the single best match based on Jaccard score
    best_match_idx, best_score = sorted(scores, key=lambda x: -x[1])[0]

    # Return the *entire row* of the best match
    return df.iloc[best_match_idx]


# --- 4. The Streamlit App UI ---

# --- Header Section ---
col1, col2 = st.columns([8, 1])
with col1:
    st.markdown(
        '<div class="header"><h1>🌿 EcoSence | Sustainability Predictor</h1></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        '<div class="header"><span style="font-size: 55px;">🌿</span></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    "Enter a product description to predict if it's sustainable based on similar products in our database."
)

# --- Input Section ---
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# --- Initialize Session State ---
if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False
if "skip_suggestions" not in st.session_state:
    st.session_state.skip_suggestions = False
if "user_description" not in st.session_state:
    st.session_state.user_description = "a plastc botle"  # Default
if "word_to_correct" not in st.session_state:
    # This new state stores the word the user has selected
    st.session_state.word_to_correct = None
if "correction_map" not in st.session_state:
    st.session_state.correction_map = []

# --- FIX: Apply pending text update BEFORE rendering the widget ---
# Check if a correction was just applied in the previous run
if "new_text" in st.session_state:
    st.session_state.user_description = st.session_state.new_text  # Set the main state
    del st.session_state.new_text  # Clean up the temporary variable

# User Input
st.markdown('<p class="input-label">Product Description:</p>', unsafe_allow_html=True)
# The text_input widget's value is now controlled by session_state
# This will now correctly use the updated st.session_state.user_description
user_description = st.text_input(
    " ", key="user_description", label_visibility="collapsed"
)

# --- Predict Button ---
if st.button("🔍 Predict", key="predict", use_container_width=True):
    st.session_state.run_prediction = True
    st.session_state.skip_suggestions = False  # Reset skip flag
    st.session_state.word_to_correct = None  # Reset word selection
    st.rerun()

# --- Main Prediction Logic Block ---
if st.session_state.run_prediction:

    if df is None:
        st.error("Application is not ready. Data file could not be loaded.")
        st.session_state.run_prediction = False
    elif not user_description.strip():
        st.warning("Please enter a product description.")
        st.session_state.run_prediction = False
    else:
        # 1. Run Spell Check ONLY if we don't have a map already
        if not st.session_state.word_to_correct:
            correction_map, category_suggestions, spell_note = enhanced_spell_check(
                user_description, MASTER_VOCAB, CATEGORIES
            )
            st.session_state.correction_map = correction_map
        else:
            # We are in the middle of a correction, use the stored map
            correction_map = st.session_state.correction_map
            category_suggestions = (
                []
            )  # Don't show category hints while confirming a word

        # We show suggestions if there are corrections to be made AND
        # the user has not clicked "Keep Original"
        show_suggestion_box = (
            bool(correction_map) and not st.session_state.skip_suggestions
        )

        if show_suggestion_box:
            # --- DISPLAY SUGGESTION BOX ---
            st.markdown('<div class="suggestions-box">', unsafe_allow_html=True)

            # Check if user has selected a word to correct
            if st.session_state.word_to_correct is None:
                # --- Step 1: Show list of misspelled words ---
                st.subheader("💡 Select a word to correct:")

                num_cols = min(len(correction_map), 3)  # Max 3 buttons per row
                if num_cols > 0:
                    cols = st.columns(num_cols)
                    for i, (original, corrected_list) in enumerate(correction_map):
                        col = cols[i % num_cols]
                        with col:
                            button_key = f"select_{original}"
                            # Show only the original word on the button
                            if st.button(
                                original, key=button_key, use_container_width=True
                            ):
                                # User selected this word. Store it and rerun.
                                st.session_state.word_to_correct = original
                                st.rerun()

                # Show category suggestions as text (if any)
                if category_suggestions:
                    st.write("**Other suggestions:**")
                    for suggestion in category_suggestions:
                        st.write(f"• {suggestion}")

            else:
                # --- Step 2: Show suggestions for the selected word ---
                selected_original = st.session_state.word_to_correct
                # Find the correction list from the map
                suggestion_list = []
                for o, s_list in st.session_state.correction_map:  # Use stored map
                    if o == selected_original:
                        suggestion_list = s_list
                        break

                if suggestion_list:
                    st.subheader(f"Select a correction for '{selected_original}':")

                    # Create buttons for each suggestion
                    num_cols = min(len(suggestion_list), 1)
                    if num_cols > 0:
                        cols = st.columns(num_cols)
                        for i, corrected_word in enumerate(suggestion_list):
                            col = cols[i % num_cols]
                            with col:
                                if st.button(
                                    corrected_word,
                                    key=f"apply_{corrected_word}",
                                    use_container_width=True,
                                ):
                                    # --- Apply this correction ---
                                    current_text = st.session_state.user_description
                                    new_text = re.sub(
                                        r"\b" + re.escape(selected_original) + r"\b",
                                        corrected_word,
                                        current_text,
                                        count=1,
                                        flags=re.IGNORECASE,
                                    )

                                    # --- FIX: Use temp state var ---
                                    st.session_state.new_text = new_text
                                    st.session_state.word_to_correct = (
                                        None  # Reset selection
                                    )
                                    st.session_state.correction_map = (
                                        []
                                    )  # Clear map to force re-check
                                    st.rerun()

                    st.markdown("---")  # Divider
                    # Add a 'Go Back' button
                    if st.button(
                        "⬅️ Go Back (Select different word)",
                        key="confirm_back",
                        use_container_width=True,
                    ):
                        st.session_state.word_to_correct = None  # Reset selection
                        st.rerun()
                else:
                    # Failsafe in case state gets weird
                    st.session_state.word_to_correct = None
                    st.rerun()

            st.markdown("---")  # Divider

            # Add control buttons (Keep Original / Cancel)
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "➡️ Keep Original and Predict",
                    key="keep_original_predict",
                    use_container_width=True,
                ):
                    st.session_state.skip_suggestions = True
                    st.session_state.word_to_correct = None
                    st.rerun()
            with col2:
                if st.button(
                    "❌ Cancel Correction",
                    key="cancel_suggestions",
                    use_container_width=True,
                ):
                    st.session_state.run_prediction = False
                    st.session_state.skip_suggestions = False
                    st.session_state.word_to_correct = None
                    # --- FIX: Use temp state var ---
                    st.session_state.new_text = "a plastc botle"  # Reset to default
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            # --- PROCEED WITH PREDICTION ---
            # This runs if:
            # 1. No corrections were found (correction_map is empty)
            # 2. User clicked "Keep Original and Predict"

            use_text = st.session_state.user_description
            st.info(f"**Analyzing:** '{use_text}'")

            best_match = find_best_match(use_text, df)

            if best_match is None:
                st.error(
                    "❌ Sorry, I couldn't find a similar product in my database.\n\n"
                    "**Tips:**\n"
                    "• Be specific about materials (plastic, metal, glass, etc.)\n"
                    "• Use common product names\n"
                )
            else:
                # Resolve dataset column names flexibly to support variants
                level_col = (
                    resolve_column(
                        df,
                        [
                            "Sustainability_Level",
                            "Sustainability Level",
                            "Sustainability_Score",
                            "Sustainability Score",
                            "sustainability_level",
                        ],
                    )
                    or "Sustainability_Level"
                )

                alternative_col = (
                    resolve_column(
                        df,
                        [
                            "Sustainable_Alternative",
                            "Sustainable Alternative",
                            "Alternative",
                            "Suggested Alternative",
                            "sustainable_alternative",
                        ],
                    )
                    or "Sustainable_Alternative"
                )

                name_col = (
                    resolve_column(df, ["Name", "Product Name", "product"]) or "Name"
                )

                missing = [
                    c
                    for c in (level_col, alternative_col, name_col)
                    if c not in best_match.index
                ]
                if missing:
                    st.error(
                        "Error: Missing required columns in the dataset."
                        f" Could not find columns: {missing}. Available columns: {', '.join(df.columns)}"
                    )
                else:
                    # Use flexible getter in get_prediction_details (it already handles
                    # common alternate names) but pass through the resolved names for
                    # clarity in the UI and future extensions.
                    level, alternative, match_name, reason = get_prediction_details(
                        use_text, best_match
                    )

                    st.markdown("---")
                    st.subheader("🎯 Prediction Result:")

                    if str(level).strip().lower() == "high":
                        st.balloons()
                        st.markdown(
                            '<div class="success-box">✅ Yes, likely sustainable!</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(reason)  # Use the new dynamic reason
                        st.markdown(
                            f"**Original Match Level:** {best_match[level_col]}"
                        )

                        st.write("### 🌍 Eco Score (Estimated)")
                        st.progress(0.9)
                        st.success("High Sustainability Score: 90%")
                    else:
                        st.markdown(
                            '<div class="error-box">❌ No, likely not sustainable.</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(reason)  # Use the new dynamic reason
                        st.markdown(
                            f"**Original Match Level:** {best_match[level_col]}"
                        )

                        st.write("### 🌍 Eco Score (Estimated)")
                        st.progress(0.4)
                        st.error("Low Sustainability Score: 40%")

                        st.markdown("---")
                        st.subheader("💡 Suggested Sustainable Alternative:")
                        st.info(f"**{alternative}**")

            # Reset flags so the prediction doesn't run in a loop
            st.session_state.run_prediction = False
            st.session_state.skip_suggestions = False
            st.session_state.word_to_correct = None

st.markdown("</div>", unsafe_allow_html=True)  # Close main-box

# --- Refresh Button ---
if st.button("🔄 Start Over", key="refresh", use_container_width=True):
    st.session_state.run_prediction = False
    st.session_state.skip_suggestions = False
    st.session_state.word_to_correct = None
    # --- FIX: Use temp state var ---
    st.session_state.new_text = "a plastc botle"  # Reset to default
    st.rerun()
