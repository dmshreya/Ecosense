import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from rapidfuzz import process, fuzz

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


# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Sustainability Predictor", page_icon="♻️", layout="centered"
)


# --- 2. Load Assets (Simplified & Corrected Filename) ---
@st.cache_resource
def load_assets():
    # --- Use the exact filename provided by the user ---
    df_path = "sustainable_Dataset.csv"
    vocab_path = "app_vocab.json"  # Assumes you saved this from your notebook

    df = None
    vocab = None
    spell = None

    # Load DataFrame
    try:
        df = pd.read_csv(df_path)
        st.success(f"Successfully loaded data from: {df_path}")
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
        st.warning(
            f"Warning: Vocab file not found at '{vocab_path}'. Spell check might be limited. Building vocab from data..."
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

            st.success("Spell checker initialized.")
        except Exception as e:
            st.error(f"Error initializing spell checker: {e}")
            spell = None  # Disable spell checking on error
    else:
        st.warning(
            "Warning: 'pyspellchecker' library not installed. Spell checking disabled. Run 'pip install pyspellchecker'"
        )
        spell = None  # Disable spell checking

    return df, vocab, spell


# --- Load the data when the script runs ---
df, vocab, spell = load_assets()


# --- 3. Helper Functions ---
def tokenize_text(s):
    s = str(s)
    tokens = re.findall(r"[a-zA-Z]+", s.lower())
    return [t for t in tokens if len(t) >= 2]


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

st.title("♻️ Sustainability Predictor")
st.markdown(
    "Enter a product description to predict if it's sustainable based on similar products in our database."
)

# User Input - Provide a default example (give the text_input a key so we can update it)
user_description = st.text_input(
    "Product Description:", "a plastc bottle", key="user_description"
)

if st.button("Predict"):
    # --- Input Validation ---
    if df is None:
        st.error("Application is not ready. Data file could not be loaded.")
    elif not user_description:
        st.warning("Please enter a product description.")
    else:
        # 1. Correct Spelling (if spellchecker available)
        corrected_description, note = validate_and_correct_input(
            user_description, vocab, spell
        )

        # If there's a suggested correction different from the input, show it and offer a button to accept it
        use_text = user_description
        if (
            corrected_description
            and corrected_description.strip().lower()
            != user_description.strip().lower()
        ):
            st.info(f"Did you mean: **{corrected_description}** ?")
            # If user clicks this, update the text input and use corrected text for analysis
            if st.button("Use suggestion", key="use_suggestion"):
                st.session_state["user_description"] = corrected_description
                use_text = corrected_description
                # no rerun: continue using corrected text in this run

        # If no suggestion or user didn't accept, fall back to either corrected (if no suggestion was shown) or original input
        if use_text == user_description:
            # If there was no differing suggestion but correction produced something (e.g., casing), prefer it
            if (
                corrected_description
                and corrected_description.strip().lower()
                == user_description.strip().lower()
            ):
                use_text = corrected_description

        # Display which text we're analyzing
        st.info(f"Analyzing: **'{use_text}'**...")
        if note and "Auto-corrected" in note:
            st.warning(note)
        elif note:
            st.info(note)

        # 2. Find the best matching product in the database
        best_match = find_best_match(use_text, df)  # Pass the chosen text

        if best_match is None:
            st.error(
                "Sorry, I couldn't find a similar product in my database to analyze."
            )
        else:
            # 3. Get the prediction and alternative from the matched product
            # --- Make sure these column names match your CSV EXACTLY ---
            level_col = "Sustainability_Level"
            alternative_col = "Sustainable_Alternative"
            name_col = "Name"

            # Check if columns exist in the Series returned by .iloc
            if (
                level_col not in best_match.index
                or alternative_col not in best_match.index
                or name_col not in best_match.index
            ):
                st.error(
                    f"Error: Missing required columns ('{level_col}', '{alternative_col}', '{name_col}') in the best match data."
                )
                st.write("Best match data found:")
                st.write(best_match)  # Show what was found
            else:
                level = best_match[level_col]
                alternative = best_match[alternative_col]
                match_name = best_match[name_col]

                st.markdown("---")
                st.subheader("Prediction Result:")

                # 4. Give the Yes/No answer and the alternative
                if str(level).strip().lower() == "high":
                    st.success("✅ **Yes, likely sustainable.**")
                    st.markdown(
                        f"Based on similarity to: **{match_name}** (Level: {level})"
                    )

                else:
                    st.error("❌ **No, likely not sustainable.**")
                    st.markdown(
                        f"Based on similarity to: **{match_name}** (Level: {level})"
                    )

                    st.markdown("---")
                    st.subheader("💡 Suggested Sustainable Alternative:")
                    st.info(alternative)


# Data Explorer removed as per user request. Only the prediction UI remains.
