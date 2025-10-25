# Run the Sustainability Predictor Streamlit app

Steps to run locally (Windows / bash.exe):

1. Create a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/Scripts/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Open the URL printed in the terminal (usually http://localhost:8501) in your browser.

Notes:

- The app expects `sustainable_Dataset.csv` to be in the same folder as `app.py` (it already is).
- If you don't want spell-checking features, `pyspellchecker` may be omitted.
- If you run into permission issues on Windows when creating virtual environments, run your shell as administrator.
