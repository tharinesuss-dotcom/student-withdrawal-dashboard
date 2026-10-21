# Student Withdrawal Early Warning Dashboard

A web-hosted Streamlit prototype using existing XGBoost prediction outputs. **No model retraining is required.**

## Included

- `app.py` — interactive Lecturer Review and Model Monitoring pages
- `demo_predictions.csv` — **entirely synthetic example data** suitable for public demonstrations, not your research results
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — dashboard theme
- `.gitignore` — prevents accidental commit of named real-data CSVs

## Public demonstration deployment

1. Create a new GitHub repository, such as `withdrawal-warning-demo`.
2. Upload `app.py`, `requirements.txt`, `demo_predictions.csv`, and `.streamlit/config.toml` to the repository. **Do not upload real student data.**
3. Visit https://share.streamlit.io and sign in with GitHub.
4. Choose **Create app**, select your repository and `main` branch, and select `app.py` as the main file.
5. Deploy. Streamlit supplies a shareable `*.streamlit.app` web address.
6. Test Lecturer review and Model monitoring. The on-screen notice confirms synthetic data.

Official deployment guide
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Private / approved institutional installation

`lecturer_intervention_dashboard.csv` is student-level data and should **never be published in a public GitHub repository or unrestricted Streamlit deployment**. Use an institution-approved private server with restricted access, encryption, a retention policy, and appropriate data authorization. A private Streamlit app may be an option if approved by the institution, but merely placing a login link or an app password on an otherwise public app is not a complete security design.

For a local run with your real CSV in an access-controlled environment, install dependencies and set the path.

Windows PowerShell

```
python -m pip install -r requirements.txt
$env:DASHBOARD_CSV_PATH = "C:\\path\\to\\lecturer_intervention_dashboard.csv"
streamlit run app.py
```

macOS / Linux

```
python -m pip install -r requirements.txt
export DASHBOARD_CSV_PATH=/secure/path/lecturer_intervention_dashboard.csv
streamlit run app.py
```

The application automatically uses `demo_predictions.csv` if that environment variable is not set. Real-data hosting requires a protected file volume or approved database made available to the hosted application securely. Do not commit the real file or use the demo application's publicly accessible address with real information.

## Notes

- The review threshold slider is exploratory and does not re-estimate the model or recompute original CSV priority classifications.
- All counts are interval observations unless the metric explicitly refers to distinct student enrolments within the selected scoring day.
- SHAP contributions are not causal effects or directly interpretable as changes in withdrawal probability.
- Recommended actions are not evidence that an intervention has taken place or that it worked.
- The synthetic dataset is not an anonymised transformation of the user's original student-level data.
