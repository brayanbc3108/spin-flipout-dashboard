# Spin Flip-Out dashboard (Team 4, Group 602)

Brayan Beristain Cortez, Roberto Reyes Ruiz, Ana Karen Chavez and Abril Rodriguez Herrera.

**This dashboard helps the CRM lead at Spin decide which users to target next month with a non-OXXO purchase incentive, which message each persona gets, and how many of them to contact.**

Live app: _link added after deployment_

## Run it locally

```
pip install -r requirements.txt
streamlit run app.py
```

The app only reads the small tables in `artifacts/`. It never needs the raw Spin file.

## What is inside

- **Overview:** key numbers, a one-line answer for a 10% campaign, persona size and Flip-Out rate, and our team question: does payday change how people spend?
- **Personas:** six behavior-based personas with who they are, what makes them different and the suggested action.
- **Target list:** a capacity slider (5% to 30%) against a random list, and the ranked list you can filter and download.
- **Model:** the model in use, its drivers, a side by side comparison of any two versions and the experiment log.
- **Migration list:** users who may leave OXXO (Flip-Out) and users who may come back (Flip-In) in one list.

## What changed in this version (Nov 25)

Two new Flip-Out experiments, graded on June and July 2026, months no model saw in training. A random list scores 0.223.

| Version | Change | PR-AUC | Reached at 10% |
|---|---|---|---|
| v5 (Nov 20 delivery) | Six snapshots, newer months weigh more | 0.390 | 20.9% |
| v6 | v5 plus each user's payday spike | 0.391 | 21.0% |
| **v7 (in use)** | v6 plus a 12-month view of the OXXO share | **0.398** | **21.5%** |

Payday barely moves the model, which matches what we see in the data: users make about 22% of their purchases on payday days, the same share those days hold in the calendar. That is why the list is built on each person's buying profile and not on the calendar. The 12-month view helps because six months can hide a user who was far from OXXO earlier in the year.

## Files

- `app.py`: the Streamlit dashboard.
- `artifacts/`: aggregated tables and the scored list with anonymous IDs (short hashes).
- `build_model.py`: rebuilds `artifacts/` from the Evidence 3 notebook. It runs inside the course project because it needs the raw file, which is not in this repository.
- `.streamlit/config.toml`: Spin colors.

## Data

The data is synthetic (made with SDV). The raw file is not in this repository.
