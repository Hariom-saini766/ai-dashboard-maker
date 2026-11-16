# AI Dashboard Maker

Upload an Excel/CSV sheet, type what you want in plain English (or ask a question like
"how many passengers went from India"), and get an instant dashboard with KPIs, charts,
filters and quick insights.

Built with Python, Streamlit, Pandas, Plotly and the Google Gemini API (free key).

## Features
- Auto-detects the header row, number and date columns (handles messy Excel sheets)
- AI planner (Gemini) turns any prompt or question into KPIs, charts and row filters
- All calculations are done in Pandas, so numbers stay accurate (the AI only plans)
- Works without a key too: keyword-based fallback planner and an Auto dashboard button
- Sidebar filters, manual chart builder, Indian number format (K / L / Cr)

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Get a free Gemini API key at https://aistudio.google.com and paste it in the sidebar.
Only column names, some category values and 5 sample rows are sent to the AI, never the full sheet.

## Example prompts
- show store-wise sales, monthly trend, top 10 products and category share
- how many passengers went from India
- average fare by destination country
