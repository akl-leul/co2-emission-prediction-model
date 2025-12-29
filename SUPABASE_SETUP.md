# Supabase Setup Guide for CO2 Charcoal Emissions App

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Set up Supabase Project
1. Go to [supabase.com](https://supabase.com) and create a new project
2. Note your project URL and API key from Settings > API

## 3. Create Database Tables
1. In your Supabase project, go to the SQL Editor
2. Copy and run the contents of `supabase_schema.sql`
3. This will create the necessary tables: `emission_summaries`, `emission_datasets`, and `emission_records`

## 4. Configure Streamlit Secrets
Create a file `.streamlit/secrets.toml` in your project directory:

```toml
[supabase]
url = "your_supabase_project_url"
key = "your_supabase_anon_key"
```

Replace with your actual Supabase URL and API key.

## 5. Run the App
```bash
streamlit run app.py
```

## Features Added
- ✅ All data automatically saved to Supabase
- ✅ Local file backup (fallback if Supabase fails)
- ✅ Three database tables for organized data storage
- ✅ Individual record tracking for detailed analysis
- ✅ Error handling with user-friendly messages
- ✅ Batch processing to avoid API limits

## Database Schema
- `emission_summaries`: High-level model metrics and summaries
- `emission_datasets`: Complete datasets with raw/processed data and predictions
- `emission_records`: Individual data records for granular queries

## Notes
- The app will save to both Supabase and local files for redundancy
- If Supabase connection fails, you'll see an error message but the app continues working
- Local JSON files are kept as backup (last 20 datasets)
