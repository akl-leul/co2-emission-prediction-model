-- Supabase Database Schema for CO2 Charcoal Emissions App
-- Run this in your Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table for emission summaries (learning history)
CREATE TABLE IF NOT EXISTS emission_summaries (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    n_samples INTEGER,
    n_raw_rows INTEGER,
    slope_a DECIMAL(10, 6),
    intercept_b DECIMAL(10, 6),
    r2_score DECIMAL(10, 6),
    auto_emission_factor DECIMAL(10, 6),
    emissions_1yr DECIMAL(15, 6),
    emissions_2yr DECIMAL(15, 6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for full datasets (raw data, processed data, predictions)
CREATE TABLE IF NOT EXISTS emission_datasets (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_file TEXT,
    n_raw_rows INTEGER,
    n_processed_rows INTEGER,
    model_params JSONB,
    is_ground_truth BOOLEAN DEFAULT FALSE,
    raw_data_table JSONB,
    processed_data_table JSONB,
    predictions_list JSONB,
    forecasts JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for individual data records (optional for detailed queries)
CREATE TABLE IF NOT EXISTS emission_records (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    dataset_id UUID REFERENCES emission_datasets(id) ON DELETE CASCADE,
    week INTEGER,
    total_charcoal_kg DECIMAL(10, 6),
    predicted_charcoal DECIMAL(10, 6),
    prediction_error DECIMAL(10, 6),
    co2_kg DECIMAL(10, 6),
    households INTEGER,
    avg_charcoal_kg DECIMAL(10, 6),
    frequency_per_week DECIMAL(5, 2),
    charcoal_per_use_kg DECIMAL(10, 6),
    household_size DECIMAL(10, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_emission_summaries_timestamp ON emission_summaries(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_emission_datasets_timestamp ON emission_datasets(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_emission_datasets_ground_truth ON emission_datasets(is_ground_truth);
CREATE INDEX IF NOT EXISTS idx_emission_records_dataset_id ON emission_records(dataset_id);
CREATE INDEX IF NOT EXISTS idx_emission_records_week ON emission_records(week);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to automatically update updated_at
CREATE TRIGGER update_emission_summaries_updated_at 
    BEFORE UPDATE ON emission_summaries 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_emission_datasets_updated_at 
    BEFORE UPDATE ON emission_datasets 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) - Enable RLS on tables
ALTER TABLE emission_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE emission_datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE emission_records ENABLE ROW LEVEL SECURITY;

-- RLS Policies (allow all operations for now - customize as needed)
CREATE POLICY "Allow all operations on emission_summaries" ON emission_summaries
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all operations on emission_datasets" ON emission_datasets
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all operations on emission_records" ON emission_records
    FOR ALL USING (true) WITH CHECK (true);

-- Grant necessary permissions
GRANT ALL ON emission_summaries TO anon;
GRANT ALL ON emission_summaries TO authenticated;
GRANT ALL ON emission_datasets TO anon;
GRANT ALL ON emission_datasets TO authenticated;
GRANT ALL ON emission_records TO anon;
GRANT ALL ON emission_records TO authenticated;
