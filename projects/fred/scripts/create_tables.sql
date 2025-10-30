-- PostgreSQL Schema for FRED Series Metadata
-- Run with: psql -d fred -f create_tables.sql

-- Drop existing tables if they exist (CASCADE to drop dependent objects)
DROP TABLE IF EXISTS series_observations CASCADE;
DROP TABLE IF EXISTS series_relationships CASCADE;
DROP TABLE IF EXISTS series_metadata CASCADE;

-- Main series metadata table
CREATE TABLE series_metadata (
    series_id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    frequency VARCHAR,
    frequency_name VARCHAR,
    popularity INTEGER,
    notes TEXT,
    source_file VARCHAR,
    category VARCHAR,
    has_notes BOOLEAN,
    notes_length INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Time series observations (placeholder for actual data)
-- This would be populated by fetching actual FRED data via API
CREATE TABLE series_observations (
    observation_id SERIAL PRIMARY KEY,
    series_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    value DOUBLE PRECISION,
    FOREIGN KEY (series_id) REFERENCES series_metadata(series_id),
    UNIQUE(series_id, date)
);

-- Series relationships (co-occurrence, correlation, etc.)
CREATE TABLE series_relationships (
    relationship_id SERIAL PRIMARY KEY,
    series_id_1 VARCHAR NOT NULL,
    series_id_2 VARCHAR NOT NULL,
    relationship_type VARCHAR, -- 'correlated', 'co_queried', 'same_category', etc.
    strength DOUBLE PRECISION, -- 0.0 to 1.0
    metadata JSONB,
    FOREIGN KEY (series_id_1) REFERENCES series_metadata(series_id),
    FOREIGN KEY (series_id_2) REFERENCES series_metadata(series_id),
    UNIQUE(series_id_1, series_id_2, relationship_type)
);

-- Create indexes for common queries
CREATE INDEX idx_category ON series_metadata(category);
CREATE INDEX idx_frequency ON series_metadata(frequency);
CREATE INDEX idx_popularity ON series_metadata(popularity DESC);
CREATE INDEX idx_obs_series ON series_observations(series_id);
CREATE INDEX idx_obs_date ON series_observations(date);
CREATE INDEX idx_rel_series1 ON series_relationships(series_id_1);
CREATE INDEX idx_rel_series2 ON series_relationships(series_id_2);

-- Create views for common queries
CREATE VIEW high_popularity_series AS
SELECT series_id, title, category, frequency_name, popularity
FROM series_metadata
WHERE popularity >= 70
ORDER BY popularity DESC;

CREATE VIEW series_by_category AS
SELECT category, 
       COUNT(*) as series_count,
       AVG(popularity) as avg_popularity,
       COUNT(DISTINCT frequency) as frequency_types
FROM series_metadata
GROUP BY category
ORDER BY series_count DESC;

-- Example: Materialized aggregate for performance
CREATE VIEW daily_series_summary AS
SELECT 
    category,
    COUNT(*) as count,
    AVG(popularity) as avg_popularity,
    MAX(popularity) as max_popularity
FROM series_metadata
WHERE frequency = 'D'
GROUP BY category;
