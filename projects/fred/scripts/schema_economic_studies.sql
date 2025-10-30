-- Economic Studies Database Schema
-- Creates tables to organize research and link to FRED series

-- =====================================================
-- Table: economic_studies
-- Stores famous economic studies and papers
-- =====================================================
CREATE TABLE IF NOT EXISTS economic_studies (
  study_id SERIAL PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  authors VARCHAR(255),
  year_published INT,
  field VARCHAR(100),
  summary TEXT,
  journal VARCHAR(150),
  citations INT DEFAULT 0,
  impact VARCHAR(50),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- Table: fred_series
-- Master list of FRED series referenced in studies
-- =====================================================
CREATE TABLE IF NOT EXISTS fred_series (
  series_id VARCHAR(32) PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  category VARCHAR(100),
  source VARCHAR(50) DEFAULT 'FRED',
  UNIQUE (series_id)
);

-- =====================================================
-- Table: economic_study_series (Junction Table)
-- Many-to-many relationship between studies and series
-- =====================================================
CREATE TABLE IF NOT EXISTS economic_study_series (
  study_id INT NOT NULL REFERENCES economic_studies(study_id) ON DELETE CASCADE,
  series_id VARCHAR(32) NOT NULL REFERENCES fred_series(series_id) ON DELETE RESTRICT,
  is_primary BOOLEAN NOT NULL DEFAULT FALSE,
  notes TEXT,
  PRIMARY KEY (study_id, series_id)
);

-- =====================================================
-- Indexes for better query performance
-- =====================================================
CREATE INDEX IF NOT EXISTS ess_series_idx ON economic_study_series(series_id);
CREATE INDEX IF NOT EXISTS ess_primary_idx ON economic_study_series(is_primary);
CREATE INDEX IF NOT EXISTS studies_year_idx ON economic_studies(year_published);
CREATE INDEX IF NOT EXISTS studies_field_idx ON economic_studies(field);

-- =====================================================
-- Seed common FRED series
-- =====================================================
INSERT INTO fred_series (series_id, name, category) VALUES
('GDP', 'Gross Domestic Product', 'GDP'),
('GDPC1', 'Real Gross Domestic Product', 'GDP'),
('CPIAUCSL', 'Consumer Price Index for All Urban Consumers: All Items', 'Inflation'),
('UNRATE', 'Unemployment Rate', 'Employment'),
('PAYEMS', 'All Employees: Total Nonfarm', 'Employment'),
('M2SL', 'M2 Money Stock', 'Money Supply'),
('PCE', 'Personal Consumption Expenditures', 'Consumer Spending'),
('FEDFUNDS', 'Effective Federal Funds Rate', 'Interest Rates'),
('GS10', '10-Year Treasury Constant Maturity Rate', 'Interest Rates'),
('MORTGAGE30US', '30-Year Fixed Rate Mortgage Average', 'Interest Rates'),
('HOUST', 'Housing Starts: Total New Privately Owned', 'Housing'),
('PERMIT', 'New Privately-Owned Housing Units Authorized', 'Housing'),
('DEXUSEU', 'U.S. / Euro Foreign Exchange Rate', 'Exchange Rates'),
('SP500', 'S&P 500', 'Markets'),
('VIXCLS', 'CBOE Volatility Index: VIX', 'Markets')
ON CONFLICT (series_id) DO NOTHING;

-- =====================================================
-- Example famous studies
-- =====================================================
INSERT INTO economic_studies 
(title, authors, year_published, field, summary, journal, citations, impact)
VALUES
('The General Theory of Employment, Interest, and Money', 
 'John Maynard Keynes', 1936, 'Macroeconomics',
 'Introduced Keynesian economics, arguing that aggregate demand drives economic cycles.',
 'Macmillan', 50000, 'High'),

('The Market for "Lemons": Quality Uncertainty and the Market Mechanism',
 'George Akerlof', 1970, 'Information Economics',
 'Explains how asymmetric information causes market failure.',
 'Quarterly Journal of Economics', 45000, 'High'),

('Economic Growth and Income Inequality',
 'Simon Kuznets', 1955, 'Development Economics',
 'Proposed the Kuznets Curve linking inequality with stages of economic growth.',
 'American Economic Review', 35000, 'High'),

('Rational Expectations and the Theory of Price Movements',
 'John Muth', 1961, 'Expectations Theory',
 'Introduced the rational expectations hypothesis.',
 'Econometrica', 30000, 'High'),

('Employment, Interest and Money',
 'Don Patinkin', 1956, 'Monetary Economics',
 'Extended Keynesian theory with explicit consideration of money.',
 'Row, Peterson and Company', 15000, 'Moderate')
ON CONFLICT DO NOTHING;

-- =====================================================
-- Link studies to FRED series
-- =====================================================

-- Keynes' General Theory uses GDP, Unemployment, Interest Rates
INSERT INTO economic_study_series (study_id, series_id, is_primary, notes) VALUES
(1, 'GDP', TRUE, 'Central focus on aggregate output'),
(1, 'UNRATE', TRUE, 'Employment is key variable'),
(1, 'FEDFUNDS', FALSE, 'Interest rates affect investment');

-- Akerlof's Lemons paper references market prices
INSERT INTO economic_study_series (study_id, series_id, is_primary, notes) VALUES
(2, 'CPIAUCSL', FALSE, 'Price levels in markets'),
(2, 'PCE', FALSE, 'Consumer expenditure patterns');

-- Kuznets' inequality work uses GDP growth
INSERT INTO economic_study_series (study_id, series_id, is_primary, notes) VALUES
(3, 'GDP', TRUE, 'Economic growth measurement'),
(3, 'GDPC1', TRUE, 'Real economic growth');

-- Muth's rational expectations uses various indicators
INSERT INTO economic_study_series (study_id, series_id, is_primary, notes) VALUES
(4, 'CPIAUCSL', TRUE, 'Price expectations'),
(4, 'FEDFUNDS', FALSE, 'Interest rate expectations');

-- Patinkin's monetary theory
INSERT INTO economic_study_series (study_id, series_id, is_primary, notes) VALUES
(5, 'M2SL', TRUE, 'Money supply analysis'),
(5, 'FEDFUNDS', TRUE, 'Interest rates and liquidity'),
(5, 'GDP', FALSE, 'Output and money relationship');

-- =====================================================
-- Helpful Views
-- =====================================================

-- View: Studies with their series
CREATE OR REPLACE VIEW study_series_view AS
SELECT 
    es.study_id,
    es.title,
    es.authors,
    es.year_published,
    fs.series_id,
    fs.name AS series_name,
    ess.is_primary,
    ess.notes
FROM economic_studies es
JOIN economic_study_series ess ON es.study_id = ess.study_id
JOIN fred_series fs ON ess.series_id = fs.series_id
ORDER BY es.year_published DESC, es.study_id, ess.is_primary DESC;

-- View: Series usage count
CREATE OR REPLACE VIEW series_usage_view AS
SELECT 
    fs.series_id,
    fs.name,
    fs.category,
    COUNT(ess.study_id) AS study_count,
    SUM(CASE WHEN ess.is_primary THEN 1 ELSE 0 END) AS primary_count
FROM fred_series fs
LEFT JOIN economic_study_series ess ON fs.series_id = ess.series_id
GROUP BY fs.series_id, fs.name, fs.category
ORDER BY study_count DESC, primary_count DESC;

-- =====================================================
-- Common Queries (as comments for reference)
-- =====================================================

/*
-- Find all studies that use GDP
SELECT es.title, es.authors, es.year_published
FROM economic_studies es
JOIN economic_study_series ess ON es.study_id = ess.study_id
WHERE ess.series_id = 'GDP';

-- Find all series used in a specific study
SELECT fs.series_id, fs.name, ess.is_primary
FROM fred_series fs
JOIN economic_study_series ess ON fs.series_id = ess.series_id
WHERE ess.study_id = 1;

-- Most referenced FRED series
SELECT * FROM series_usage_view WHERE study_count > 0;

-- Studies by field that use unemployment data
SELECT es.title, es.field, es.year_published
FROM economic_studies es
JOIN economic_study_series ess ON es.study_id = ess.study_id
WHERE ess.series_id = 'UNRATE'
  AND es.field = 'Macroeconomics';

-- Add a new study
INSERT INTO economic_studies (title, authors, year_published, field, summary)
VALUES ('Your Study Title', 'Your Name', 2024, 'Your Field', 'Description');

-- Link the new study to series
INSERT INTO economic_study_series (study_id, series_id, is_primary)
VALUES (CURRVAL('economic_studies_study_id_seq'), 'GDP', TRUE);
*/
