-- ============================================================
-- EXAMPLE SQL QUERIES FOR MONOLITH TABLES
-- Copy ONE QUERY AT A TIME and paste into CLI
-- ============================================================

-- 1. List all tables in database
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;

-- 2. Count relationships
SELECT COUNT(*) FROM monolith_series_relationships;

-- 3. Top 10 most connected series (by degree centrality)
SELECT series_id, title, category, degree_centrality, popularity FROM monolith_series_features ORDER BY degree_centrality DESC LIMIT 10;

-- 4. Relationship types breakdown
SELECT relationship_type, COUNT(*) as count FROM monolith_series_relationships GROUP BY relationship_type ORDER BY count DESC;

-- 5. Strongest relationships
SELECT source_id, target_id, relationship_type, strength FROM monolith_series_relationships ORDER BY strength DESC LIMIT 20;

-- 6. Find relationships for a specific series (e.g., GDP)
SELECT r.source_id, r.target_id, r.relationship_type, r.strength, f.title FROM monolith_series_relationships r LEFT JOIN monolith_series_features f ON r.target_id = f.series_id WHERE r.source_id = 'GDPC1' ORDER BY r.strength DESC LIMIT 10;

-- 7. Most popular series in each category
SELECT DISTINCT ON (category) category, series_id, title, popularity FROM monolith_series_features ORDER BY category, popularity DESC;

-- 8. Series with high betweenness centrality (bridge nodes)
SELECT series_id, title, category, betweenness_centrality, degree_centrality FROM monolith_series_features WHERE betweenness_centrality > 0 ORDER BY betweenness_centrality DESC LIMIT 10;

-- 9. Find all series related to 'UNRATE' (unemployment rate)
SELECT r.relationship_type, r.strength, f.series_id, f.title, f.category FROM monolith_series_relationships r JOIN monolith_series_features f ON (r.source_id = f.series_id OR r.target_id = f.series_id) WHERE (r.source_id = 'UNRATE' OR r.target_id = 'UNRATE') AND f.series_id != 'UNRATE' ORDER BY r.strength DESC LIMIT 20;

-- 10. Network density by category (avg connections per series)
SELECT f.category, COUNT(DISTINCT f.series_id) as series_count, COUNT(r.relationship_id) as total_connections, ROUND(COUNT(r.relationship_id)::numeric / COUNT(DISTINCT f.series_id), 2) as avg_connections_per_series FROM monolith_series_features f LEFT JOIN monolith_series_relationships r ON (f.series_id = r.source_id OR f.series_id = r.target_id) GROUP BY f.category HAVING COUNT(DISTINCT f.series_id) > 10 ORDER BY avg_connections_per_series DESC LIMIT 15;

-- 11. Find series similar to housing/mortgage topics
SELECT f.series_id, f.title, f.category, f.popularity, f.degree_centrality FROM monolith_series_features f WHERE f.title ILIKE '%housing%' OR f.title ILIKE '%mortgage%' OR f.category ILIKE '%housing%' ORDER BY f.degree_centrality DESC LIMIT 20;

-- 12. Cross-category relationships (series from different categories connected)
SELECT r.relationship_type, f1.category as source_category, f2.category as target_category, COUNT(*) as count FROM monolith_series_relationships r JOIN monolith_series_features f1 ON r.source_id = f1.series_id JOIN monolith_series_features f2 ON r.target_id = f2.series_id WHERE f1.category != f2.category GROUP BY r.relationship_type, f1.category, f2.category ORDER BY count DESC LIMIT 20;

-- 13. Check table structure
SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'monolith_series_relationships' ORDER BY ordinal_position;

SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'monolith_series_features' ORDER BY ordinal_position;
