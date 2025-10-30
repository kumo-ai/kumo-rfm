# Customizing FRED Search Terms

There are now **4 ways** to specify what data to fetch from FRED:

## Option 1: Custom Terms File (NEW! Recommended)

Create/edit `search_terms.txt` with your own terms:

```bash
# Edit the file
nano search_terms.txt

# Fetch using your custom terms
python3 01_fetch_fred_data.py --from-file search_terms.txt
```

**search_terms.txt format:**
```
# Comments start with #
GDP
inflation
unemployment
your custom term here
```

## Option 2: Command Line (Quick)

```bash
# Single term
python3 01_fetch_fred_data.py --search "GDP"

# Multiple terms
python3 01_fetch_fred_data.py --search "GDP" "inflation" "unemployment"

# With custom limit
python3 01_fetch_fred_data.py --search "employment" --limit 100
```

## Option 3: Predefined Categories

```bash
# List available categories
python3 01_fetch_fred_data.py --list-categories

# Fetch a category
python3 01_fetch_fred_data.py --category core      # GDP, inflation, etc.
python3 01_fetch_fred_data.py --category housing   # Housing indicators
python3 01_fetch_fred_data.py --category monetary  # Interest rates, M1, M2
```

**Available categories:**
- `core` - GDP, inflation, CPI, unemployment, wages
- `monetary` - Interest rates, money supply, bank reserves
- `trade` - Exports, imports, exchange rates
- `housing` - Housing starts, home prices, rent
- `consumer` - Retail sales, consumer confidence
- `business` - Industrial production, manufacturing
- `debt` - Household, corporate, government debt
- `markets` - Stock market, S&P 500, volatility
- `commodities` - Energy, oil, food prices

## Option 4: All Predefined Terms

```bash
# Fetch ALL predefined indicators (59 terms)
python3 01_fetch_fred_data.py --all
```

** Warning:** This fetches ~59,000 series and takes a while!

## Customizing Predefined Categories

Edit `01_fetch_fred_data.py` lines 236-273 to change the `ECONOMIC_INDICATORS` dictionary:

```python
ECONOMIC_INDICATORS = {
    'core': [
        "GDP", "inflation", "unemployment", ...
    ],
    'your_category': [
        "your term 1",
        "your term 2",
    ]
}
```

## Examples

### Focused Dataset (Recommended for Testing)

Create `my_terms.txt`:
```
GDP
unemployment rate
CPI
federal funds rate
```

Fetch:
```bash
python3 01_fetch_fred_data.py --from-file my_terms.txt --limit 50
```

### Specific Topic

```bash
python3 01_fetch_fred_data.py --search "housing market" "real estate" "mortgage"
```

### Quick Test

```bash
# Just get GDP data
python3 01_fetch_fred_data.py --search "GDP" --limit 10
```

## Controlling API Usage

### Adjust Rate Limiting

```bash
# Slower (more polite to FRED API)
python3 01_fetch_fred_data.py --search "GDP" --delay 1.0

# Faster (may hit rate limits)
python3 01_fetch_fred_data.py --search "GDP" --delay 0.2
```

### Limit Results Per Term

```bash
# Get only 50 series per term instead of 1000
python3 01_fetch_fred_data.py --category core --limit 50
```

## Best Practices

1. **Start small**: Test with 1-5 terms before running --all
2. **Use categories**: Predefined categories are well-organized
3. **Custom file**: Best for reproducible research
4. **Set delay**: Use --delay 1.0 to be nice to FRED API
5. **Interrupt gracefully**: Ctrl+C will stop (already saved terms remain)

## Workflow Recommendation

### For Recommendations Project

Get focused, high-quality data:

```bash
# Create focused terms file
cat > focused_terms.txt << EOF
GDP
Real GDP
inflation
CPI
unemployment rate
employment
federal funds rate
10-year treasury
housing starts
retail sales
EOF

# Fetch (about 5-10 minutes)
python3 01_fetch_fred_data.py --from-file focused_terms.txt --limit 100 --delay 0.5

# Parse
python3 02_parse_fred_txt.py --input txt --output data --format parquet

# Now you have clean data for recommendations!
```

## Output

All fetched data goes to `txt/` directory:
```
txt/
├── gdp_series_20251029.txt
├── inflation_series_20251029.txt
└── ...
```

Parse them with:
```bash
python3 02_parse_fred_txt.py --input txt --output data --format parquet
```

## Troubleshooting

### "Too many results"
Use --limit to reduce results per term:
```bash
python3 01_fetch_fred_data.py --search "GDP" --limit 50
```

### "API rate limit"
Increase delay between requests:
```bash
python3 01_fetch_fred_data.py --search "GDP" --delay 2.0
```

### "Takes too long"
- Use fewer terms
- Use smaller --limit
- Use specific terms instead of --all

### "Duplicate data"
Delete old txt files before fetching:
```bash
rm txt/*.txt
python3 01_fetch_fred_data.py --from-file search_terms.txt
```
