# Quick Reference: Date-Based Visualizations

## 📅 How It Works

All visualizations are automatically organized by date:

```
outputs/visualizations/
├── 2025-10-29/    ← Today's visualizations
├── 2025-10-30/    ← Tomorrow's visualizations  
└── 2025-10-31/    ← Next day's visualizations
```

## 🎨 Generating New Visualizations

### Option 1: NetworkX Graph (Recommended)
```bash
python 05_kumo_rfm_integration.py --visualize-graph
```
**Output:** `outputs/visualizations/YYYY-MM-DD/relationship_graph_YYYYMMDD_HHMMSS.png`

### Option 2: With Recommendations
```bash
python 05_kumo_rfm_integration.py --recommend "housing" --visualize-graph
```
**Output:** Graph + recommendations

### Option 3: Save to Database + Visualize
```bash
python 05_kumo_rfm_integration.py --visualize-graph --save-to-db
```
**Output:** Graph + database tables

## 📂 Finding Your Visualizations

### Latest visualizations (today):
```bash
ls -lh outputs/visualizations/$(date +%Y-%m-%d)/
```

### Specific date:
```bash
ls -lh outputs/visualizations/2025-10-29/
```

### All dates:
```bash
tree outputs/visualizations/
# or
ls -lh outputs/visualizations/
```

## 🔍 View Specific File Types

### Only PNG files (graphs):
```bash
find outputs/visualizations/ -name "*.png" -ls
```

### Only CSV files (data):
```bash
find outputs/visualizations/ -name "*.csv" -ls
```

### Specific visualization type:
```bash
find outputs/visualizations/ -name "relationship_graph_*.png"
find outputs/visualizations/ -name "category_network_*.png"
```

## 📊 Compare Across Dates

### Size changes over time:
```bash
find outputs/visualizations/ -name "relationship_graph_*.png" -exec ls -lh {} \;
```

### Count files per date:
```bash
for dir in outputs/visualizations/*/; do
    echo "$(basename $dir): $(ls $dir | wc -l) files"
done
```

## 🧹 Cleanup Old Visualizations

### Keep only last 7 days:
```bash
find outputs/visualizations/ -type d -mtime +7 -exec rm -rf {} \;
```

### Keep only last 30 days:
```bash
find outputs/visualizations/ -type d -mtime +30 -exec rm -rf {} \;
```

### Remove specific date:
```bash
rm -rf outputs/visualizations/2025-10-29/
```

## 📈 File Naming Convention

All files include both date and time:

```
relationship_graph_20251029_172820.png
                   ^^^^^^^^ ^^^^^^
                   DATE     TIME
                   
Format: {name}_{YYYYMMDD}_{HHMMSS}.{ext}
```

**Benefits:**
- ✅ Multiple runs per day don't conflict
- ✅ Easy to sort chronologically
- ✅ Grouped by date in subdirectories

## 🎯 Common Tasks

### 1. View today's latest graph:
```bash
open $(ls -t outputs/visualizations/$(date +%Y-%m-%d)/relationship_graph_*.png | head -1)
# or for Linux:
xdg-open $(ls -t outputs/visualizations/$(date +%Y-%m-%d)/relationship_graph_*.png | head -1)
```

### 2. Compare two dates:
```bash
ls outputs/visualizations/2025-10-29/
ls outputs/visualizations/2025-10-30/
diff <(ls outputs/visualizations/2025-10-29/) <(ls outputs/visualizations/2025-10-30/)
```

### 3. Get total storage:
```bash
du -sh outputs/visualizations/
du -sh outputs/visualizations/*/
```

### 4. Archive old visualizations:
```bash
tar -czf visualizations_backup_$(date +%Y%m%d).tar.gz outputs/visualizations/
```

## 💡 Tips

1. **Disk space**: Large graphs (7.5MB each) add up quickly. Clean old ones periodically.

2. **Git**: Add to `.gitignore` if you don't want to commit visualizations:
   ```
   outputs/visualizations/*/*.png
   outputs/visualizations/*/*.csv
   ```

3. **Automation**: Set up cron job to generate daily visualizations:
   ```bash
   0 9 * * * cd /path/to/project && python 05_kumo_rfm_integration.py --visualize-graph
   ```

4. **Comparison**: Keep one "baseline" visualization for comparison and delete intermediate ones.

## 🚀 Quick Commands Cheatsheet

```bash
# Generate graph
python 05_kumo_rfm_integration.py --visualize-graph

# View today's files
ls outputs/visualizations/$(date +%Y-%m-%d)/

# Count total files
find outputs/visualizations/ -type f | wc -l

# Total storage
du -sh outputs/visualizations/

# Latest graph
ls -t outputs/visualizations/*/relationship_graph_*.png | head -1

# Clean old files (7+ days)
find outputs/visualizations/ -type d -mtime +7 -exec rm -rf {} \;
```
