#!/bin/bash

echo "📅 Organizing Visualizations by Date"
echo "======================================"
echo ""

VIZ_DIR="outputs/visualizations"

# Extract dates from filenames and organize
for file in "$VIZ_DIR"/*.{png,csv}; do
    [ -e "$file" ] || continue
    
    filename=$(basename "$file")
    
    # Extract date from filename (format: YYYYMMDD)
    if [[ $filename =~ ([0-9]{8})_ ]]; then
        date_str="${BASH_REMATCH[1]}"
        # Convert to YYYY-MM-DD format
        year="${date_str:0:4}"
        month="${date_str:4:2}"
        day="${date_str:6:2}"
        date_folder="${year}-${month}-${day}"
        
        # Create date folder
        mkdir -p "$VIZ_DIR/$date_folder"
        
        # Move file
        mv "$file" "$VIZ_DIR/$date_folder/"
        echo "✓ Moved: $filename → $date_folder/"
    else
        echo "⚠ Skipped (no date): $filename"
    fi
done

echo ""
echo "📁 Current structure:"
tree -L 2 "$VIZ_DIR" 2>/dev/null || {
    echo "$VIZ_DIR/"
    for dir in "$VIZ_DIR"/*/; do
        [ -d "$dir" ] && echo "  $(basename "$dir")/"
        ls -1 "$dir" 2>/dev/null | sed 's/^/    /'
    done
}

echo ""
echo "✅ Organization complete!"
