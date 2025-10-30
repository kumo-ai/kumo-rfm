#!/bin/bash
# Consolidation plan for chart directories

echo "[Plan] Chart Directory Consolidation Plan"
echo "======================================"
echo ""

# Create unified output directory
OUTPUT_DIR="outputs/visualizations"
mkdir -p "$OUTPUT_DIR"

echo "Creating unified directory: $OUTPUT_DIR"
echo ""

# Analysis of duplicates
echo "DUPLICATE FILES (same content):"
echo "  - category_distribution (in my_charts + visualizations)"
echo "  - frequency_analysis (in my_charts + visualizations)"
echo "  - popularity_distribution (in my_charts + visualizations)"
echo "  - rfm_segmentation (in my_charts + visualizations)"
echo "  - rfm_segments.csv (in my_charts + visualizations)"
echo ""

echo "UNIQUE FILES:"
echo "  - relationship_graph (only in charts/) - NEW 7.5MB NetworkX graph"
echo "  - category_network (only in visualizations/) - 4.5MB"
echo "  - centrality_analysis (only in visualizations/)"
echo "  - community_detection (only in visualizations/)"
echo "  - series_network (only in visualizations/)"
echo ""

echo "RECOMMENDATION:"
echo "  1. Keep LATEST version of duplicates (from visualizations/)"
echo "  2. Move all unique files to $OUTPUT_DIR"
echo "  3. Delete old directories: my_charts/, charts/, visualizations/"
echo ""

read -p "Proceed with consolidation? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Moving files..."
    
    # Copy latest visualizations (most complete set)
    cp visualizations/*.png "$OUTPUT_DIR/"
    cp visualizations/*.csv "$OUTPUT_DIR/"
    
    # Copy unique relationship graph from charts/
    cp charts/relationship_graph_*.png "$OUTPUT_DIR/"
    
    echo "[OK] Files moved to $OUTPUT_DIR"
    echo ""
    
    read -p "Delete old directories? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf my_charts/ charts/ visualizations/
        echo "[OK] Old directories removed"
        echo ""
        echo "[Info] All visualizations now in: $OUTPUT_DIR"
        ls -lh "$OUTPUT_DIR"
    else
        echo "Old directories kept for safety"
    fi
else
    echo "Consolidation cancelled"
fi
