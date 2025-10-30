#!/bin/bash

echo "🧹 Final Cleanup - Removing Old Directories"
echo "==========================================="
echo ""

echo "Current state:"
ls -ld my_charts charts visualizations 2>/dev/null || echo "Some directories already removed"
echo ""

echo "All files are now consolidated in: outputs/visualizations/"
echo "Total size: ~17MB (9 PNG files + 3 CSV files)"
echo ""

read -p "Remove old directories (my_charts, charts, visualizations)? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf my_charts/ charts/ visualizations/
    echo "✓ Old directories removed"
    echo ""
    echo "📁 Final structure:"
    tree -L 2 outputs/ 2>/dev/null || ls -lh outputs/visualizations/
else
    echo "Directories kept"
fi
