#!/bin/bash

echo ""
echo "Running evaluation.py..."
python evaluation.py > output_files/evaluation.txt
if [ $? -eq 0 ]; then
    echo "evaluation.py completed successfully."
else
    echo "evaluation.py failed."
    exit 1 
fi

echo "Running statistics.py..."
python statistics.py > output_files/statistics.txt
if [ $? -eq 0 ]; then
    echo "statistics.py completed successfully."
else
    echo "statistics.py failed."
    exit 1 
fi


echo "All Python programs have been executed."
exit 0 