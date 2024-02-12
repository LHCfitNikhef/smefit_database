# bin/bash

DATABASE="/data/theorie/jthoeve/smefit_database/commondata"
OUTPUT="output.yaml"

for item in "$DATABASE"/*; do
    # Do something with each item
    filename=$(basename "$item")
    trimmed_filename="${filename::-5}"

    # Print the modified file name
    echo "- $trimmed_filename" >> "$OUTPUT"

done