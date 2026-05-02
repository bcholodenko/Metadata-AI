#!/bin/bash

for file in *.[jJ][pP][gG]; do
    [ -e "$file" ] || continue

    # Extract year, month, day
    if [[ $file =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2}) ]]; then
        YEAR="${BASH_REMATCH[1]}"
        MONTH="${BASH_REMATCH[2]}"
        DAY="${BASH_REMATCH[3]}"
        
        # ExifTool format: YYYY:MM:DD HH:MM:SS
        EXIF_DATE="$YEAR:$MONTH:$DAY 12:00:00"

        echo "Updating: $file to $EXIF_DATE and clearing comments"
        
        # -Comment= resets the standard JPEG comment
        # -UserComment= resets the EXIF-specific user comment
        exiftool "-AllDates=$EXIF_DATE" \
                 "-FileCreateDate=$EXIF_DATE" \
                 "-FileModifyDate=$EXIF_DATE" \
                 "-Comment=" \
                 "-UserComment=" \
                 -overwrite_original "$file"
                 
    else
        echo "Skipping $file: Date pattern not found."
    fi
done
