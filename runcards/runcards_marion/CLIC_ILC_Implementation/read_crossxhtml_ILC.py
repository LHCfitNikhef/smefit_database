#### Code which takes a input 3 html files and outputs the coefficients, the corss-section and uncertainties and the polarisation of the run
### Here for the fourth column "1" mean 0.8 for polarisation of electron beam and "2" means -0.8

import re

# List of HTML files to read
html_files = [
    "/home/marion/eeWW_SM_ILC/crossx.html",
    "/home/marion/eeWW_NP2_ILC/crossx.html",
    "/home/marion/eeWW_NP4_ILC/crossx.html",
]

# Open the output file for writing
with open("ILC500_mg5num_WW.txt", "w", encoding="utf-8") as output_file:
    # Regular expression pattern to find the label and the two numbers
    pattern = r'href="./HTML/([^_]+(?:_[^_]+)*)_ILC_500GeV[^"]*"> ([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?) <font face=symbol>&#177;</font> ([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'

    # Process each HTML file
    for html_file in html_files:
        # Open the current HTML file
        with open(html_file, "r", encoding="utf-8") as source_file:
            # Read each line in the current source file
            for line in source_file:
                # Check for the presence of "pos80" or "neg80" and set the fourth column value
                if "pos80" in line:
                    fourth_column = "1"
                elif "neg80" in line:
                    fourth_column = "2"
                else:
                    fourth_column = ""  # Empty if neither is present

                # Check if "_sq" is present in the line
                has_sq = "_sq" in line

                # Search for the specific pattern in the line
                match = re.search(pattern, line)
                if match:
                    # Extract the label and the two numbers
                    label = match.group(1)
                    number1 = match.group(2)
                    number2 = match.group(3)

                    # Process the label based on conditions
                    if (
                        has_sq and "_" not in label
                    ):  # Single segment and "_sq" is present
                        modified_label = f"{label}^2"
                    else:
                        # Replace underscores with asterisks if there are multiple segments
                        modified_label = label.replace("_", "*")

                    # Write the modified label, numbers, and fourth column to the output file
                    output_file.write(
                        f"{modified_label}\t{number1}\t{number2}\t{fourth_column}\n"
                    )
