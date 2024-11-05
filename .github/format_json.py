import json
import sys


def format_json_file(file_path):
    """
    Formats a JSON file to ensure consistent indentation and readability.

    This function reads a JSON file, checks if it is valid JSON, and then
    reformats the file with 2-space indentation and each array element on a
    separate line for better readability. If the JSON is invalid, an error message
    is printed, and the function returns False.

    Parameters:
    -----------
    file_path : str
        The path to the JSON file that needs to be formatted.

    Returns:
    --------
    bool
        Returns True if the JSON file was formatted successfully.
        Returns False if the file contains invalid JSON.

    Raises:
    -------
    json.JSONDecodeError
        Raised when the JSON file is invalid, though it is handled within the function.
    """
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: {file_path} is not valid JSON. {e}")
            return False

    # Reformat the data by ensuring arrays have their elements on separate lines
    formatted_data = json.dumps(data, indent=2, separators=(",", ": "))

    # Write the reformatted JSON back to the file
    with open(file_path, "w") as f:
        f.write(formatted_data + "\n")  # Adding a newline at the end of the file

    print(f"Formatted {file_path} successfully.")
    return True


if __name__ == "__main__":
    """
    Main function that processes command-line arguments.

    This block allows the script to be run directly from the command line. It
    iterates over each argument passed to the script, checking if the file has a
    .json extension. If so, it attempts to format the JSON file. If the file does
    not have a .json extension, a message is printed indicating it is skipped.

    Usage:
    ------
    python script_name.py <file1.json> <file2.json> ...

    Example:
    --------
    $ python format_json.py data1.json data2.json
    """
    # Loop over all files passed as arguments
    for file in sys.argv[1:]:
        if file.endswith(".json"):
            format_json_file(file)
        else:
            print(f"Skipping non-JSON file: {file}")
