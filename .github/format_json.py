import json
import sys


def sort_innermost(d):
    """
    Recursively sorts ONLY innermost dictionaries.

    Innermost dictionary sorting rules:
      1. "SM" always first (if present)
      2. Keys WITHOUT "*" next (alphabetical)
      3. Keys WITH "*" last (alphabetical)
    """
    if isinstance(d, dict):
        # Recursively process values first
        new_dict = {k: sort_innermost(v) for k, v in d.items()}

        # Check if this dictionary contains nested dicts
        contains_dict = any(isinstance(v, dict) for v in new_dict.values())

        if not contains_dict:
            keys = list(new_dict.keys())

            def key_priority(k):
                # Priority tuple:
                # 1. "SM" → priority 0
                # 2. No "*" → priority 1
                # 3. Contains "*" → priority 2
                if k == "SM":
                    return (0, k)
                if "*" in k:
                    return (2, k)
                return (1, k)

            keys_sorted = sorted(keys, key=key_priority)

            return {k: new_dict[k] for k in keys_sorted}

        return new_dict

    # The recursion only visits lists to process any dictionaries they contain, it doesn't sort list themselves!
    elif isinstance(d, list):
        return [sort_innermost(i) for i in d]

    return d


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

    # Apply custom sort rule
    data = sort_innermost(data)

    # Write formatted output (NO sort_keys here!)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, separators=(",", ": "))
        f.write("\n")

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
