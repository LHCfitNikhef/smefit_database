from pathlib import Path
import sys

# add the path of the root, so that we can import python modules in the tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
