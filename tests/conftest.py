import sys
from pathlib import Path

# Let tests import the rag package without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
