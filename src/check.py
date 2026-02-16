import sys

import numpy as np
import pandas as pd
import sklearn


def main() -> None:
    print("Python:", sys.version)
    print("numpy:", np.__version__)
    print("pandas:", pd.__version__)
    print("sklearn:", sklearn.__version__)


if __name__ == "__main__":
    main()