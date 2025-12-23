import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aart_func import *
from params import *

if __name__ == "__main__":
    print("Computing the lensing bands")
    lb.lb()
