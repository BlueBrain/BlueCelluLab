"""Test script 2"""

import sys
sys.path.append("/home/vangeit/local/bglibpy/lib/python2.7/dist-packages")

import bglibpy


def main():
    """Main"""
    cell = bglibpy.Cell("", "")
    print cell

if __name__ == "__main__":
    main()
