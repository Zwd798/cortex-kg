import argparse

def main():
    parser = argparse.ArgumentParser(description='Create a knowledge graph from raw text')
    parser.add_argument('--version', action='version', version='0.1')
    args = parser.parse_args()
    print(args)

