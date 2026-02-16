# Preprocessing script for Leinster galas

import os
import sys
import json
import string
import random
import argparse
from typing import List
import pandas as pd
import numpy as np

from pypdf import PdfReader

from difflib import SequenceMatcher
from copy import deepcopy

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def read_entries(filename: str):

    print(f"[NAC] Loading entry report from: {filename}")
    reader = PdfReader(filename)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]

    entries = ""
    for page in pdf_texts:
        entries += page + "\n"
    lines = entries.split('\n')
    n = len(lines)
    lines = lines[5:n-6] # remove preamble and summary
    print(f"[NAC] Found {len(lines)} lines in the entry report.")

    return lines

# Set the random seed globally
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

SESSIONS = {
    "S1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "S2": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
}


# SESSIONS = {
#     "S1": [1, 2, 3, 4, 5],
#     "S3": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#     "S5": [22, 23, 25, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
# }

# SESSIONS = {
#     "S1": [1, 2, 3, 4, 5, 6],
#     "S2": [7],
#     "S3": [8, 9, 10, 11, 12],
#     "S4": [13, 14, 15, 16, 17, 18, 19, 20],
#     "S5": [21, 22],
#     "S6": [23, 24, 25, 26, 27, 28],
#     "S7": [29, 30, 31, 32, 33, 34, 35],
#     "S8": [36],
#     "S9": [37, 38, 39, 40, 41, 42],
# }


def ends_with_letter(s):
    if len(s) == 0:
        return False
    return s[-1].isalpha()

def swimmers_per_session(session_events: List[int], entries: List[str]):
    """
    Parse the entry report and find the swimmers available in a given session.
    A session is identified by `start` and `end` events.
    """

    available_swimmers = []
    for i, line in enumerate(entries):
        if not line.startswith("# "): # swimmer
            temp = line.split(',')
            last_name = temp[0]
            first_names = temp[1].split()
            swimmer = ", ".join([last_name, first_names[0]])
            if swimmer.startswith("Mc "):
                swimmer = swimmer[:2] + swimmer[3:]
            swimmer = string.capwords(swimmer)

            # Look at the entries
            j = i + 1
            stop = False
            while not stop:
                event = entries[j]
                assert event.startswith("# ")
                ev_name = event.split()[1]
                if ends_with_letter(ev_name):
                    ev = int(ev_name[:-1])
                else:
                    ev = int(ev_name)
                if ev in session_events:
                    available_swimmers.append(swimmer)
                    stop = True
                j = j + 1
                if j >= len(entries) or not entries[j].startswith("# "):
                    stop = True

    return available_swimmers

if __name__ == "__main__":

    set_seed(442)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--entry_file', 
        type=str, 
        default=None, 
        help="Path to the entry report file (PDF)."
    )

    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None, 
        help="The output directory where the results will be stored."
    )

    # Parse the command line arguments
    args = parser.parse_args()

    entries = read_entries(args.entry_file)

    for s in sorted(SESSIONS.keys()):
        session_events = SESSIONS[s]
        print(f"[NAC] SESSION {s} (events: {session_events})")
        swimmers = swimmers_per_session(session_events, entries)
        # random.shuffle(swimmers)
        print(f"[NAC] Available swimmers: {len(swimmers)}")
        for swimmer in swimmers:
            print(f"   - {swimmer}")

    print("Done.")
