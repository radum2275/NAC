# NAC gala scheduler

import os
import sys
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

ROLES = [
    "DTK",
    "PA",
    "TM1",
    "TM2",
    "MEDALS",
    "DESK",
    "LINEUP1",
    "LINEUP2",
    "RESULTS",
    "TK0",
    "TK1",
    "TK2",
    "TK3",
    "TK4",
    "TK5",
    "TK6",
    "TK7",
    "TK8",
    "TK9",
    "TJ0",
    "TJ1",
    "TJ2",
    "TJ3",
    "TJ4",
    "TJ5",
    "TJ6",
    "TJ7",
    "TJ8",
    "TJ9",
    "RAFFLE",
    "TAKEDOWN",
]

ROLE_WITH_EXPERIENCE = [
    "LINEUP1",
    "LINEUP2",
    "DESK"
]

NOVICE_SQUADS = [
    "Skills Squad", 
    "Development Squad"
]

EXPERIENCED_SQUADS = [
    "Age Group Squad", 
    "Development Squad Plus", 
    "Performance Pathway Squad", 
    "Performance Pathway Squad Plus",
    "Performance Squad"
]

# Set the random seed globally
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

# Load the availability file
def read_availability(filename: str):
    """
    Availability is given in a .csv file with the following columns:
        - Swimmer's First Name
        - Swimmer's Last Name
        - Availability per Session -- Session 1 AM, Session 2 PM
        - Team Manager? -- Yes, No
    """

    print(f"[NAC] Loading availability from: {filename}")
    df = pd.read_csv(filename)
    df = df.drop('Timestamp', axis=1)

    def create_athlete(row):
        last = row['Last']
        first = row['First']
        return last.strip() + ", " + first.strip()
    
    df['Athlete'] = df.apply(create_athlete, axis=1)
    return df

def read_signup(filename: str):
    """
    Event signup is given in a .csv file with the following columns:
        - Athlete -- format is Last, First
        - Birth Date
        - Billing Group
        - Squad
        - Location
        - Record Last Updated
        - Notes -- Y (default, schedule), N (do not schedule)
    """

    print(f"[NAC] Loading event signup from: {filename}")
    df = pd.read_csv(filename)
    df = df.drop(columns=['Birth Date', 'Billing Group', 'Location', 'Record Last Updated'])

    # Fix `Mc ABC` to `McABC`
    df['Athlete'] = df['Athlete'].apply(lambda x: x[:2] + x[3:] if x.startswith("Mc ") else x)

    return df

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
    return lines

TEAM_MANAGERS = [
    "Aoife Gennochi",
    "Teresa Martin",
    "Peter Walsh-Hussey",
    "Sinead Scully",
    "Andrea O'Doherty",
    "Orna Kiernan",
    "Eoin O'Donoghue",
    "Peter Walsh-Hussey",
    "Radu Marinescu",
    "Mags Connelly",
    "Paul O'Neill",
    "Teresa Martin",
    "Ana Melnicova",
    "Anne Lynam",
    "Michelle McCormack",
]

TM_TO_SWIMMER = {
    "Aoife Genocchi": ["Genocchi, Cillian", "Genocchi, Alicia"],
    "Teresa Martin": ["Martin, Kenny", "Martin, Maisie"],
    "Peter Hussey": ["Walsh-Hussey, Hannah"],
    "Sinead Scully": ["Scully, Emily"],
    "Andrea O'Doherty": ["O'Donoghue, Leah"],
    "Orna Kiernan": ["Byrne, Ribh", "Byrne, Fionn"],
    "Eoin O'Donoghue": ["O'Donoghue, Shane"],
    "Radu Marinescu": ["Marinescu, Ioachim", "Marinescu, Yohan"],
    "Mags Connelly": ["Driver, Seana"],
    "Paul O'Neill": ["O'Neill, Cian", "O'Neill, Daire"],
    "Ana Melnicova": ["Melnicova, Maria"],
    "Anne Lynam": ["Lynam, Daniel"],
    "Michelle McCormack": ["McCormack, Sophie"],
}

# Session 1 AM -- seeded with initial preferences
SESSION_AM = {
    "PA": "Karen Jennings",
    "TM1": "Aoife Genocchi",
    "TM2": "Peter Hussey",
    "RESULTS": "Radu Marinescu",
    "MEDALS": "Annabel Farrington-Knight",
    "TAKEDOWN": "xxxxxxxxxxxxxxxxx",
    "TJ0": "Rinah Ho",
}

# Session 2 AM -- seeded with initial preferences
SESSION_PM = {
    "PA": "Karen Jennings",
    "TM1": "Teresa Martin",
    "TM2": "Michelle McCormack",
    "RESULTS": "Radu Marinescu",
    "TK0": "Barbara Stanczak",
}

SESSION_CUTOFF = 15

def to_name(athlete: str):
    temp = athlete.split(',')
    return temp[1].strip() + " " + temp[0].strip()

def is_assignable(candidate: str, pool: set):
    if candidate in pool:
        return True # exact match
    else:
        for s in pool:
            if similarity(candidate, s) >= 0.9:
                return True
        return False

def has_sibling_assigned(candidate: str, pool):
    last_name = candidate.split(',')[0] # assumed `Last, First`
    for s in pool:
        if last_name in s:
            return True
    return False

def is_available_for_session(candidate: str, session: str, cutoff: int, entries: List[str]):
    """
    Check if a candidate as `Last, First` is available for a session (AM or PM).
    """
    assert session in ["am", "pm"], "Unknown session (am or pm)."

    for i, line in enumerate(entries):
        if not line.startswith("# "): # swimmer
            temp = line.split()[:-1]
            swimmer = " ".join(temp)
            if swimmer.startswith("Mc "):
                swimmer = swimmer[:2] + swimmer[3:]
            swimmer = string.capwords(swimmer)
            if similarity(swimmer, candidate) >= 0.9:
                j = i + 1
                stop = False
                while not stop:
                    event = entries[j]
                    assert event.startswith("# ")
                    ev = int(event.split()[1][:-1])
                    if session == "am" and ev <= cutoff:
                        return True
                    elif session == "pm" and ev > cutoff:
                        return True
                    j = j + 1
                    if j >= len(entries) or not entries[j].startswith("# "):
                        stop = True
    return False

def is_candidate_novice(candidate: str, signups: pd.DataFrame):
    """
    Check to see if a candidate `Last, First` is novice or not. Basically, his/her
    squad is either Skills or Development.
    """

    # Loop over all signed up candidates and check the squad
    signups = signups.reset_index() # make sure indexes pair with number of rows
    result = False
    for index, row in signups.iterrows():
        swimmer = row['Athlete']
        squad = row['Squad']
        if similarity(swimmer, candidate) > 0.9: # most likely swimmer found
            if squad in NOVICE_SQUADS:
                result = True
                break
    return result

def is_candidate_experienced(candidate: str, signups: pd.DataFrame):
    """
    Check to see if a candidate `Last, First` is novice or not. Basically, his/her
    squad is either Skills or Development.
    """

    # Loop over all signed up candidates and check the squad
    signups = signups.reset_index() # make sure indexes pair with number of rows
    result = False
    for index, row in signups.iterrows():
        swimmer = row['Athlete']
        squad = row['Squad']
        if similarity(swimmer, candidate) > 0.9: # most likely swimmer found
            if squad in EXPERIENCED_SQUADS:
                result = True
                break
    return result

def make_schedule(availability: pd.DataFrame, signups: pd.DataFrame, entries: List[str]):
    """
    Create the gala schedule by filling in the partially scheduled sessions.
    """  

    # Get all event signups that can be assigned (flag Notes is Y)
    mask = signups['Notes'] == "Y"
    signups_yes = pd.DataFrame(signups[mask])
    assignables = set([string.capwords(s) for s in signups_yes['Athlete'].to_list()])

    # Remove the swimmers associate with seeded TMs (if any)
    print(f"[NAC] Remove swimmer associated with TMs:")
    print(f"Assignables: {len(assignables)}")
    for tm in ["TM1", "TM2"]:
        if tm in SESSION_AM:
            for swimmer in TM_TO_SWIMMER[SESSION_AM[tm]]:
                try:
                    print(f"Removing swimmer: {string.capwords(swimmer)}")
                    assignables.remove(string.capwords(swimmer))
                except:
                    None
        if tm in SESSION_PM:
            for swimmer in TM_TO_SWIMMER[SESSION_PM[tm]]:
                try:
                    print(f"Removing swimmer: {string.capwords(swimmer)}")
                    assignables.remove(string.capwords(swimmer))
                except Exception:
                    None
    print(f"Assignables: {len(assignables)}")

    # Schedule Session 1 (AM). Assign first the roles marked as "Session 1 AM" in
    # the availability dataframe. If not all roles filled, pull from the the 
    # availability df those available for both sessions. Then pull from event
    # signup but filter out those only available in Session 2.
    mask_am = availability['Availability'] == "Session 1 AM"
    mask_pm = availability['Availability'] == "Session 2 PM"
    mask_both = availability['Availability'] == "Session 1 AM, Session 2 PM"
    avails_am = pd.DataFrame(availability[mask_am])
    avails_pm = pd.DataFrame(availability[mask_pm])
    avails_both = pd.DataFrame(availability[mask_both])

    print(f"[NAC] AM only availability: {avails_am.shape[0]}")
    print(f"[NAC] PM only availability: {avails_pm.shape[0]}")
    print(f"[NAC] AM+PM availability  : {avails_both.shape[0]}")
    print(f"AM only: {[string.capwords(s) for s in avails_am['Athlete'].to_list()]}")
    print(f"PM only: {[string.capwords(s) for s in avails_pm['Athlete'].to_list()]}")

    print(f"[NAC] Scheduling AM only roles")
    candidates = [string.capwords(s) for s in avails_am['Athlete'].to_list()]
    used_names = set()
    for val in list(SESSION_AM.values()):
        used_names.add(string.capwords(val))

    random.shuffle(candidates)
    print(f"Available candidates: {len(candidates)}")
    print(f"Assignable signups: {len(assignables)}")
    print(f"Roles to assign: {len(ROLES) - len(SESSION_AM)}")

    # Select first the candidates that expressed AM availability only
    for role in ROLES: # for each role
        if role not in SESSION_AM: # select a name from the AM candidates (availability)
            if role in ROLE_WITH_EXPERIENCE:
                for candidate in candidates: # formatted as `Last, First`
                    candidate_name = to_name(candidate) # change to `First Last
                    if candidate_name not in used_names \
                        and is_assignable(candidate, assignables) \
                        and is_candidate_experienced(candidate, signups):
                        
                        used_names.add(candidate_name)
                        SESSION_AM[role] = candidate_name
                        print(f"Role {role} assigned to {candidate}")
                        break
            else:
                for candidate in candidates: # formatted as `Last, First`
                    candidate_name = to_name(candidate) # change to `First Last
                    if candidate_name not in used_names \
                        and is_assignable(candidate, assignables):

                        used_names.add(candidate_name)
                        SESSION_AM[role] = candidate_name
                        print(f"Role {role} assigned to {candidate}")
                        break
        else:
            print(f"Role {role} already assigned to {SESSION_AM[role]}")
    
    # Check if all roles in SESSION_AM are filled
    used_names_ampm = set()
    if len(SESSION_AM) < len(ROLES):
        print(f"Roles remaining after AM availability: {len(ROLES) - len(SESSION_AM)}")
        candidates = [string.capwords(s) for s in avails_both['Athlete'].to_list()]
        print(f"Selecting candidates from AM+PM availability: {len(candidates)}")
        random.shuffle(candidates)
        for role in ROLES:
            if role not in SESSION_AM: # role is not assigned yet
                if role in ROLE_WITH_EXPERIENCE: # the role requires EXPERIENCE
                    for candidate in candidates: # formatted as `Last, First`
                        candidate_name = to_name(candidate) # change to `First Last`
                        if candidate_name not in used_names \
                            and is_assignable(candidate, assignables) \
                            and is_available_for_session(candidate, "am", SESSION_CUTOFF, entries) \
                            and is_candidate_experienced(candidate, signups):

                            used_names.add(candidate_name)
                            used_names_ampm.add(candidate_name)
                            SESSION_AM[role] = candidate_name
                            print(f"Role {role} assigned to {candidate}")
                            break
                else:
                    for candidate in candidates: # formatted as `Last, First`
                        candidate_name = to_name(candidate) # change to `First Last`
                        if candidate_name not in used_names \
                            and is_assignable(candidate, assignables) \
                            and is_available_for_session(candidate, "am", SESSION_CUTOFF, entries):

                            used_names.add(candidate_name)
                            used_names_ampm.add(candidate_name)
                            SESSION_AM[role] = candidate_name
                            print(f"Role {role} assigned to {candidate}")
                            break

    # Check if all roles in SESSION_AM are filled
    if len(SESSION_AM) < len(ROLES):
        print(f"Roles remaining after AM+PM availability: {len(ROLES) - len(SESSION_AM)}")
        print(f"Selecting candidates from the SIGNUPS pool")
        candidates = deepcopy(assignables)
        random.shuffle(candidates)
        for role in ROLES:
            if role not in SESSION_AM:
                if role in ROLE_WITH_EXPERIENCE:
                    for candidate in candidates:
                        if not has_sibling_assigned(candidate, used_names) \
                            and is_available_for_session(candidate, "am", SESSION_CUTOFF, entries) \
                            and is_candidate_experienced(candidate, signups):

                            candidate_name = to_name(candidate)
                            used_names.add(candidate_name)
                            SESSION_AM[role] = candidate_name
                            print(f"Role {role} assigned to {candidate}")
                            break
                else:
                    for candidate in candidates:
                        if not has_sibling_assigned(candidate, used_names) \
                            and is_available_for_session(candidate, "am", SESSION_CUTOFF, entries):

                            candidate_name = to_name(candidate)
                            used_names.add(candidate_name)
                            SESSION_AM[role] = candidate_name
                            print(f"Role {role} assigned to {candidate}")
                            break

    print(f"[NAC] AM session length: {len(SESSION_AM)}")
    print(f"[NAC] Number of roles: {len(ROLES)}")
    assert(len(SESSION_AM) == len(ROLES)), f"Not possible to schedule the AM session!"
    print("[NAC] AM roles assignment complete.")

    print(SESSION_AM)
    
    # Schedule Session 2 (PM). Assign first the roles marked as "Session 2 PM" in
    # the availability dataframe. If not all roles filled, pull from those that
    # are available for both sessions. Then, pull from the event signup but filter
    # out those available in Session 1.
    print(f"[NAC] Scheduling PM only roles")
    candidates = [string.capwords(s) for s in avails_pm['Athlete'].to_list()]
    for val in list(SESSION_PM.values()): # add the PM seeds to the used names
        used_names.add(string.capwords(val))

    random.shuffle(candidates)
    print(f"Available candidates: {len(candidates)}")
    print(f"Assignable signups: {len(assignables)}")
    print(f"Roles to assign: {len(ROLES) - len(SESSION_PM)}")

    # Select first the candidates that expressed PM availability only
    for role in ROLES:
        if role not in SESSION_PM: # select an unassigned candidated
            if role in ROLE_WITH_EXPERIENCE:
                for candidate in candidates:
                    candidate_name = to_name(candidate)
                    if candidate_name not in used_names \
                        and is_assignable(candidate, assignables) \
                        and is_candidate_experienced(candidate, signups):

                        used_names.add(candidate_name)
                        SESSION_PM[role] = candidate_name
                        print(f"Role {role} assigned to {candidate}")
                        break
            else:
                for candidate in candidates:
                    candidate_name = to_name(candidate)
                    if candidate_name not in used_names \
                        and is_assignable(candidate, assignables):
                        
                        used_names.add(candidate_name)
                        SESSION_PM[role] = candidate_name
                        print(f"Role {role} assigned to {candidate}")
                        break
        else:
            print(f"Role {role} already assigned to {SESSION_PM[role]}")

    # Check if all roles in SESSION_PM are filled
    if len(SESSION_PM) < len(ROLES):
        print(f"Roles remaining after PM availability: {len(ROLES) - len(SESSION_PM)}")
        all_pms = set([string.capwords(s) for s in avails_both['Athlete'].to_list()])
        candidates = all_pms.difference(used_names_ampm)
        candidates = list(candidates)
        print(f"Selecting candidates from the remaining AM+PM availability: {len(candidates)}")
        random.shuffle(candidates)
        for role in ROLES:
            if role not in SESSION_PM:
                if role in ROLE_WITH_EXPERIENCE:
                    for candidate in candidates:
                        candidate_name = to_name(candidate)
                        if candidate_name not in used_names \
                            and is_assignable(candidate, assignables) \
                            and is_available_for_session(candidate, "pm", SESSION_CUTOFF, entries) \
                            and is_candidate_experienced(candidate, signups):

                            used_names.add(candidate_name)
                            SESSION_PM[role] = candidate_name
                            print(f"Role {role} assigned to {candidate}")
                            break
                else:
                    for candidate in candidates:
                        candidate_name = to_name(candidate)
                        if candidate_name not in used_names \
                            and is_assignable(candidate, assignables) \
                            and is_available_for_session(candidate, "pm", SESSION_CUTOFF, entries):

                            used_names.add(candidate_name)
                            SESSION_PM[role] = candidate_name
                            print(f"Role {role} assigned to {candidate}")
                            break


    # Check if all roles in SESSION_AM are filled
    if len(SESSION_PM) < len(ROLES):
        print(f"Roles remaining after AM+PM availability: {len(ROLES) - len(SESSION_PM)}")
        print(f"Selecting candidates from the SIGNUPS pool")
        candidates = list(deepcopy(assignables))
        random.shuffle(candidates)
        for role in ROLES:
            if role not in SESSION_PM:
                if role in ROLE_WITH_EXPERIENCE:
                    for candidate in candidates:
                        if not has_sibling_assigned(candidate, used_names) \
                            and is_available_for_session(candidate, "pm", SESSION_CUTOFF, entries) \
                            and is_candidate_experienced(candidate, signups):

                            candidate_name = to_name(candidate)
                            used_names.add(candidate_name)
                            SESSION_PM[role] = candidate_name
                            print(f"Role {role} assigned to {candidate}")
                            break
                else:
                    for candidate in candidates:
                        if not has_sibling_assigned(candidate, used_names) \
                            and is_available_for_session(candidate, "pm", SESSION_CUTOFF, entries):

                            candidate_name = to_name(candidate)
                            used_names.add(candidate_name)
                            SESSION_PM[role] = candidate_name
                            print(f"Role {role} assigned to {candidate}")
                            break


    assert(len(SESSION_PM) == len(ROLES)), "Not possible to schedule the PM session!"
    print("[NAC] PM roles assignment complete.")

    print(SESSION_PM)

if __name__ == "__main__":

    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--availability_file', 
        type=str, 
        default=None, 
        help="Path to the availability file (CSV)."
    )
    
    parser.add_argument(
        '--signup_file', 
        type=str, 
        default=None, 
        help="Path to the event signup file (CSV)."
    )

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

    # Read the input files
    df_avails = read_availability(args.availability_file)
    df_signup = read_signup(args.signup_file)
    entries = read_entries(args.entry_file)

    make_schedule(availability=df_avails, signups=df_signup, entries=entries)

    print("===" * 20)
    print("[SESSION AM]")
    for role in ROLES:
        print(f"{role:>10}: {SESSION_AM[role]}")    
    print("===" * 20)
    print("[SESSION PM]")
    for role in ROLES:
        print(f"{role:>10}: {SESSION_PM[role]}")    

    print("Done.")
