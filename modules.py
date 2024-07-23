
import pandas as pd
import regex as re
import json
from keys import *
import itertools
import pickle
with open(Keys.REMOVE_WORDS, "r") as f:
    remove_words = json.load(f)
with open (Keys.MALWARE_LIST, "r") as f:
    malwares = json.load(f)

with open(Keys.FIXING_PATTERN, "r", encoding="utf-8") as f:
        common_fixing_pattern = json.load(f)

# create mapper
with open(r"data/meta data/proID_techID.json", "r") as f:
    proID_techID = json.load(f)
    
with open(r"data/meta data/tech_tac_mapper.json","r") as f:
    tech_tac_mapper = json.load(f)
with open(r"data/meta data/tactic_combinations.json", "r") as f:
    tactic_combinations = json.load(f)
# strong_verb_group = ["delete"]
with open(r"data/meta data/verb_similarity.json", "r") as f:
    verb_data = json.load(f)
    verb_similarity = verb_data["group"]
    imporant_verbs = verb_data["important"]

def check_if_verb_is_strong(verbs, strong_verb_group = Keys.STRONG_VERB_GROUP):
    for k,v in verb_similarity.items():
        if k not in strong_verb_group:
            continue
        set_v = set(v)
        set_v1 = set(verbs)
        lists1 = list(set_v1.intersection(set_v ))
        if len(lists1) > 0:
            return True
    return False
def check_strong_verb_similarity_mismatch(verbs1, verbs2, strong_verb_group = Keys.STRONG_VERB_GROUP):
    if len(verbs1) == 0 or len(verbs2) == 0:
        return False
    set_v1 = set(verbs1)
    set_v2 = set(verbs2)
    for k,v in verb_similarity.items():
        if k not in strong_verb_group:
            continue
        set_v = set(v)
        lists1 = list(set_v1.intersection(set_v ))
        lists2 = list(set_v2.intersection(set_v ))
        if len(lists1) > 0 and len(lists2) == 0:
            return True # this is the key, verb is very strong and it is not in the second list
        if len(lists2) > 0 and len(lists1) == 0:
            return True
    return False
def check_verbs_similarity(verbs1:list, verbs2:list):
    if len(verbs1) == 0 or len(verbs2) == 0:
        return False
    set_v1 = set(verbs1)
    set_v2 = set(verbs2)
    if len(set_v1.intersection(set_v2)) > 0:
        return True
    for k,v in verb_similarity.items():
        set_v = set(v)
        lists1 = list(set_v1.intersection(set_v ))
        lists2 = list(set_v2.intersection(set_v ))
        if len(lists1) > 0 and len(lists2) > 0:
            return True
    return False
def check_verb_similarity(verb1, verb2):
    if verb1 == verb2:
        return True
    for k,v in verb_similarity.items():
         if verb1 in v and verb2 in v:
             return True
    return False
heuristic_tactic_combinations = []
for c in tactic_combinations:
    source = c["first"]["id"]
    dest = c["second"]["id"]
    status = c["status"]
    if status == 0:
        id1 = source + "__" + dest
        id2 = dest + "__" + source
        heuristic_tactic_combinations.append(id1)
        heuristic_tactic_combinations.append(id2)
    if status == 1:
        id1 = source + "__" + dest
        heuristic_tactic_combinations.append(id1)
    if status == 2:
        id2 = dest + "__" + source
        heuristic_tactic_combinations.append(id2)


def recognize_platform(text):
    pattern = r"\[E1\].*?\[\/E1\]"
    match = re.search(pattern, text)
    if match:
        return match.group(0).replace("[E1]", "").replace("[/E1]", "").strip()

def add_association(source_id, dest_id, association_data):
    new_id = source_id + "__" + dest_id
    if new_id not in association_data:
                association_data[new_id] = {}
                association_data[new_id]["source"] = source_id
                association_data[new_id]["target"] = dest_id
                association_data[new_id]["count"] = 1
    else:
                association_data[new_id]["count"] += 1
def tactic_2_tech(tactic):
    techs = []
    for k,v in tech_tac_mapper.items():
        if tactic in v:
            techs.append(k)
    return techs
def read_pre_association(file_name = Keys.PRE_ASSOCIATION_FILE):
    association_data = {}
    df = pd.read_excel(file_name)
    for i, row in df.iterrows():
        if i == 0:
            continue
        source_id = row["id"]
        text = row["text"]
        status = row["type"]
        dest_id = recognize_platform(text)
        if dest_id.startswith("TA"):
             dests = tactic_2_tech(dest_id)
        else:
            dests = [dest_id]
        for dest_id in dests:
            if source_id == dest_id or status == 3:
                continue
            if status == 1:
                add_association(source_id, dest_id, association_data)
            if status == 0:
                add_association(dest_id, source_id, association_data)
            if status == 2: #bidirectional
                add_association(source_id, dest_id, association_data)
                add_association(dest_id, source_id, association_data)
    return association_data

def genereate_tatics_combination(association_data):
    tactics_combination = []
    for k, v in association_data.items():
        source = v["source"]
        target = v["target"]
        id_ = k
        if source in tech_tac_mapper:
            source_tactics = tech_tac_mapper[source]
        else:
            if source == "T1086":
                source = "T1059.001"
        if not target.startswith("TA"):
            target_tactics = tech_tac_mapper[target]
            products = itertools.product(source_tactics, target_tactics)
        else:
            products = itertools.product(source_tactics, [target])
        for p in products:
            source_tactic = p[0]
            target_tactic = p[1]
            if source_tactic == target_tactic:
                continue
            tactic_combied_id = source_tactic + "__" + target_tactic
            tactics_combination.append(tactic_combied_id)
    return list(set(tactics_combination))
pre_association = read_pre_association()
# tatic_combined = genereate_tatics_combination(pre_association)


def refine_similar_procedure(ratio = 0.90):
    similar_mapper = {}
    with open (Keys.SIMILAR_PROCEDURE_FILE, "r") as f:
        similar_procedures = json.load(f)
    for s in similar_procedures:
        if s["f1"] < ratio:
              continue
        keys = [k for k in s.keys() if k.startswith("relationship")]
        source = keys[0]
        dest = keys[1]
        if source not in similar_mapper:
            similar_mapper[source] = {}
        similar_mapper[source][dest] = s["f1"]
        if dest not in similar_mapper:
            similar_mapper[dest] = {}
        similar_mapper[dest][source] = s["f1"]
    return similar_mapper
similar_mapper = refine_similar_procedure()
