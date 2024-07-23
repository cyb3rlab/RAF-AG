file1= r"/Users/khangmai/Library/CloudStorage/GoogleDrive-khangmt@uit.edu.vn/Other computers/My Laptop/Doctoral program/Main/Paper/RAF-AG/2024 May 04 work/evaluation.xlsx"
file2 = "matching_threshold_assing.xlsx"
file3 = "evaluated_result.xlsx"
import math
import json
import pandas as pd
import statistics
import jsonlines
def find_085(_evaluate_file:str, _threshold_file:str):
    df = pd.read_excel(_evaluate_file, sheet_name="RAS")
    threshold = pd.read_excel(_threshold_file, sheet_name="Sheet1")
    RAF = {}
    for index, row in df.iterrows():
        id_ = row["name"]
        f1 = row["f1"]
        num_attack = row["TP"] + row["FN"]
        duplicate = row["duplicate"]
        if pd.isnull(id_):
            continue
        RAF[id_] = {"f1": f1, "num_attack": num_attack, "duplicate": duplicate}
    columns = threshold.columns
    data2 = {}
    for c in columns:
        if c == "Unnamed: 0":
            continue
        data = []
        read_data = threshold[c].to_dict()
        for k,v in read_data.items():
            v = v.replace("(","").replace(")","").replace(" ","")
            splits = v.split(",")
            data.append((float(splits[0].strip()), int(splits[1].strip())))
        data2[c] = data
        print()
    #finding optimal threshold for each report
    for k,v in data2.items():
        meta = RAF[k]
        attacks = meta["num_attack"]
        distances = [abs(m[1]-attacks) for m in v]
        min_distance = min(distances)
        indexs= [i for i, d in enumerate(distances) if d == min_distance]
        if len(indexs) == 1:
            RAF[k]["optimal_threshold"] = v[indexs[0]][0]
        else:
            temp = []
            for i in indexs:
                temp.append(v[i][0])
            RAF[k]["optimal_threshold"] = statistics.mean(temp)
        print()
    total_f1 = 0
    total_weighted = 0
    for k,v in RAF.items():
        f1 = RAF[k]["f1"]
        total_f1 += f1
        total_weighted += f1*v["optimal_threshold"]
    calculated_dict = {}
    calculated_dict["optimal_threshold_all"] = round(total_weighted/total_f1,3)
    #calculated_duplicate_average


    for k,v in data2.items():
        data_ = abs(v[0][1] - v[-1][1])/2
        RAF[k]["ratio"] = data_
    print()
    avarge_ratio = sum([v["ratio"]*v["f1"] for k,v in RAF.items()])/total_f1
    average_duplicate = sum([v["duplicate"]*v["f1"] for k,v in RAF.items()])/total_f1
    tolerate_threshold = average_duplicate/avarge_ratio * 0.1
    calculated_dict["tolerate_threshold"] = round(tolerate_threshold,3)
    calculated_dict["average_duplicate"] = round(average_duplicate,3)
    calculated_dict["average_ratio"] = round(avarge_ratio,3)
    with open("calculated.json", "w") as f:
        json.dump(calculated_dict, f, indent=4)
    saved_data = []
    for k,v in RAF.items():
        v["name"] = k
        saved_data.append(v)

    pd.DataFrame(saved_data).to_excel("RAF.xlsx")
    print()

# find_085(file3, file2)
from mitre_attack import MitreAttack

import json
def generate_procedure_file(saved_file):

    procedures = list(MitreAttack.get_procedures())
    df = pd.DataFrame(procedures)
    df.to_csv(saved_file, index=False)

# generate_procedure_file(r"data/procedure/input/procedures.csv")

def get_list_of_malware():
    malwares = MitreAttack.mitre_attack_data.get_software()
    data =[]
    for m in malwares:
        if m.id.startswith("malware"):
            if hasattr(m, "x_mitre_aliases"):
                data.extend(m.x_mitre_aliases)
            data.append(m.name)
    return list(set(data))


def get_list_of_tools():
    tools = MitreAttack.mitre_attack_data.get_software()
    data =[]
    for t in tools:
        if t.id.startswith("tool"):
            if hasattr(t, "x_mitre_aliases"):
                data.extend(t.x_mitre_aliases)
            data.append(t.name)
    return data

def get_proID_techID_mapper(saved_file = r"data/meta data/proID_techID.json"):
    mapper = dict()
    rel = MitreAttack.mitre_attack_data.get_objects_by_type("relationship")
    procedures = [r for r in rel if (r.relationship_type == "uses" and r.target_ref.startswith("attack-pattern"))]
    for p in procedures:
        tech_id = MitreAttack.StixID_2_MitreId[p.target_ref]
        mapper[p.id] = tech_id
    with open(saved_file, "w") as f:
        json.dump(mapper, f, indent=4)

tactic_name2id = {
    "exfiltration": "TA0010",
    "collection": "TA0009",
    "credential-access": "TA0006",
    "command-and-control": "TA0011",
    "defense-evasion": "TA0005",
    "discovery": "TA0007",
    "execution": "TA0002",
    "impact": "TA0040",
    "initial-access": "TA0001",
    "lateral-movement": "TA0008",
    "persistence": "TA0003",
    "privilege-escalation": "TA0004",
    "reconnaissance": "TA0043",
    "resource-development": "TA0042",


}

def get_tech_tac_mapper(saved_file = r"data/meta data/tech_tac_mapper.json"):
    mapper = dict()

    techniques = MitreAttack.mitre_attack_data.get_techniques(include_subtechniques=True, remove_revoked_deprecated=True)
    for t in techniques:
        kill_chain_phases = [tactic_name2id[kc["phase_name"]] for kc in t.kill_chain_phases]
        id_ = t.external_references[0].external_id
        mapper[id_] = kill_chain_phases

    with open(saved_file, "w") as f:
        json.dump(mapper, f, indent=4)


file = r"/Users/khangmai/Library/CloudStorage/GoogleDrive-khangmt@uit.edu.vn/Other computers/My Laptop/Doctoral program/Main/Paper/SaTM/evaluation.xlsx"
def get_True_Positive(ground_truth:list, predicted:list):
    return len(set(ground_truth).intersection(set(predicted)))
def get_False_Positive(ground_truth:list, predicted:list):
    count = 0
    for p in predicted: #this one is predicted as positive
        if p == -1:
            count += 1
    # predicted = set(predicted)
    # for p in predicted: #this one is predicted as positive
    #     if p not in ground_truth: #but not in ground truth => False Positive
    #         count += 1
    return count
def get_False_Negative(ground_truth:list, predicted:list):
    count = 0
    for p in ground_truth: #this one is positive
        if p not in predicted:# not in predicted means Negative => a positive become negative => False Negative
            count += 1
    return count

import Levenshtein
def evalute(ground_truth:list, predicted:list):
    TP = get_True_Positive(ground_truth, predicted)
    FP = get_False_Positive(ground_truth, predicted)
    FN = get_False_Negative(ground_truth, predicted)
    assert TP + FN == len(ground_truth)
    if len(predicted) == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = round(TP/(TP+FP),3)
        recall = round(TP/(TP+FN),3)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = round(2*precision*recall/(precision+recall),3)
    uniques = []
    for g in predicted:
        if g not in uniques:
            uniques.append(g)
    predicted = uniques
    ratio = Levenshtein.ratio(str(ground_truth), str(predicted))
    data = {
        "name": "RAS",
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return data
def evaluation(excel_file="", sheet_name = "Sheet1"):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    ground_truth = df["GT ID"].dropna().tolist()
    predicted = df["Output ID"].dropna().tolist()
    counter_path = df["AttacKG ID"].dropna().tolist()
    # predicted_ = []
    # for p in predicted:
    #     if p not in predicted_:
    #         if isinstance(p, str):
    #             if "," in p:
    #                 splits = p.split(",")
    #                 for s in splits:
    #                     predicted_.append(int(s))
    #         else:
    #             predicted_.append(p)
    predicted = [p for p in predicted if p != 0]# we remove PRE tactics from our evaluation
    ground_truth = [g for g in ground_truth if g != 0]# we remove PRE tactics from our evaluation
    distinct_predicted = list(set(predicted))
    number_duplicate = len(predicted) - len(distinct_predicted)
    RAS_data = evalute(ground_truth, predicted)
    RAS_data["duplicate"] = number_duplicate
    counter_data = evalute(ground_truth, counter_path)
    return {"RAS": RAS_data, "Counter": counter_data}

def average_evaluation(input_data:dict, total:int):
    f1_s = [v["f1"] for k,v in input_data.items()]
    ava_f1 = sum(f1_s)/total
    return ava_f1
def all_evaluation(excel_file="", sheet_names = []):
    final_evaluation = dict()
    if len(sheet_names) == 0:
        sheet_names = pd.ExcelFile(excel_file).sheet_names
    RAS_data = list()
    counter_data = list()
    for s in sheet_names:
        rs = evaluation(excel_file, s)
        rs["RAS"]["name"] = s
        rs["Counter"]["name"] = s
        RAS_data.append(rs["RAS"])
        counter_data.append(rs["Counter"])
    # RAS_data["Average"] = evalute(RAS_data, len(sheet_names))
    # counter_data["Average"] = evalute(counter_data, len(sheet_names))
    return {"RAS": RAS_data, "Counter": counter_data}

# rs = all_evaluation(file1)
# evaluated_file  ="evaluated_result.xlsx"
# with pd.ExcelWriter(evaluated_file) as writer:
#     for k,v in rs.items():
#         df = pd.DataFrame(v)
#         df.to_excel(writer, sheet_name=k)

from keys import *
from classes.decoder import *
import os
import pandas as pd

def record_matching_threshold(directory:str= r"data/campaign/procedure_alignment",tech_alignment_dir = r"data/campaign/tech_alignment"):
    files = os.listdir(directory)
    recoder = dict()
    max_threshold = 1.01
    min_threshold = 0.8
    interval = 0.01
    for f in files:
        if f.endswith(".json"):
            id_ = f.replace(".json", "")
            recoder[id_] = dict()
            with open(os.path.join(directory, f), "r") as file:
                data = json.load(file)
            start_threshold = min_threshold
            tech_alignment_file = os.path.join(tech_alignment_dir, f)
            with open(tech_alignment_file, "r") as file:
                tech_alignment = json.load(file)
            while start_threshold <= max_threshold:
                    print(f"Processing {id_} with threshold {start_threshold}")
                    decoded = Decoder.attack_path_decoding(data, matching_threshold=start_threshold, relax = True,criteria="heuristic", tech_alignment_mapper=tech_alignment,recode=False)
                    if 0 in decoded[0]:
                        del decoded[0][0]
                    recoder[id_][start_threshold] = str((round(start_threshold,2),len(decoded[0])))
                    start_threshold += interval
    df = pd.DataFrame(recoder)
    file_name = "matching_threshold_assing1.xlsx"
    df.to_excel(file_name)
                

# record_matching_threshold()
    
def _find_best_threshold(id_:str, procedure_alignment_dir:str, tech_alignment_dir:str):
    procedure_alignment_file = os.path.join(procedure_alignment_dir, f"{id_}.json")
    tech_alignment_file = os.path.join(tech_alignment_dir, f"{id_}.json")
    with open(procedure_alignment_file, "r") as file:
        procedure_alignment = json.load(file)
    with open(tech_alignment_file, "r") as file:
        tech_alignment = json.load(file)
    recoder = dict()
    max_threshold = 1.01
    min_threshold = 0.8
    interval = 0.01
    start_threshold = min_threshold
    while start_threshold <= max_threshold:
        print(f"Processing {id_} with threshold {start_threshold}")
        decoded = Decoder.attack_path_decoding(procedure_alignment, matching_threshold=start_threshold, relax = True,criteria="heuristic", tech_alignment_mapper=tech_alignment,recode=False)
        recoder[(round(start_threshold,2))]= decoded[3]
        start_threshold += interval
    with open(f"{id_}_threshold.json", "w") as file:
        json.dump(recoder, file, indent=4)

# _find_best_threshold("Frankenstein Campaign", r"data/campaign/procedure_alignment", r"data/campaign/tech_alignment")
        

def record_output_while_varying_threshold(directory:str= r"data/campaign/procedure_alignment",tech_alignment_dir = r"data/campaign/tech_alignment", base_dir:str = "data/evaluation" ):
    files = os.listdir(directory)   
    max_threshold = 1.01
    min_threshold = 0.8
    interval = 0.005
    recode_ = False
    relax  = True
    recoder = dict()
    for f in files:
        
        if f.endswith(".json"):          
            id_ = f.replace(".json", "")
            sheet_name = id_
            recoder[id_] = dict()
            with open(os.path.join(directory, f), "r") as file:
                data = json.load(file)
            uniques = []
            start_threshold = min_threshold
            tech_alignment_file = os.path.join(tech_alignment_dir, f)
            with open(tech_alignment_file, "r") as file:
                tech_alignment = json.load(file)
            while start_threshold <= max_threshold:
                    print(f"Processing {id_} with threshold {start_threshold}")
                    recode_ = False
                    decoded = Decoder.attack_path_decoding(data, matching_threshold=start_threshold, relax = True,criteria="heuristic", tech_alignment_mapper=tech_alignment,recode=False)
                    removed = None
                    if 0 in decoded[0]:
                        removed = decoded[0][0][0]["techID"]
                    recoder[id_][start_threshold] = []
                    for c in decoded[2]:
                        if c == removed:
                            continue
                        uniques.append(c)
                        recoder[id_][start_threshold].append(c)
                    
                    start_threshold += interval
                    start_threshold = round(start_threshold,3)
                    print()
            recoder[id_]["unique"] = list(set(uniques))
    file_name = os.path.join(base_dir, f"varying_threshold.json")
    with open(file_name, "w") as file:
        json.dump(recoder, file, indent=4)

#record_output_while_varying_threshold()
# print()

def _track_metrics_change(original_file:str ="data/evaluation/varying_threshold.json", ground_truth:str = "/Users/khangmai/Library/CloudStorage/GoogleDrive-khangmt@uit.edu.vn/Other computers/My Laptop/Doctoral program/Main/Paper/RAF-AG/2024 May 04 work/varying_threshold.xlsx", choose_threshold_:float = 0.87):
    with open(original_file, "r") as file:
        original = json.load(file)
    data_ = dict()
    sheet_names = pd.ExcelFile(ground_truth).sheet_names
    for s in sheet_names:
        data_[s] = dict()
        df = pd.read_excel(ground_truth, sheet_name=s)
        ids = df["Output Tech ID"].dropna().tolist()
        values = df["Output ID"].dropna().tolist()
        for key, value in zip(ids, values):
            key = str(key).replace("\"", "").replace(",", "").strip()
            if key not in data_[s]:
                data_[s][key] = value
    print()
    final_data = dict()
    for k,v in original.items():
        CTI_name = k
        final_data[CTI_name] = dict()
        df = pd.read_excel(ground_truth, sheet_name=k)
        ground_truth_values = df["GT ID"].dropna().tolist()
        for threshold, values in v.items():
            if threshold == "unique":
                continue
            threshold = round(float(threshold),3)

            predicted = []
            for value in values:
                if value not in data_[k]:
                    print(k, value)
                _label = data_[k][value]
                if _label not in predicted or _label == -1:
                    predicted.append(_label)

            result = evalute(ground_truth_values, predicted)
            final_data[CTI_name][threshold] = result
        print()
    shown_data = []
    for k,v in final_data.items():
        CTI_name = k
        for threshold, values in v.items():
            if threshold == choose_threshold_:
                data__ = values
                data__["name"] = CTI_name
        shown_data.append(data__)
    shown_file = os.path.join("data/evaluation", "chosen_threshold.xlsx")
    with pd.ExcelWriter(shown_file) as writer:

        df = pd.DataFrame(shown_data)
        df.to_excel(writer)
    print()
    data2 = dict()
    average_across_CTI = dict()
    for k,v in final_data.items():
        CTI_name = k
        data2[CTI_name] = dict()
        precision_list = []
        recall_list = []
        f1_list = []
        previous_precision = 0
        for threshold, values in v.items():
            threshold = round(float(threshold),3)
            if threshold not in average_across_CTI:
                average_across_CTI[threshold] = dict()
                average_across_CTI[threshold]["precision"] = []
                average_across_CTI[threshold]["recall"] = []
                average_across_CTI[threshold]["f1"] = []
            if threshold == 0.99:
                previous_precision = values["precision"]
            if threshold == 1.0:
                average_across_CTI[threshold]["precision"].append(previous_precision)
                average_across_CTI[threshold]["recall"].append(0)
                average_across_CTI[threshold]["f1"].append(0)
            else:
                average_across_CTI[threshold]["precision"].append(values["precision"])
                average_across_CTI[threshold]["recall"].append(values["recall"])
                average_across_CTI[threshold]["f1"].append(values["f1"])
            precision_list.append((threshold,values["precision"]))
            recall_list.append((threshold,values["recall"]))
            f1_list.append((threshold,values["f1"]))
        data2[CTI_name]["name"] = CTI_name
        data2[CTI_name]["precision"] = precision_list
        data2[CTI_name]["recall"] = recall_list
        data2[CTI_name]["f1"] = f1_list

    data_for_graph = os.path.join("data/evaluation", "data_for_graph.jsonl")
    for k,v in data2.items():
        with jsonlines.open(data_for_graph, mode="a") as writer:
            writer.write(v)
    print()
    average = dict()
    average["precision"] = []
    average["recall"] = []
    average["f1"] = []
    average["f2"] = []
    for threshold,value in average_across_CTI.items():
        # average_precision = round(sum(value["precision"])/len(value["precision"]),2)
        # average_recall = round(sum(value["recall"])/len(value["recall"]),2)
        average["precision"].append((threshold, round(sum(value["precision"])/len(value["precision"]),3)))
        average["recall"].append((threshold, round(sum(value["recall"])/len(value["recall"]),3)))
        average["f1"].append((threshold, round(sum(value["f1"])/len(value["f1"]),3)))
    for p, r in zip(average["precision"], average["recall"]):
        try:
            average["f2"].append((p[0], round(2*p[1]*r[1]/(p[1]+r[1]),3)))
        except:
            average["f2"].append((p[0], 0))
    average_file = os.path.join("data/evaluation", "average_across_CTI.json")
    with open(average_file, "w") as file:
        json.dump(average, file)
    print()
_track_metrics_change()