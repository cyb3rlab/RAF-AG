

#read association data
#create statistic data
#choose top k best frequency association from a specific technique_ID
#For each pair, connect platform information
import os
import statistics
import json
from keys import Keys
import pandas as pd
import regex as re
from modules import *
class stat_Calculator():
    def __init__(self, attack_path_dir:str = "", saved_dir:str = ""):
        self.dir = attack_path_dir
        self.saved_dir = saved_dir
        self.data = dict()
        self.tech_index = dict()
        self.pair_index = dict()


    def save(self):
        data_file_path = os.path.join(self.saved_dir, "frequency_data.json")
        tech_index_file_path = os.path.join(self.saved_dir, "tech_index.json")
        pair_index_file_path = os.path.join(self.saved_dir, "pair_index.json")
        with open(data_file_path, "w") as f:
            json.dump(self.data, f, indent=4)
        with open(tech_index_file_path, "w") as f:
            json.dump(self.tech_index, f, indent=4)
        with open(pair_index_file_path, "w") as f:
            json.dump(self.pair_index, f, indent=4)
    @classmethod
    def check_pair_potential(cls, source:str, target:str):
        if source["techID"] == target["techID"]:
            return False
        source_sent_indexes = source["sent_indexes"]
        target_sent_indexes = target["sent_indexes"]
        source_sent = statistics.mean(source_sent_indexes)
        target_sent = statistics.mean(target_sent_indexes)
        if abs(source_sent - target_sent) > 5:
            return False # two techniques are too far from each other
        
        return True

    def threat_pair_generation(self, data:dict):
        data = { int(k):v for k,v in data.items()}
        keys = list(data.keys())
        keys.sort()
        combined_keys = list()
        for i in range(0, len(keys)-1):
            combined_keys.append((i, i+1))
        unique_pairs = list()
        for k in combined_keys:
            k1 = keys[k[0]]
            k2 = keys[k[1]]
            combinations = list(itertools.product(data[k1], data[k2]))
            for c in combinations:
                if c[0]["techID"] == c["techID"]:
                    continue
                pairID = c[0]["techID"] + "__" + c[1]["techID"]
                if not stat_Calculator.check_pair_potential(c[0], c[1]):
                    continue
                if pairID not in unique_pairs:
                    unique_pairs.append(pairID)

        for k,v in data.items():
            if len(v) == 0 or len(v) == 1:
                continue
            if len(v) == 2:
                source = v[0]
                target = v[1]
                if not stat_Calculator.check_pair_potential(source, target):
                    continue
            source_order_ids = source["order_ids"]
            target_order_ids = target["order_ids"]
            source_mean_id = statistics.mean(source_order_ids)
            target_mean_id = statistics.mean(target_order_ids)
            if source_mean_id < target_mean_id:
                pairID = source["techID"] + "__" + target["techID"]
                if pairID not in unique_pairs:
                    unique_pairs.append(pairID)
            else:
                if source_mean_id > target_mean_id:
                    pairID = target["techID"] + "__" + source["techID"]
                    if pairID not in unique_pairs:
                        unique_pairs.append(pairID)
        return unique_pairs
    def load(self):
        data_file_path = os.path.join(self.saved_dir, "frequency_data.json")
        tech_index_file_path = os.path.join(self.saved_dir, "tech_index.json")
        pair_index_file_path = os.path.join(self.saved_dir, "pair_index.json")
        with open(data_file_path, "r") as f:
            self.data = json.load(f)
        with open(tech_index_file_path, "r") as f:
            self.tech_index = json.load(f)
        with open(pair_index_file_path, "r") as f:
            self.pair_index = json.load(f)
        self.calculate_pair_propapility()
    def calculate_frequency(self):
       files = os.listdir(self.dir)
       for f in files:
            unique_tech = list()
            if f.endswith(".json"):
                path = os.path.join(self.dir, f)
                with open(path, "r") as f:
                    data = json.load(f)

                    attack_path = data["attack_path"]
                    threat_pairs = self.threat_pair_generation(attack_path)
                    for p in threat_pairs:
                        if p not in self.pair_index:
                            self.pair_index[p] = 0
                        self.pair_index[p] += 1
                        technique = p.split("__")
                        source = technique[0]
                        target = technique[1]
                        if source not in self.data:
                            self.data[source] = dict()
                        if target not in self.data[source]:
                            self.data[source][target] = 0
                        self.data[source][target] += 1

            unique_tech = list(set(unique_tech))
            for t in unique_tech:
                if t not in self.tech_index:
                    self.tech_index[t] = 0
                self.tech_index[t] += 1
        
    def calculate_pair_propapility(self):
        self.prop_data = self.data.copy()
        for k,v in self.prop_data.items():
            total = sum(v.values())
            for k1,v1 in v.items():
                self.prop_data[k][k1] = v1/total

    def get_pair_frequency(self, sourceID:str, targetID:str):
        if sourceID in self.data:
            if targetID in self.data[sourceID]:
                return self.data[sourceID][targetID]
        return 0
    
    def get_pair_propability(self, sourceID:str, targetID:str):
        if sourceID in self.prop_data:
            if targetID in self.prop_data[sourceID]:
                return self.prop_data[sourceID][targetID]
        return 0
    
    def get_top_K_common_tech(self, k:int=1):
        sorted_index = sorted(self.tech_index.items(), key=lambda x: x[1], reverse=True)
        return sorted_index[:k]
    
    def get_top_K_common_pair(self, k:int=1):
        sorted_index = sorted(self.pair_index.items(), key=lambda x: x[1], reverse=True)
        return sorted_index[:k]
    
    def get_top_K_common_pair_from_source(self, sourceID:str, k:int=1):
        sorted_index = sorted(self.data[sourceID].items(), key=lambda x: x[1], reverse=True)
        return sorted_index[:k]
    

    def pre_association_addition(self, addition_data:dict):
        for k,v in addition_data.items():
            if k not in self.data:
                self.data[k] = dict()
            for k1,v1 in v.items():
                if k1 not in self.data[k]:
                    self.data[k][k1] = 0
                self.data[k][k1] += v1
        self.calculate_pair_propapility()