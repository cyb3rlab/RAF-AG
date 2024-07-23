
import itertools
import json
import re
from modules import tech_tac_mapper
from mitre_attack import MitreAttack
class Technique:
    def __init__(self,tech_id, locations= [], procedures = []):
        self.tech_id = tech_id
        self.tech_name = MitreAttack.get_technique_name(tech_id)
        self.main_pattern = r"\b" + self.tech_name + r"\b"
        self.tactics = tech_tac_mapper[tech_id]
        self.best_phrases = []
        if "." in self.tech_id and "TA0007" not in self.tactics and "TA0009" not in self.tactics and " " not in self.tech_name:
            self.best_phrases.append(self.tech_name)
        self.locations = locations
        self.procedures = procedures # list of procedure IDs
        # self.id = tech_id+ "_" + "_".join(locations)
        self.id = tech_id
        self.label_2_id = dict()
        self.id_2_label = dict()
        self.graph_nodes = dict()
        self.graph_edges = dict()
        self.verbs = dict()



    def best_entities(self):
        labels = list()
        nodes = list(self.graph_nodes.values())
        nodes.sort(key = lambda x: x["count"], reverse = True)
        for i in range(len(nodes)):
            if nodes[i]["label"] == "ACTOR" or nodes[i]["label"] == "OTHER":
                continue
            if i >= len(nodes):
                break
            labels.append(nodes[i]["label"]) 
        self.features = labels
    def initilize_label(self, label):
        if label in self.label_2_id:
            return
        id_ = len(self.label_2_id)
        self.label_2_id[label] = id_
        self.id_2_label[id_] = label
        self.graph_nodes[id_] = dict()
        self.graph_nodes[id_]["count"] = 0
        self.graph_nodes[id_]["id"] = id_
        self.graph_nodes[id_]["texts"] = []

    def add_node(self, _node, procedure):
        node = _node["meta"]
        node_label = node["label"].copy()
        node_label = list(set(node_label))
        if "texts" in node:
            texts = procedure.get_true_text(texts = node["texts"])
        else:
            texts = [procedure.get_true_text(text = node["text"])]
        texts = list(set(texts))
        cve_pattern = r"\b(cve\-[0-9]{4}\-[0-9]{4,6})\b"
        for t in texts:
            if re.search(cve_pattern, t,re.IGNORECASE):
                self.best_phrases.append(t)
            if " " not in self.tech_name:
                if re.search(self.main_pattern, t,re.IGNORECASE) and  not (len(node_label) == 1 and node_label[0] == "DATA") and "." in self.tech_id and "TA0007" not in self.tactics and "TA0009" not in self.tactics:
                    self.best_phrases.append(t.lower())
        # self.best_phrases.extend(procedure.procedure_special_phrases)
        id_list = []
        # if this node from procedure belong to more than one type, we add the texts to all of related node types
        for label in node_label:
            self.initilize_label(label)
            label_id = self.label_2_id[label]
            self.graph_nodes[label_id]["texts"].extend(texts)
            self.graph_nodes[label_id]["label"] = label
            self.graph_nodes[label_id]["count"] += len(texts)
            id_list.append(label_id)
        return id_list
    

    def add_procedures(self, data):
        for p in self.procedures:
            if p not in data:
                continue
            procedure = data[p]
            self.add_procedure(procedure)
        # self.total_node_occurence = sum([v["count"] for k,v in self.graph_nodes.items()])
        self.total_edge_occurence = sum([v["count"] for k,v in self.graph_edges.items()])
        self.best_entities()
    def add_edge(self, source, dest):
        new_edge_id = str(source) + "_" + str(dest)
        if new_edge_id in self.graph_edges:
            self.graph_edges[new_edge_id]["count"] += 1
            return
        self.graph_edges[new_edge_id] = dict()
        data = dict()
        data["id"] = new_edge_id
        data["source"] = source
        data["dest"] = dest
        data["count"] = 1
        self.graph_edges[new_edge_id] = data



    def add_procedure(self, procedure):
        if procedure.id not in self.procedures:
            self.procedures.append(procedure.id)
        _added_nodes = {} #prevent adding same node twice
        for k,v in procedure.graph_edges.items():
            verb = v["verb"]
            if verb not in self.verbs:
                    self.verbs[verb] = 1
            else:
                    self.verbs[verb] += 1
            if v["source"] in procedure.graph_nodes:

                source = procedure.graph_nodes[v["source"]]

                if source["meta"]["id"] not in _added_nodes:
                    source_ids = self.add_node(source, procedure)
                    _added_nodes[source["meta"]["id"]] = source_ids
            if v["dest"] in procedure.graph_nodes:
                dest = procedure.graph_nodes[v["dest"]]
                if dest["meta"]["id"] not in _added_nodes:
                    dest_ids = self.add_node(dest, procedure)
                    _added_nodes[dest["meta"]["id"]] = dest_ids
            if v["source"] in procedure.graph_nodes and v["dest"] in procedure.graph_nodes:    
                source_ids = _added_nodes[source["meta"]["id"]]
                dest_ids = _added_nodes[dest["meta"]["id"]]
                product = list(itertools.product(source_ids, dest_ids))
                for p in product:
                    if p[0] == p[1]:
                        continue
                    self.add_edge(p[0], p[1])



    
    def to_json(self, path: str):
        data = dict()
        data["id"] = self.id
        data["tech_id"] = self.tech_id
        data["tech_name"] = self.tech_name
        if len(self.best_phrases) > 0:
            self.best_phrases = list(set(self.best_phrases))
        data["best_phrases"] = self.best_phrases
        data["tactics"] = self.tactics
        data["features"] = self.features
        data["locations"] = self.locations
        data["procedures"] = self.procedures
        graph_node_list = [v for k,v in self.graph_nodes.items()] # prevent json change int keys to string
        data["graph_nodes"] = graph_node_list
        graph_edge_list = [v for k,v in self.graph_edges.items()]
        data["graph_edges"] = graph_edge_list
        data["verbs"] = self.verbs
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
            id = data["id"]
        verbs = data["verbs"]
        tech_id = data["tech_id"]
        locations = data["locations"]
        procedures = data["procedures"]
        tactics = data["tactics"]
        obj = cls(tech_id, locations, procedures)
        obj.tech_name = data["tech_name"]
        obj.tactics = tactics
        obj.best_phrases = data["best_phrases"]
        obj.main_pattern = r"\b" + obj.tech_name + r"\b"
        obj.graph_nodes = dict()
        obj.graph_edges = dict()
        obj.features = data["features"]
        obj.verbs = verbs
        phrases = []
        for node in data["graph_nodes"]:
            obj.graph_nodes[node["id"]] = node
            if node["label"] == "ACTOR" or node["label"] == "OTHER":
                continue
            phrases.extend(node["texts"])
        for edge in data["graph_edges"]:
            obj.graph_edges[edge["id"]] = edge
        obj.phrases = list(set(phrases))
        return obj
    


    def normalization_scale(self, num_occurances):
        return num_occurances/self.total_node_occurence