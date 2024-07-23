from classes.paragraph import Paragraph
import json
import re
from modules import *
from keys import *
import statistics


class Procedure(Paragraph):
    def __init__(self, text:str= "", tech_id:str = "", procedure_id = "", locations:list = [], special_id = "", image_path = ""):

        if text != "":
            self.id = procedure_id
            self.tech_id = tech_id
            self.locations = locations
            if text.startswith("can"): # fix some special cases lacking of subject
                 text = "It " + text
            self.text = text
            self.special_id = special_id
            super().__init__(self.text)
            super().data_generation(is_campaign = False)
            
            if image_path != "":
                super().rescontruct_graph()
                self.draw(image_path)
            # self.remove_none_entity_node()

        # self.self_standard_deviation()
    


    def remove_none_entity_node(self):
        none_entity_nodes = []
        temp_graph = self.graph_nodes.copy()
        for k,v in self.graph_nodes.items():
            if len(v["meta"]["label"]) == 0:
                if "text" in v["meta"]:
                    none_entity_nodes.append(v["meta"]["text"])
                if "texts" in v["meta"]:
                    none_entity_nodes.extend(v["meta"]["texts"])
                del temp_graph[k]
        self.graph_nodes = temp_graph
        return none_entity_nodes


    def to_json(self, path: str, reverse_text = True):
        data = super().to_dict(reverse_text)
        data["id"] = self.id
        data["tech_id"] = self.tech_id
        data["location"] = self.locations
        data["special_id"] = self.special_id
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def from_json(self, path: str="", json_object = None):
        if json_object is not None:
            data = json_object
            self.id = data["id"]
            self.tech_id = data["tech_id"]
            self.locations = data["location"]
            self.special_id = data["special_id"]
            super().from_dict(data)
            super().get_phrases()
        else:
            with open(path, "r") as f:
                data = json.load(f)
                self.id = data["id"]
                self.tech_id = data["tech_id"]
                self.locations = data["location"]
                self.special_id = data["special_id"]
                super().from_dict(data)
                super().get_phrases()

   
 

    

    def self_standard_deviation(self):
        input = []
        for e in self.graph_edges:
            input.append(e["index"])
        
        self.std = statistics.stdev(input)
    



