from classes.paragraph import Paragraph
import json
from modules import *
from keys import *
from copy import deepcopy

import os
class Campaign(Paragraph):
    def __init__(self, text:str = "", id:str = "", image_path:str = ""):
        if text != "":
            super().__init__(text)
            super().data_generation(is_campaign = True)
            super().get_phrases()
            if image_path != "":
                super().rescontruct_graph() # this step will change the graph to relfect current graph nodes and edges
                self.draw(image_path)
            self.mapper = dict()
            self.id = id
        
    def to_dict(self):
        data = super().to_dict()
        data["mapper"] = self.mapper
        data["id"] = self.id
        return data
    
    def to_json(self, path: str):
        data = super().to_dict()
        data["mapper"] = self.mapper
        data["id"] = self.id
        with open(path, "w") as f:
            json.dump(data, f, indent = 4)
    def from_json_object(self, json_object):
        data = json_object
        self.mapper = data["mapper"]
        self.id = data["id"]
        super().from_dict(data)
        super().get_phrases()
    def from_json(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
            self.mapper = data["mapper"]
            self.id = data["id"]
            super().from_dict(data)
            super().get_phrases()
    
    def to_pickle(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
    @classmethod
    def my_copy(cls, campaign):
        new_campaign = cls()
        new_campaign.graph_nodes = deepcopy(campaign.graph_nodes)
        new_campaign.graph_edges = deepcopy(campaign.graph_edges)
        new_campaign.graph = deepcopy(campaign.graph)
        new_campaign.mapper = {}
        new_campaign.id = campaign.id
        return new_campaign

    


    