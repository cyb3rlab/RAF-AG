from classes.preliminary_extraction import action_extraction_per_sentence
from classes.subject_verb_object_extract import get_examples_cases
# from classes.ner_model import ner_extract
from classes.heuristic_model import heuristic_extract
from keys import Keys
import Levenshtein
top_value = Keys.TOP_VALUE
import json
import os
import re
def remove_redudant_label(labels):
    for l in labels:
        if l != "ACTOR" and l != "OTHER":
            return l
    
def delete_ENTITY(text):
    entity_pattern = r"\b(ENTITY)[0-9]+\b"
    return re.sub(entity_pattern, "", text).strip()
class Sentence:
    def __init__(self, sent = "", sent_id= "", replacement:dict=None):
        self.id = sent_id # order number of the sentence in the paragrah
        self.text = sent # the text of the sentence
        if self.text != "":
            self.replacement_mapper = replacement         
            self.doc, self.svos, self.chains =  action_extraction_per_sentence(sentence= sent) #list of tripple S-V-O (subject-verb-object)
            # self.example_cases = get_examples_cases(self.doc)

            self._extract_entities() # list of entities in the sentence
            self._svos_analysis()
            # self.handling_examples()





    def log_events(self,file):
        self.log_path = file
        try:
            with open(self.log_path, "r") as f:
                self.event_log = json.load(f)
        except:
            self.event_log = list()
        events = list()
        for svo in self.svos:
            events.append({"sub": svo["sub"]["text"], "verb": svo["verb"]["text"], "obj": svo["obj"]["text"]})
        self.event_log.append({"text": self.text, "svos": events})
        with open(self.log_path, "w") as f:
            json.dump(self.event_log, f,indent=4)
    def to_dict(self,reverse_text = True):
        data = dict()
        data["id"] = self.id
        #reverse the replacement
        if reverse_text:
            data["text"] = self.reverse_replacement(self.text)[0]
        else:
            data["text"] = self.text
        data["svos"] = self.svos
        # try:
        #     with open("data.json", "w") as f:
        #         json.dump(data, f)
        # except:
        #     print("sentence error")
        return data
    
    
    def from_dict(self, data):
        
        self.id = data["id"]
        self.text = data["text"]
        self.svos = data["svos"]

    
    def _extract_entities(self):
        if len(self.svos) == 0:
            return
        # self.spacy_entities = ner_extract(self.text)
        if len(self.svos) > 0:
            self.heuristic_entities = heuristic_extract(self.svos)
            print()


        

    def _extract_label_from_replacement(self, text):
        entities = list()
        for key, value in self.replacement_mapper.items():
            if key in text:
                entities.append(value["label"])
        return entities
    
    def _svos_analysis(self):
        if len(self.svos) == 0:
            return
        # examples =[e["text"] for e in self.example_cases]
        # for i in range(0, len(self.example_cases)):
        #     self.example_cases[i] = self.element_refinement(self.example_cases[i], _type = "object")
        
        elements = set()
        svos =  list()
        entities_marker = [0]* len(self.heuristic_entities)
        for s in self.svos:
            svo = dict()
            sub= s[0]
            
            verb = s[1]
            obj = s[2]
            # if obj["text"] in examples :
            #     continue
            if verb["negation"]: # if the verb is negated, we ignore the sentence
                continue
            svo["sub"] = self.element_refinement(sub, _type = "subject")
            
            
            svo["verb"] = verb
           
           
           
            svo["obj"] = self.element_refinement(obj, _type = "object")
            if len(svo["sub"]["label"]) == 1 and len(svo["obj"]["label"]) == 1 and "OTHER" in svo["sub"]["label"] and "OTHER" in svo["obj"]["label"]:
                continue
            svos.append(svo)
        self.svos = svos
        self.entities_marker = entities_marker


    def element_refinement(self, element, _type = "subject"):
        data = element
        if isinstance(element, str) : # "ANY" case
                data = {"text":data,"start":0, "end":0, "label":["ACTOR"], "id": 0, "sent_index": self.id}

        _text = data["text"]
        if "label" not in data:
                data["label"] = list()
        data["label"].extend(self._extract_label_from_replacement(data["text"]))


        if _text.lower() in ["attacker", "adversary"]:
            data["label"].append("ACTOR")
        data["sent_index"] = self.id
        data["type"] = _type



        #todo how to split ENTITY into a seperate entity.
        for i in range(0, len(self.heuristic_entities)):
            e = self.heuristic_entities[i]
            _data_text = delete_ENTITY(data["text"])
            if Levenshtein.ratio(e["text"], _data_text) > 0.8:
                    data["label"].append(e["label"])
                #update the label of the entity
        # for i in range(0, len(self.spacy_entities)):
        #     e = self.spacy_entities[i]
        #     if Levenshtein.ratio(e["text"], data["text"]) > 0.8 or data["text"] in e["text"]:
        #         # if e["label"] == "ACTOR" and len(data["label"]) > 0 and "ACTOR" not in data["label"]:
        #         #     continue
        #         # if e["label"] == "OTHER" and len(data["label"]) > 0 and "OTHER" not in data["label"]:
        #         #     continue
        #         data["label"].append(e["label"])

        data["label"] = list(set(data["label"]))
        if len(data["label"]) == 2:
            if "OTHER" in data["label"]:
                data["label"].remove("OTHER")
            else:
                if "ACTOR" in data["label"]:
                    data["label"].remove("ACTOR")
        # #return true text
        _text,_ = self.reverse_replacement(data["text"])
        data["text"] = _text

        return data



    def handling_examples(self):
        _new_svos = list()
        _visited = list()
        if len(self.example_cases) == 0:
            return
        index = self.example_cases[0]["id"]
        elements = []
        for i in range(0, len(self.svos)):
            if self.svos[i]["sub"]["id"] < index:
                    elements.append(self.svos[i]["sub"])
            if self.svos[i]["obj"]["id"] < index:
                    elements.append(self.svos[i]["obj"])
        for e in self.example_cases:
            keys =   e["label"].copy()         


            for j in range(len(elements)-1,-1,-1 ):
                ele = elements[j]
                _keys  = ele["label"].copy()
                if len(set(keys).intersection(set(_keys))) > 0:
                    subject = ele
                    object = e
                    verb = {"text": "is", "id": subject["id"]  }       
                    svo = {"sub": subject, "verb": verb, "obj": object}
                    _new_svos.append(svo)
        self.svos.extend(_new_svos)

    def reverse_replacement(self, text):
        flag = False
        for k,v in self.replacement_mapper.items():
            if k in text:
                text = text.replace(k, v["text"])
                flag = True
        return text, flag