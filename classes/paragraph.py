from classes.sentence import Sentence
from language_models import nlp
from modules import common_fixing_pattern
from classes.preprocessings import text_preprocessing
from classes.heuristic_model import heuristic_extract_, replace_special_entities
import pandas as pd
import re
import os
import json
from networkx import DiGraph
import math
import networkx as nx
from matplotlib import figure
import matplotlib.pyplot as plt
image_dir = r"Images"
json_dir = r"Json"
from modules import malwares
from modules import check_if_verb_is_strong
from keys import Keys
top_value = Keys.TOP_VALUE
from copy import deepcopy, copy
purposes  = ["persistence", "execution","privilege escalation","defense evasion","credential access","discovery","lateral movement","collection","exfiltration","command and control","impact","initial access"]
purpose_pattern = r"(lateral movement|initial access|execution|exfiltration|escalation|evasion|discovery|collection|credential access|persistence)$"
class Paragraph:
    def __init__(self, text):
        if text is not None:
            self.graph = nx.DiGraph()
            self.text = text

            # self.draw()

    def data_generation(self, is_campaign = True):
            self.is_campaign = is_campaign
            self.preprocessing(flag = is_campaign)
            self.text, self.replacement_mapper = self.replace_special_entity()


            self.doc = nlp(self.text)
            self.backup_sents = list(self.doc.sents)
            self.graph_nodes = dict()
            self.graph_edges = dict()
            # the spacy doc of the paragraph
            sents = list(self.doc.sents)
            self.sentences = list()
            for i in range(0,len(sents)):
                sent = sents[i]
                if "where" in sent.text or "when" in sent.text or "how" in sent.text or "why" in sent.text or "what" in sent.text or "who" in sent.text or "which" in sent.text:
                    print("debug")
                _sent = Sentence(sent.text, i, self.replacement_mapper)
                self.sentences.append(_sent)
            if is_campaign: #this is a campaign
                # self.coreferee_resolution()
                self.generate_graph()
                # self.handling_coref_graph()
                self.back_up_graphs = self.graph.copy()
                # self.handling_coref_graph_updated_version()
                self.simplify_graph1()
                self.simplify_graph2()
                self.generate_edge_dict()
                self.regenerate_graph_nodes()
            else: # 
                self.coreferee_resolution()
                self.generate_graph()
                # self.handling_coref_graph()
                self.back_up_graphs = self.graph.copy()
                self.handling_coref_graph_updated_version()
                self.simplify_graph1()
                self.simplify_graph2()
                self.conjuntion_simplification()
                self.regenerate_graph_nodes()
                self.generate_edge_dict()
            self.graph = self.graph.to_undirected()

    def to_dict(self, reverse_text = True):
        data= dict()
        # data["is_campaign"] = self.is_campaign

        data["text"] = self.text
        graph_node_list = [v for k,v in self.graph_nodes.items()] # prevent json change int keys to string
        data["graph_nodes"] = graph_node_list
        graph_edge_list = [v for k,v in self.graph_edges.items()]
        data["graph_edges"] = graph_edge_list
        #since json cannnot serialize contracted networkx graph, we need to save the original graph instead
        # data["graph"] = nx.node_link_data(self.back_up_graphs)
        # data["chains"]  = self.chains
        data["replacement_mapper"] = self.replacement_mapper
        # if not self.is_campaign:
        #     data["special_phrases"] = self.procedure_special_phrases
        data["sentences"] = list()
        for s in self.sentences:
            data["sentences"].append(s.to_dict(reverse_text))
        return data
    


    def rescontruct_graph(self):
        self.graph = nx.DiGraph()
        for node in self.graph_nodes.values():
            self.graph.add_node(node["id"])
        for edge in self.graph_edges.values():
            if "verbs" in edge:
                self.graph.add_edge(edge["source"],edge["dest"], verb = edge["verb"], verbs = edge["verbs"], index = edge["index"])
            else:
                self.graph.add_edge(edge["source"],edge["dest"], verb = edge["verb"], index = edge["index"])
        self.graph = self.graph.to_undirected()
    def from_dict(self, data):
        # if "special_phrases" in data:
        #     self.procedure_special_phrases = data["special_phrases"]
        self.text = data["text"]
        self.replacement_mapper = data["replacement_mapper"]
        graph_node_list = data["graph_nodes"]
        self.graph_nodes = dict()
        for node in graph_node_list:
            self.graph_nodes[node["id"]] = node
        graph_edge_list = data["graph_edges"]
        self.graph_edges = dict()
        for edge in graph_edge_list:
            self.graph_edges[edge["id"]] = edge
        self.rescontruct_graph()
        # self.chains = data["chains"]
        # self.handling_coref_graph_updated_version() since text are accumuated into contracted node, no need to handle coref again
        self.sentences = list()
        for s in data["sentences"]:
                sent = Sentence()
                sent.from_dict(s)
                self.sentences.append(sent)
        # # after loading the graph, we need to handle coref and update the graph edges 
        # self.handling_coref_graph_updated_version()
        # # self.simplify_graph1()
        # self.generate_edge_dict()
        # self.regenerate_graph_nodes()
    def preprocessing(self, flag = True):
        # char_remove = "\n|\t|\r"
        # self.text = re.sub(char_remove, " ", self.text).replace("  ", " ")
        self.text = text_preprocessing(self.text, flag = flag)
        #we could do some more preprocessing here

    def coreferee_resolution(self):
        if "coreferee" not in nlp.pipe_names:
            print("adding coreferee to pipeline")
            nlp.add_pipe("coreferee")
        if self.doc is None:
            self.doc = nlp(self.text)
        coref= self.doc._.coref_chains
        sents = list(self.doc.sents)
        new_chains = dict()

        for i in range (0, len(coref)):
                
                c= coref[i]
                main=  self.doc[c.most_specific_mention_index]
                for m in c: 

                    token_indexes = m.token_indexes
                    chains= list()
                    unique = list()
                    if len(token_indexes) > 1:
                        continue
                    for token_index in token_indexes:
                        token = self.doc[token_index]
                        if main.pos_ in ["PROPN"] and token.pos_ in ["PROPN"] and main.text != token.text:
                            continue
                        if token.text in ["My", "my", "his", "their", "our", "Our", "His", "Their", "its", "Its","ANY"]:
                            continue
                        token_start = token.idx
                        token_end = token.idx + len(token.text)
                        for sent_index in range(0,len(sents)):
                            sent = sents[sent_index]
                            sent_start = sent.start_char
                            sent_end = sent.end_char
                            sentence = self.sentences[sent_index]
                            if token_start >= sent_start and token_end <= sent_end:
                                # this mention is within this sentence
                                new_token_start = token_start - sent_start
                                new_token_end = token_end - sent_start
                                assert sent.text[new_token_start:new_token_end] == token.text
                                assert sentence.text[new_token_start:new_token_end] == token.text
                                # check if this mention is within the sub and obj of svo
                                for svo in sentence.svos:
                                    if svo["sub"]["start"]<= new_token_start and svo["sub"]["end"]>= new_token_end:
                                        token_id = svo["sub"]["id"]
                                        if token_id in unique:
                                            continue
                                        chains.append({"text":token.text, "start":new_token_start, "end":new_token_end, "sent_index":sent_index, "mention_index":token_id, "token_pos":token.pos_})
                                        unique.append(token_id)
                                    if svo["obj"]["start"]<= new_token_start and svo["obj"]["end"]>= new_token_end:
                                        token_id = svo["obj"]["id"]
                                        if token_id in unique:
                                            continue
                                        unique.append(token_id)
                                        _chain = {"text":token.text, "start":new_token_start, "end":new_token_end, "sent_index":sent_index, "mention_index":token_id, "token_pos":token.pos_}
                                        chains.append(_chain)
                    if len(chains)>0:
                        if i not in new_chains:
                            new_chains[i] = list()
                            new_chains[i].append(chains)
                        else:
                            new_chains[i].append(chains)
        self.chains = new_chains




    def _coref_resolve(self, sent_index, mention_index ):
        for k,v in self.chains.items():
            chain = v
            for c in chain:
                if len(c)>1:
                    continue
                token = c[0]
                text = token["text"]
                mention_start = token["start"]
                mentioned_end = token["end"]
                _mention_index = token["mention_index"]
                _sent_index = token["sent_index"]
                if _sent_index == sent_index and mention_index == _mention_index:
                    return chain

        return None
    

    def generate_graph(self):
        edge_index = 0
        for idx in range(0,len(self.sentences)):
            sent_idx = self.sentences[idx].id
            sent = self.sentences[idx]
            for jdx in range(0,len(sent.svos)):
                
                svo = sent.svos[jdx]
                #we will add anything to the graph and reduce it by checking valid edges / nodes later
                # if not self._check_valid_svo(svo):

                sub = svo["sub"]
                self.add_node(sub,sent_idx*1000+ sub["id"])
                verb = svo["verb"]
                obj = svo["obj"]
                if self.is_campaign:
                    obj_id = sent_idx*1000+ obj["id"]
                    if self.graph.has_node(obj_id) and "verbs" in obj:
                        temp_obj = self.graph_nodes[obj_id]
                        if "verbs" not in temp_obj["meta"]:
                            temp_obj["meta"]["verbs"] = list()
                        temp_obj["meta"]["verbs"].extend(obj["verbs"])
                        

                self.add_node(obj,sent_idx*1000+ obj["id"])
                edge_index += self.add_edge(sent_idx*1000+ sub["id"], sent_idx*1000+ obj["id"], verb["text"], edge_index, verb["flag"])
                # the edge index now represent the order of svo in the whole paragraph

    def add_edge(self, sub_id, obj_id, verb, index, flag): 
        if not self.graph.has_node(sub_id) or not self.graph.has_node(obj_id):
            return 0
        self.graph.add_edge(sub_id, obj_id, verb = verb, index = index, flag = flag )  
        return 1

    def add_node(self, node_dict, id ):
        if node_dict["text"] in ["I", "we", "We"]: # we do not care about these pronouns since they imply the speaker/ security expert not the threat actor, thus, the action is not performed by the threat actor
            return
        if self.graph.has_node(id):
            return
        
        #since there are no two entities with the same ID but different text, we can use the ID as the node ID and overwrite the existing info in graph_nodes wihout worrying
        self.graph_nodes[id] = { "id": id , "contracted": 0, "meta": node_dict} # we save the backup here incase we need to retrive origin svo back from graph
        self.graph.add_node(id) #we do not add any attribute to the node since they are saved into graph_nodes dict already

    def find_main_coref(self, chain):
        if chain is None:
            return False
        for c in chain:
            token = c[0]
            if token["token_pos"] in ["PROPN"]:
                return token
        for c in chain:
            token = c[0]
            if token["token_pos"] in ["NOUN"]:
                return token
        return chain[0][0]
    def handling_coref_graph(self):
        for id in self.graph:
            sent_index = id//1000
            mention_index = id%1000
            chain = self._coref_resolve(sent_index, mention_index)
            if chain is None: # there is no coref to/from this node
                continue
            main_coref = self.find_main_coref(chain)
            main_id = main_coref["sent_index"]*1000 + main_coref["mention_index"]
            if not self.graph.has_node(main_id):
                main_id = id # if this main_id is not in the graph, we let the first mention to be the main since node added in temporal order
            for c in chain:
                    if len(c)>1:
                        continue
                    token = c[0]
                    
                    _sub_id = token["sent_index"]*1000 + token["mention_index"]
                    if _sub_id == main_id:
                        continue
                    if not self.graph.has_node(_sub_id) :
                        continue
                    if self.graph.has_edge(main_id, _sub_id) or self.graph.has_edge(_sub_id, main_id):
                        continue
                    else:
                        self.graph.add_edge(main_id, _sub_id, verb = "coref")
    def handling_coref_graph_updated_version(self):
        # print("handling coref graph")
        for k,v in self.chains.items():
            main_coref = self._find_main_coref_updated_version(v)
            if main_coref is None:
                continue
            main_id = main_coref["sent_index"]*1000 + main_coref["mention_index"]
            assert self.graph.has_node(main_id)
            main_id = str(main_id) if main_id not in self.graph_nodes else main_id
            
            
            main_node = self.graph_nodes[main_id].copy()
            for c in v:
                # if len(c) > 1 :
                #     continue
                if len(c) == 0:
                    print("debug")
                token = c[0]
                _sub_id = token["sent_index"]*1000 + token["mention_index"]
                _sub_id = str(_sub_id) if _sub_id not in self.graph_nodes else _sub_id
                if _sub_id == main_id:
                    continue # we dont want create loop
                if not self.graph.has_node(_sub_id):
                    continue # we only care about the node that is in the graph
                sub_node = self.graph_nodes[_sub_id]
                if len(sub_node["meta"]["label"]) == 1 and "OTHER" in sub_node["meta"]["label"] and "OTHER" not in main_node["meta"]["label"] \
                and sub_node["meta"]["text"].lower() not in ["it", "they","them","he","she","him","her","this","that","these","those","itself","themselves","himself","herself", "ANY"]:
                    continue

                if sub_node["meta"]["text"].lower() in  ["itself","themselves","himself","herself","them","they","it","he","she","her","him"]:
                    if sub_node["meta"]["text"].lower() in ["itself","themselves","himself","herself"]:
                        continue
                    if "verbs" in sub_node["meta"]:
                        verbs = sub_node["meta"]["verbs"]

                        if check_if_verb_is_strong(verbs):
                                # update the subnode with information from the main node
                                self.graph_nodes[_sub_id]["meta"]["label"] = main_node["meta"]["label"]
                                # self.graph_nodes[_sub_id]["meta"]["text"] = main_node["meta"]["text"]
                                
                                continue # we do not contract the node that is the object of strong verb

                if not (len(sub_node["meta"]["label"]) == 1 and "OTHER" in sub_node["meta"]["label"]):  
                    main_node["meta"]["label"].extend(sub_node["meta"]["label"])
                if "ids" not in main_node["meta"]:
                    main_node["meta"]["ids"] = list()
                    main_node["meta"]["ids"].append(main_id)
                if "ids" in sub_node["meta"]:
                    main_node["meta"]["ids"].extend(sub_node["meta"]["ids"])
                else:
                    main_node["meta"]["ids"].append(_sub_id) 
                if "texts" not in main_node["meta"]:
                    main_node["meta"]["texts"] = list()
                    if main_node["meta"]["text"].lower() not in ["it", "they","them","he","she","him","her","this","that","these","those","itself","themselves","himself","herself","ANY"]:
                        main_node["meta"]["texts"].append(main_node["meta"]["text"])
                    # main_node["meta"]["texts"].append(main_node["meta"]["text"])
                if "texts" in sub_node["meta"]:
                    main_node["meta"]["texts"].extend(sub_node["meta"]["texts"])
                else:
                    if sub_node["meta"]["text"].lower() not in ["it", "they","them","he","she","him","her","this","that","these","those","itself","themselves","himself","herself","ANY"]:
                        main_node["meta"]["texts"].append(sub_node["meta"]["text"]) # reserve thr text infor from the sub_node

                if self.graph.has_node(main_id) and  self.graph.has_node(_sub_id ):
                    print("contracting ", main_id, _sub_id)
                    self.graph_nodes[_sub_id]["contracted"] = 1
                    self.graph = nx.contracted_nodes(self.graph, main_id, _sub_id, self_loops = False) #now contract sub_id to main_id
            main_node["meta"]["label"] = list(set(main_node["meta"]["label"]))
            self.graph_nodes[main_id] = main_node #update the main node
        # we need to coref first before we now if "it" mention the actor or not
        for k, v in self.graph_nodes.items():
            if v["contracted"] == 0:
                if v["meta"]["text"].lower() in ["they","it","any"]:
                    self.graph_nodes[k]["meta"]["label"] = ["ACTOR"]

    def _find_main_coref_updated_version(self, chain):
        if chain is None:
            return False
        for c in chain:
            if len(c) > 1:
                continue  #this is the case , 2 entities (Peter and his wife)  refer to one entity (they), we do not care about this case at the moment
            token = c[0]
            sent_index = token["sent_index"]
            mention_index = token["mention_index"]
            id = sent_index*1000 + mention_index
            if token["token_pos"] in ["PROPN"]: #Proper noun, this is the case with the name of the person/ group
                if self.graph.has_node(id):
                    return token # we only care about the entity that is in the graph
                else:
                    continue
        for c in chain:
            if len(c) > 1:
                continue
            token = c[0]
            sent_index = token["sent_index"]
            mention_index = token["mention_index"]
            id = sent_index*1000 + mention_index
            if token["token_pos"] in ["NOUN"]: # the first noun phrase in the chain
                if self.graph.has_node(id):
                    return token
                else:
                    continue
        for c in chain:
            if len(c) > 1:
                continue
            token = c[0]
            sent_index = token["sent_index"]
            mention_index = token["mention_index"]
            id = sent_index*1000 + mention_index
            if self.graph.has_node(id):
                return token
    
    def draw(self, image_path: str = "") -> figure:
        fig_size = math.ceil(math.sqrt(self.graph.number_of_nodes())) * 10
        plt.subplots(figsize=(fig_size, fig_size))  # Todo: re-consider the figure size.

        graph_pos = nx.spring_layout(self.graph, scale=2)

        nx.draw_networkx_nodes(self.graph,
                                   graph_pos, # nodelist=[node.id for node in filter(lambda n: n.type == label, self.attackNode_dict.values())],
                                   node_size=100,
                                   alpha=0.6)
        nx.draw_networkx_labels(self.graph,
                                graph_pos,
                                labels={node: self._node_to_str(self.graph_nodes[node]) for node, nodedata in self.graph.nodes.items()},
                                verticalalignment='top',
                                horizontalalignment='left',
                                font_color='blue',
                                font_size=6)
        nx.draw_networkx_edges(self.graph, graph_pos)

        nx.draw_networkx_edge_labels(self.graph,
                                     graph_pos,
                                     font_color='red',
                                     edge_labels=nx.get_edge_attributes(self.graph, 'verb'),
                                     font_size=6)

        if image_path == "":
            plt.show()
        else:
            try:
                plt.savefig(image_path)
            except:
                return

    

    def _node_to_str(self, node_info):
        node_dict = node_info["meta"]
        if "text" not in node_dict:
            return ""
        text = node_dict["text"]

        if "label" in node_dict and len(node_dict["label"]) > 0:
                node_type = node_dict["label"][0]
                return f"{text} \n ({node_type.upper()})"

        return text
        


    def generate_edge_dict(self):
        self.graph_edges = dict()
        for e in list(self.graph.edges.data()):
            source_node_id = e[0]
            dest_node_id = e[1]
            if source_node_id not in self.graph_nodes or dest_node_id not in self.graph_nodes:
                continue
            verb = e[2]["verb"]
            index = e[2]["index"]
            edge_id =str(source_node_id)+"_"+ str(dest_node_id)
            edge_ = {"source": source_node_id, "dest": dest_node_id, "index":index , "verb":verb, "id": edge_id}
            self.graph_edges[edge_id] = edge_
        if len(self.graph_edges) == 0:
            if len(self.graph_nodes) >= 2:
                source_node_id = list(self.graph_nodes.keys())[0]
                dest_node_id = list(self.graph_nodes.keys())[1]
                edge_id =str(source_node_id)+"_"+ str(dest_node_id)
                verb = "any"
                edge_ = {"source": source_node_id, "dest": dest_node_id, "index":0 , "verb":verb, "id": edge_id}
                self.graph_edges[edge_id] = edge_

    def regenerate_graph_nodes(self):
        """ due to coref handling, some nodes are contracted and no used anymore, this function will remove those nodes from the graph_nodes dict
        """
        nodes = list(self.graph.nodes)
        _graph_nodes = self.graph_nodes.copy()
        _graph_nodes = dict(sorted(_graph_nodes.items()))
        # flag = False
        for k, v in self.graph_nodes.items():
            if _graph_nodes[k]["meta"]["text"].lower() in ["it", "they", "any"]:
                _graph_nodes[k]["meta"]["label"] = ["ACTOR"]
            if "texts" in v["meta"]:
                _graph_nodes[k]["meta"]["texts"] = list(set(_graph_nodes[k]["meta"]["texts"]))
            if "label" in v["meta"]:
                _graph_nodes[k]["meta"]["label"] = list(set(_graph_nodes[k]["meta"]["label"]))
            if "verbs" in v["meta"]:
                _graph_nodes[k]["meta"]["verbs"] = list(set(_graph_nodes[k]["meta"]["verbs"]))
            # if v["meta"]["text"] in malwares:
            #     if k in _graph_nodes:
            #         if "texts" not in _graph_nodes[k]["meta"]:
            #             _graph_nodes[k]["meta"]["texts"] = list()
            #             _graph_nodes[k]["meta"]["texts"].append(v["meta"]["text"])
            #         _graph_nodes[k]["meta"]["texts"].append("malicious script")
            if not self.is_campaign:
                _text = v["meta"]["text"]
                if re.search(purpose_pattern, _text, re.IGNORECASE):
                    if k in _graph_nodes:
                            del _graph_nodes[k]
                            continue

            if k not in nodes and k in _graph_nodes: # this node is not in the graph anymore, it is contracted, we delete it from nodes dict
                del _graph_nodes[k]
            if v["meta"]["text"] == "ANY" and "texts" not in v["meta"] and k in _graph_nodes:
                # if self.is_campaign:
                del _graph_nodes[k] # delete the node that is not used anymore
                # else:
                #     _graph_nodes[k]["meta"]["text"] = "attacker"
                #     _graph_nodes[k]["meta"]["label"] = ["ACTOR"]
            # if not self.is_campaign:
            #     if v["contracted"] == 1:
            #         if k in _graph_nodes:
            #             del _graph_nodes[k]
        if not self.is_campaign:
            keys = list(_graph_nodes.keys())
            if len(keys) > 1:
                key = keys[0]
                if _graph_nodes[key]["meta"]["text"] in malwares:
                    _graph_nodes[key]["meta"]["label"]= ["ACTOR"]
                    _graph_nodes[key]["meta"]["sub_label"]= ["MALWARE"]
        if self.is_campaign:
            if 0 not in _graph_nodes:
                flag = False
                for k,v in _graph_nodes.items():
                    if "ACTOR" in v["meta"]["label"]:
                        flag = True
                        break
                if not flag: # if there is no actor in the graph, we add pseudo actor
                    _graph_nodes[0] = {"id": 0, "contracted": 0, "meta": {"label": ["ACTOR"], "text": "attacker", "id": 0, "sent_index": 0, "type": "subject"}}
        
        self.graph_nodes = _graph_nodes


        
    def get_edge_info(self, source_id, dest_id):
        edge = str(source_id)+"_"+str(dest_id)
        if edge not in self.graph_edges:
            return None
        else:
            return self.graph_edges[edge]

 

    def replace_special_entity(self, start_id = 0):
        text=self.text
        # self.procedure_special_phrases = []    
        patterns = common_fixing_pattern["entity"]
        mapper = {}
        for p in patterns: 
            matches = re.finditer(p, text)
            entities = [m.group(0).strip() for m in matches]
            for entity in entities:
                real_entity = entity.replace("<code>","").replace("</code>","").replace("`","")
                # temp_entity = real_entity
                # if temp_entity.endswith("()"):
                #     temp_entity = temp_entity.replace("()","")
                # if temp_entity.endswith("("):
                #     temp_entity = temp_entity[:-1]
                # if temp_entity.startswith("\""):
                #     temp_entity = temp_entity.replace("\"","")
                # self.procedure_special_phrases.append(temp_entity)
                if len(real_entity.split()) > 1:
                    replacement = "ENTITY" + str(start_id)
                    start_id += 1
                    mapper[replacement] = {"ID": replacement,"phrase": real_entity}
                    text = text.replace(entity, replacement, 1)
                else:
                    text = text.replace(entity, real_entity, 1)
        values = list(mapper.values())
        replacement = {}
        if len(values) > 0:
            result = heuristic_extract_(pd.DataFrame(values))
            for s in result:
                replacement[s["ID"]] = {"text": s["text"], "label": s["label"]}
        
        rs =replace_special_entities(text)        
        for rs_ in rs:
            # if "\\" not in rs_[0] and "/" not in rs_[0]:
            #  continue
            flag = False
            if rs_[3] == "REGISTRY":
                id_= "REGISTRY" + str(start_id)
                flag = True
            if rs_[3] == "DIRECTORY":
                id_= "DIRECTORY" + str(start_id)
                flag = True
            if rs_[3] == "DATA":
                id_= "DATA" + str(start_id)
                flag = True
            if rs_[3] == "NETWORK":
                id_= "NETWORK" + str(start_id)
                flag = True
            if rs_[3] == "FUNCTION":
                id_= "FUNCTION" + str(start_id)
                flag = True
            if rs_[3] == "ENCRYPTION":
                id_= "ENCRYPTION" + str(start_id)
                flag = True
            # if flag:
            #     print()
            start_id += 1
            text = text.replace(rs_[0], id_, 1)
            replacement[id_] = {"text": rs_[0].strip(), "label": rs_[3]}

        return text, replacement


    def simplify_graph1(self):
        #we do not simplify if graph has less than 3 nodes
        if len(self.graph_nodes) <= 3:
            return
        # print("simplify graph, add all mentioned actors to the main node")
        main_id = -1
        for k,v in self.graph_nodes.items():
            if "ACTOR" in v["meta"]["label"] and len(v["meta"]["label"]) == 1  and v["meta"]["text"].lower() not in ["it", "they", "any"]:
                main_id = k
                break
        if main_id == -1:
            return
        main_node = self.graph_nodes[main_id]
        for k,v in self.graph_nodes.items():
            if "ACTOR" in v["meta"]["label"] and len(v["meta"]["label"]) == 1 and k != main_id:
                _sub_id = k
                sub_node = self.graph_nodes[_sub_id]
                main_node["meta"]["label"].extend(sub_node["meta"]["label"])               
                if "texts" not in main_node["meta"]:
                    main_node["meta"]["texts"] = list()
                    
                    main_node["meta"]["texts"].append(main_node["meta"]["text"])
                if "texts" in sub_node["meta"]:
                    main_node["meta"]["texts"].extend(sub_node["meta"]["texts"])
                else:
                    if sub_node["meta"]["text"].lower() not in ["it", "they","any"]:
                        main_node["meta"]["texts"].append(sub_node["meta"]["text"]) # reserve thr text infor from the sub_node
#update the main node
                if self.graph.has_node(main_id) and  self.graph.has_node(_sub_id ):
                    print("contracting ", main_id, _sub_id)
                    self.graph_nodes[_sub_id]["contracted"] = 1
                    self.graph_nodes[main_id]["contracted"] = 0
                    self.graph = nx.contracted_nodes(self.graph, main_id, _sub_id, self_loops = False) #  
        main_node["meta"]["label"] = list(set(main_node["meta"]["label"]))
        self.graph_nodes[main_id] = main_node 

    def get_true_text(self, text= None, texts = None):
        assert text is not None or texts is not None
        if text is not None:
            if text in self.replacement_mapper:
                text = self.replacement_mapper[text]["text"].lower()
            # text = text.replace("a ","").replace("an ","").replace("the ","").replace("A ","").replace("An ","").replace("The ","")
            if text.lower() in  ["it", "they"]:
                text = "attacker"
            return text
        else:
            return [self.get_true_text(text = t) for t in texts]
    
    def get_phrases(self):
        phrases = []
        for node in self.graph_nodes.values():
            node = node["meta"]
            if "texts" in node:
                _phrase = self.get_true_text(texts = node["texts"])
                phrases.extend(_phrase)
            else:
                _phrase = [self.get_true_text(text = node["text"])]
                phrases.extend(_phrase)
        self.phrases = list(set(phrases))


    def simplify_graph2(self):
        # if len(self.graph_nodes) <= 3:
        #     return
        # simplify the edge that has verb in ["is", "name", "call"]
        edges = [edge for edge in self.graph.edges(data= True)]
        for edge in edges:
            source = edge[0]
            target = edge[1]
            verb = edge[2]["verb"]
            flag = edge[2]["flag"]
            if flag == 1:
                main_node = self.graph_nodes[source]
                sub_node = self.graph_nodes[target]
                if main_node["meta"]["text"].lower() in ["itself", "themselves", "himself", "herself"]:
                    continue
                main_node["meta"]["label"].extend(sub_node["meta"]["label"])               
                if "texts" not in main_node["meta"]:
                    main_node["meta"]["texts"] = list()
                    
                    main_node["meta"]["texts"].append(main_node["meta"]["text"])
                if "texts" in sub_node["meta"]:
                    main_node["meta"]["texts"].extend(sub_node["meta"]["texts"])
                else:
                    if sub_node["meta"]["text"].lower() not in ["it", "they"]:
                        main_node["meta"]["texts"].append(sub_node["meta"]["text"]) # reserve thr text infor from the sub_node
#update the main node
                if self.graph.has_node(source) and  self.graph.has_node(target ):
                    # print("contracting ", source, target)
                    self.graph_nodes[target]["contracted"] = 1
                    self.graph = nx.contracted_nodes(self.graph, source, target, self_loops = False)  
                main_node["meta"]["label"] = list(set(main_node["meta"]["label"]))
                self.graph_nodes[source] = main_node
        # print("testing simplify graph 2")
                
    def conjuntion_simplification(self):
        _chains = []
        for sent_index in range(0, len(self.sentences)):
            sent = self.sentences[sent_index]
            chains = sent.chains
            for chain in chains:
                _chain = []
                for c in chain:
                    _chain.append(c + sent_index*1000)
                _chains.append(_chain)
        print()
        for chain in _chains:
            main = chain[0]
            try:
                main_node = self.graph_nodes[main]
            except:
                continue
            for c in chain[1:]:
                try:
                    sub_node = self.graph_nodes[c]
                except:
                    continue
                if self.graph.has_node(main) and  self.graph.has_node(c ):
                    main_node["meta"]["label"].extend(sub_node["meta"]["label"])               
                    if "texts" not in main_node["meta"]:
                        main_node["meta"]["texts"] = list()                  
                        main_node["meta"]["texts"].append(main_node["meta"]["text"])
                    if "texts" in sub_node["meta"]:
                        main_node["meta"]["texts"].extend(sub_node["meta"]["texts"])
                    else:
                        if sub_node["meta"]["text"].lower() not in ["it", "they"]:
                            main_node["meta"]["texts"].append(sub_node["meta"]["text"])
                    if "verbs" in sub_node["meta"]:
                        if "verbs" not in main_node["meta"]:
                            main_node["meta"]["verbs"] = list()
                        main_node["meta"]["verbs"].extend(sub_node["meta"]["verbs"])

                    self.graph_nodes[c]["contracted"] = 1
                    self.graph = nx.contracted_nodes(self.graph, main, c, self_loops = False)