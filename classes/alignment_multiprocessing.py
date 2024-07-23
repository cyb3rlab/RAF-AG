#since python can not use multi cores for multithreading, that's why we need to use multiprocessing
from classes.campaign import Campaign
from classes.big_campaign import BigCampaign
from classes.procedure import Procedure
import statistics
import re
import Levenshtein
from modules import *
import os
import itertools
import networkx as nx
import math
from classes.cosine_similarity import CosineSimilarity
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import List
from keys import Keys
NUM_PROCESS = Keys.NUM_PROCESSES
NUM_PROCEDURES_PER_PROCESS = Keys.NUM_WORK_PER_PROCESS  # number of procedure per per process
import multiprocessing as mp
# bert_similarity = None
procedure_mapper = proID_techID
global bert_similarity
try:    
    bert_similarity = CosineSimilarity.from_pickle(os.path.join(Keys.CONTEXT_SIMILARITY_PATH,"all.pkl"))
except:
    bert_similarity = CosineSimilarity()
def check_texts_similarity_simple(texts1:list, texts2:list):
    flag = False
    for t1 in texts1:
        re_pattern  = r"\b" + t1 + r"\b"
        for t2 in texts2:
            try:
                if re.search(re_pattern, t2, re.IGNORECASE):
                    flag = True
                    return flag
            except:
                continue
       
    return flag
def calculate_spanningarea(sim_value,mapper:dict):
    data = []
    for k,v in mapper.items():
        if v:
            data.append(v[0]//1000)
        else:
            return 0.0
    std= statistics.pstdev(data)
    if std == 0:
        return sim_value
    return sim_value/std
def get_procedure_phrases( procedures:dict):
        procedure_phrases = []
        for k,v in procedures.items():
            procedure_phrases.extend(v.phrases)
        return list(set(procedure_phrases))
def alignment_(campaign: Campaign, procedure:Procedure, technique):
        features = technique.features
        rs = {}
        if not hasattr(campaign, "nodeID_2_order") :
            campaign.nodeID_2_order = {}
            node_ids = list(campaign.graph_nodes.keys())
            node_ids = sorted(node_ids)
            for i in range(len(node_ids)):
                campaign.nodeID_2_order[node_ids[i]] = i
        if len(procedure.graph_nodes) == 0 or len(procedure.graph_edges) == 0:
            return
        sub_graph = list(campaign.graph_nodes.keys())

        max_value, max_combination = Alignment.graph_alignment(campaign,sub_graph, procedure)
        #localize where the procedure is in the campaign
        # id_ = Alignment._get_id(max_combination, procedure, features)
        # if id_ != -1 and max_value > Keys.MATCHING_THRESHOLD:
        #     order_id = campaign.nodeID_2_order[id_]
        #     if order_id not in rs:
        #         rs[order_id] = []
        #     print(f"procedure id : {procedure.id} \n max_value: {max_value} \n order_id : {order_id}")
        #     rs[order_id].append((procedure.id, max_value))
        alignment_rs = Alignment.alignment_localization(max_combination, procedure, features, campaign.nodeID_2_order,campaign= campaign)
        if alignment_rs != -1 and max_value > Keys.MATCHING_THRESHOLD:
            index = alignment_rs[0]
            if index not in rs:
                rs[index] = []
            print(f"procedure id : {procedure.id} \n max_value: {max_value} \n order_id : {index}")
            # mapper = {}
            # for k,v in max_combination.items():
            #     if v is None:
            #         continue
            #     _recoreded_data = campaign.graph_nodes[v[0]]["meta"]
            #     mapper[procedure.graph_nodes[k]["meta"]["text"]] = _recoreded_data
            tech_name = technique.tech_name  
            record= {"id": procedure.id, "techID":procedure_mapper[procedure.id], "tech_name": tech_name , "value": max_value,"min_id":index, "order_ids": alignment_rs[1], "sent_indexes": alignment_rs[2], "combine_ids": alignment_rs[3], "phrases": alignment_rs[4]}
            rs[index].append(record)
        if alignment_rs!= -1 and len(alignment_rs[1]) < 2:
            labels = []
            for k,v in procedure.graph_nodes.items():
                labels.extend(v["meta"]["label"])
            labels = [l for l in labels if l != "OTHER"]
            labels = list(set(labels))
            if "ACTOR" in labels and "USER" in labels and len(labels) == 2:
                return None #only actor and user, this is not enough to make a decision
        if alignment_rs!= -1 and len(procedure.sentences) == 1 and len(alignment_rs[1]) < 2:
            # the sentence is complicated but the procedure is too simple, this is the indication of the wrong match
            if "if" in procedure.text or "when " in procedure.text or "where " in procedure.text or "how " in procedure.text or "why " in procedure.text or "who " in procedure.text:
                return None
        return rs

def alignment_with_range(campaign: Campaign, procedures: List[Procedure], technqiques: dict, bert_BERT_path:str = None, testing:bool = False):
    # print("alignment with range- multiprocessing")
    if Keys.MULTI_PROCESSING:
        if Keys.BERT_SIM_ENABLE and bert_BERT_path is not None:
            global bert_similarity
            bert_similarity = CosineSimilarity.from_pickle(bert_BERT_path)
    rs = {}
    # if testing:

    for procedure in procedures:
        if len(procedure.graph_nodes) <2:
            continue
        rs_ = alignment_(campaign, procedure, technqiques[procedure.special_id])
        if rs_ is None:
            continue
        for k,v in rs_.items():
            if k not in rs:
                rs[k] = list()
            rs[k].extend(v)
    # print("alignment with range- multiprocessing-this process is done")
    return rs

class Alignment():
    @classmethod
    def get_sent_distance(cls,source_node, dest_node):
        distances = []
        if "ids" in source_node["meta"]:
            source_ids = source_node["meta"]["ids"]
        else:
            source_ids = [source_node["id"]]
        if "ids" in dest_node["meta"]:
            dest_ids = dest_node["meta"]["ids"]
        else:
            dest_ids = [dest_node["id"]]
        for source_id in source_ids:
            for dest_id in dest_ids:
                sent_source_index = source_id // 1000
                sent_dest_index = dest_id // 1000
                sent_diff = abs(sent_source_index - sent_dest_index)
                id_diff = abs(source_id - dest_id)
                distance2 = sent_diff
                distance3 = 0.0
                if sent_diff == 0:
                    if id_diff > 10:
                            distance3 = 0.1
                    if id_diff > 20:
                        distance3 = 0.15
                FACTOR = Keys.DISTANCE_FACTOR_PER_SENTENCE
                if "USER" in source_node["meta"]["label"] or "USER" in dest_node["meta"]["label"]:
                    FACTOR *= 0.5 # tolerate the distance between user and other nodes
                distance = 1 + FACTOR * distance2  + distance3
                distances.append(distance)
        return min(distances)
    @classmethod
    def procedure_graph_alignment(cls, procedure2: Procedure, sub_graph:list, procedure1: Procedure):
        global bert_similarity
        if Keys.BERT_SIM_ENABLE and bert_similarity is None:
            bert_similarity = CosineSimilarity()
        if Keys.BERT_SIM_ENABLE:
            bert_similarity.compute_range(procedure1.phrases, procedure2.phrases)

        campaign = procedure2
        procedure = procedure1

        nodeID_2_order = {}
        node_ids = list(campaign.graph_nodes.keys())
        for i in range(len(node_ids)):
            nodeID_2_order[node_ids[i]] = i

        edge_alignment = {}
        node_alignment, node_matrix = Alignment.node_alignment(campaign, sub_graph, procedure, procedure_similarity=True)
        #k_list is the list of nodes in procedure
        #prevent combination explosion
        for k,v in node_alignment.items():
            if len(v)>= 2:
                v = sorted(v, key=lambda x: x[1], reverse=True)
            if len(v)>30:           
                node_alignment[k] = v[:30]
        k_list = list(node_alignment.keys())
        #v_list is the list, each element is a list of similar nodes in campaign
        v_list = list(node_alignment.values())
        #generate all the posible subgraph combinations in campaign
        counter = 0
        max_value =0.0
        max_combination  = None
        # print("combinations: ", len(combinations))
         # mapper will map a node_id in procedure to a (similar)node_id in campaign
        for c in itertools.product(*v_list):
            counter += 1
            if counter > 20000:
                break # too many combinations, we only consider the first 10000 combinations
            # length of each c is the same as the number of keys (k_list) in the node_alignment
            _mapper = {} 
            for i in range(0, len(k_list)):
                if c[i] == -1:
                    # node i in this combination c, does not have a similar node in campaign
                    _mapper[k_list[i]] = None
                else:
                    # k_list[i] is the node id in procedure
                    # c[i] is the node id in campaign
                    _mapper[k_list[i]] = c[i]
            for k,v in procedure.graph_edges.items():
                # we will check each edge in procedure
                # we get edge source and dest node
                procedure_source= v["source"]
                procedure_dest = v["dest"]
                if "verbs" in procedure.graph_nodes[procedure_dest]["meta"]:
                    verbs = procedure.graph_nodes[procedure_dest]["meta"]["verbs"]
                else:
                    verb = v["verb"]
                    if "verbs" in v:
                        verbs = v["verbs"]
                    else:
                        verbs = [verb]
                verbs = list(set(verbs))
                verb_similarity = None
                if procedure_source not in procedure.graph_nodes or procedure_dest not in procedure.graph_nodes:
                    continue
                try:
                    if _mapper[procedure_source] is None or _mapper[procedure_dest] is None:
                        # None meaning that this procedure node does not have a similar node in campaign
                        edge_alignment[k] = 0.0
                        continue
                except:
                    edge_alignment[k] = 0.0
                    continue
                # we find the similar node in campaign for the source and the dest
                campaign_source = _mapper[procedure_source][0]
                campaign_dest = _mapper[procedure_dest][0]
                # we calculate the distance between the source and the dest in campaign

                try:
                        distance1 = nx.shortest_path_length(campaign.graph, campaign_source , campaign_dest)
                except:
                        #except mean that there is no path between source and dest
                        distance1 = 100
                verbs_2 = []
                if campaign.graph_nodes[campaign_dest]["meta"]["type"]== "object":
                    if "verbs" in campaign.graph_nodes[campaign_dest]["meta"]:
                        verbs_2 = list(set(campaign.graph_nodes[campaign_dest]["meta"]["verbs"]))

                else:
                        verbs_2 = []
                                # edge = None
                                # id2 = str(campaign_source) + "_" + str(campaign_dest)
                                # id2_ = str(campaign_dest) + "_" + str(campaign_source)
                                # if id2 in campaign.graph_edges:
                                #     edge = campaign.graph_edges[id2]
                                # else:
                                #     if id2_ in campaign.graph_edges:
                                #         edge = campaign.graph_edges[id2_]
                                # if edge is not None:            
                                #     verb_2 = edge["verb"]
                                #     if "verbs" in edge:
                                #         verbs_2 = edge["verbs"]
                                #     else:
                                #         verbs_2 = [verb_2]
                                # else:
                if len(verbs) > 0 and len(verbs_2) ==0:
                    if check_if_verb_is_strong(verbs):
                            verb_similarity = Keys.VERB_DIFF_SEVERVE_PUNISHMENT
                    else:
                        if len(procedure.graph_nodes) == 2:
                            verb_similarity = Keys.VERB_DIFF_PUNISHMENT #punish more
                        if len(procedure.graph_nodes) > 2:
                            verb_similarity = Keys.VERB_DIFF_SOFT_PUNISHMENT
                if len(verbs) > 0 and len(verbs_2)> 0 and check_verbs_similarity(verbs, verbs_2): # apply verb similarity checking
                    if len(procedure.graph_nodes) == 2:
                        verb_similarity = 1.0 # this is a very good case 
                    else:
                        verb_similarity = 1.0
                    # for v in imporant_verbs:
                    #     if v in verbs and v in verbs_2:
                    #         verb_similarity = 1.5
                if len(verbs) > 0 and len(verbs_2) > 0 and verb_similarity is None:
                                        #in this case, we have verbs information but they are very different, we need to pusnish the similarity
                        if len(procedure.graph_nodes)==2:
                            verb_similarity = Keys.VERB_DIFF_PUNISHMENT
                        else:
                            if check_strong_verb_similarity_mismatch(verbs, verbs_2):
                                verb_similarity = Keys.VERB_DIFF_SEVERVE_PUNISHMENT
                            else:
                                verb_similarity = Keys.VERB_DIFF_SOFT_PUNISHMENT

                distance = distance1
                if distance > 1:                
                    distance2 = Alignment.get_sent_distance(campaign.graph_nodes[campaign_source], campaign.graph_nodes[campaign_dest])
                    distance = min(distance1,distance2)
                # # we relaxed the distance since the coref between two nodes is not perfect
                # # resulting in a large distance or even no path between two nodes
                if (("ACTOR" in procedure.graph_nodes[procedure_source]["meta"]["label"] and len(procedure.graph_nodes[procedure_source]["meta"]["label"]) ==1 ) \
                    or \
                    ( "ACTOR" in procedure.graph_nodes[procedure_dest]["meta"]["label"] and len(procedure.graph_nodes[procedure_dest]["meta"]["label"]) ==1)):
                    # or procedure.graph_nodes[procedure_source]["meta"]["text"] in malwares or procedure.graph_nodes[procedure_dest]["meta"]["text"] in malwares:

                    distance = min(Keys.ACTOR_TOLERATE_DISTANCE, distance)
                    if ("VULNERABILITY" in procedure.graph_nodes[procedure_source]["meta"]["label"] and len(procedure.graph_nodes[procedure_source]["meta"]["label"])==1) \
                    or ("VULNERABILITY" in procedure.graph_nodes[procedure_dest]["meta"]["label"] and len(procedure.graph_nodes[procedure_dest]["meta"]["label"])==1):
                        distance = 1.0
                    if ("REGISTRY" in procedure.graph_nodes[procedure_source]["meta"]["label"] and len(procedure.graph_nodes[procedure_source]["meta"]["label"])==1) \
                    or ("REGISTRY" in procedure.graph_nodes[procedure_dest]["meta"]["label"] and len(procedure.graph_nodes[procedure_dest]["meta"]["label"])==1):
                        distance = 1.0
                if campaign_source == campaign_dest: # avoid distance = 0
                    distance = 1
                #we extract the similarity value between the sources (one from procedure and one from campaign)
                source_similarity_value = node_matrix[procedure_source][campaign_source]
                #we extract the similarity value between the dests (one from procedure and one from campaign)
                dest_similarity_value = node_matrix[procedure_dest][campaign_dest]
                # we calculate the similarity value between the edge in procedure and the edge in campaign
                edge_similarity_value = math.sqrt(source_similarity_value * dest_similarity_value) / distance
                # if source_similarity_value >=1.0 and dest_similarity_value >=1.0 and campaign_source != campaign_dest:
                #     edge_similarity_value = math.sqrt(edge_similarity_value)
                
                edge_alignment[k] = edge_similarity_value
                if verb_similarity is not None:
                    #if the verb is different, we decrease the edge similarity value
                    if edge_similarity_value > verb_similarity and verb_similarity < 1.0:
                        edge_alignment[k] = math.sqrt(edge_similarity_value * verb_similarity)
                        #punishment
                    # if the verb is the same, we increase the edge similarity value
                    if verb_similarity >= 1.0:
                        edge_alignment[k] = math.sqrt(edge_similarity_value * verb_similarity)
                    if edge_alignment[k] > 1.0:
                        edge_alignment[k] = 1.0 # we save the edge similarity value into a dictionary
            #now we calculate the similarity value between the procedure and its similar subgraph in campaign
            _sub_graph_similarity = Alignment._sub_graph_alligment_score(_mapper, edge_alignment, procedure)
            if _sub_graph_similarity >= max_value:
                # save the best subgraph
                max_value = _sub_graph_similarity
                max_combination = _mapper
        return max_value, max_combination
    @classmethod
    def graph_alignment(cls,campaign:Campaign, sub_graph:list , procedure: Procedure):
        all_result = []
        edge_alignment = {}
        node_alignment, node_matrix = Alignment.node_alignment(campaign, sub_graph, procedure)
        #k_list is the list of nodes in procedure
        #prevent combination explosion
        for k,v in node_alignment.items():
            if len(v)>= 2:
                v = sorted(v, key=lambda x: x[1], reverse=True)
            if len(v)>30:           
                node_alignment[k] = v[:30]
        k_list = list(node_alignment.keys())
        #v_list is the list, each element is a list of similar nodes in campaign
        v_list = list(node_alignment.values())
        #generate all the posible subgraph combinations in campaign
        counter = 0
        max_value =0.0
        max_combination  = None
        # print("combinations: ", len(combinations))
         # mapper will map a node_id in procedure to a (similar)node_id in campaign
        for c in itertools.product(*v_list):
            counter += 1
            if counter > 20000:
                break # too many combinations, we only consider the first 20000 combinations
            # length of each c is the same as the number of keys (k_list) in the node_alignment
            _mapper = {} 
            for i in range(0, len(k_list)):
                if c[i] == -1:
                    # node i in this combination c, does not have a similar node in campaign
                    _mapper[k_list[i]] = None
                else:
                    # k_list[i] is the node id in procedure
                    # c[i] is the node id in campaign
                    _mapper[k_list[i]] = c[i]
            for k,v in procedure.graph_edges.items():
                # we will check each edge in procedure
                # we get edge source and dest node
                procedure_source= v["source"]
                procedure_dest = v["dest"]
                if "verbs" in procedure.graph_nodes[procedure_dest]["meta"]:
                    verbs = procedure.graph_nodes[procedure_dest]["meta"]["verbs"]
                else:
                    verb = v["verb"]
                    if "verbs" in v:
                        verbs = v["verbs"]
                    else:
                        verbs = [verb]
                verbs = list(set(verbs))
                verb_similarity = None
                if procedure_source not in procedure.graph_nodes or procedure_dest not in procedure.graph_nodes:
                    continue
                try:
                    if _mapper[procedure_source] is None or _mapper[procedure_dest] is None:
                        # None meaning that this procedure node does not have a similar node in campaign
                        edge_alignment[k] = 0.0
                        continue
                except:
                    edge_alignment[k] = 0.0
                    continue
                # we find the similar node in campaign for the source and the dest
                campaign_source = _mapper[procedure_source][0]
                campaign_dest = _mapper[procedure_dest][0]
                # we calculate the distance between the source and the dest in campaign

                try:
                        distance1 = nx.shortest_path_length(campaign.graph, campaign_source , campaign_dest)
                except:
                        #except mean that there is no path between source and dest
                        distance1 = 100
                verbs_2 = []
                if campaign.graph_nodes[campaign_dest]["meta"]["type"]== "object":
                    if "verbs" in campaign.graph_nodes[campaign_dest]["meta"]:
                        verbs_2 = list(set(campaign.graph_nodes[campaign_dest]["meta"]["verbs"]))

                else:
                        verbs_2 = []
                                # edge = None
                                # id2 = str(campaign_source) + "_" + str(campaign_dest)
                                # id2_ = str(campaign_dest) + "_" + str(campaign_source)
                                # if id2 in campaign.graph_edges:
                                #     edge = campaign.graph_edges[id2]
                                # else:
                                #     if id2_ in campaign.graph_edges:
                                #         edge = campaign.graph_edges[id2_]
                                # if edge is not None:            
                                #     verb_2 = edge["verb"]
                                #     if "verbs" in edge:
                                #         verbs_2 = edge["verbs"]
                                #     else:
                                #         verbs_2 = [verb_2]
                                # else:
                if len(verbs) > 0 and len(verbs_2) ==0:
                    if check_if_verb_is_strong(verbs):
                            verb_similarity = Keys.VERB_DIFF_SEVERVE_PUNISHMENT
                    else:
                        if len(procedure.graph_nodes) == 2:
                            verb_similarity = Keys.VERB_DIFF_PUNISHMENT #punish more
                        if len(procedure.graph_nodes) > 2:
                            verb_similarity = Keys.VERB_DIFF_SOFT_PUNISHMENT
                if len(verbs) > 0 and len(verbs_2)> 0 and check_verbs_similarity(verbs, verbs_2): # apply verb similarity checking
                    if len(procedure.graph_nodes) == 2:
                        verb_similarity = 1.0 # this is a very good case 
                    else:
                        verb_similarity = 1.0
                    # for v in imporant_verbs:
                    #     if v in verbs and v in verbs_2:
                    #         verb_similarity = 1.5
                if len(verbs) > 0 and len(verbs_2) > 0 and verb_similarity is None:
                                        #in this case, we have verbs information but they are very different, we need to pusnish the similarity
                        if len(procedure.graph_nodes)==2:
                            verb_similarity = Keys.VERB_DIFF_PUNISHMENT
                        else:
                            if check_strong_verb_similarity_mismatch(verbs, verbs_2):
                                verb_similarity = Keys.VERB_DIFF_SEVERVE_PUNISHMENT
                            else:
                                verb_similarity = Keys.VERB_DIFF_SOFT_PUNISHMENT

                distance = distance1
                if distance > 1:                
                    distance2 = Alignment.get_sent_distance(campaign.graph_nodes[campaign_source], campaign.graph_nodes[campaign_dest])
                    distance = min(distance1,distance2)
                # # we relaxed the distance since the coref between two nodes is not perfect
                # # resulting in a large distance or even no path between two nodes
                if (("ACTOR" in procedure.graph_nodes[procedure_source]["meta"]["label"] and len(procedure.graph_nodes[procedure_source]["meta"]["label"]) ==1 ) \
                    or \
                    ( "ACTOR" in procedure.graph_nodes[procedure_dest]["meta"]["label"] and len(procedure.graph_nodes[procedure_dest]["meta"]["label"]) ==1)):
                    # or procedure.graph_nodes[procedure_source]["meta"]["text"] in malwares or procedure.graph_nodes[procedure_dest]["meta"]["text"] in malwares:

                    distance = min(Keys.ACTOR_TOLERATE_DISTANCE, distance)
                    if ("VULNERABILITY" in procedure.graph_nodes[procedure_source]["meta"]["label"] and len(procedure.graph_nodes[procedure_source]["meta"]["label"])==1) \
                    or ("VULNERABILITY" in procedure.graph_nodes[procedure_dest]["meta"]["label"] and len(procedure.graph_nodes[procedure_dest]["meta"]["label"])==1):
                        distance = 1.0
                    if ("REGISTRY" in procedure.graph_nodes[procedure_source]["meta"]["label"] and len(procedure.graph_nodes[procedure_source]["meta"]["label"])==1) \
                    or ("REGISTRY" in procedure.graph_nodes[procedure_dest]["meta"]["label"] and len(procedure.graph_nodes[procedure_dest]["meta"]["label"])==1):
                        distance = 1.0
                if campaign_source == campaign_dest: # avoid distance = 0
                    distance = 1
                #we extract the similarity value between the sources (one from procedure and one from campaign)
                source_similarity_value = node_matrix[procedure_source][campaign_source]
                #we extract the similarity value between the dests (one from procedure and one from campaign)
                dest_similarity_value = node_matrix[procedure_dest][campaign_dest]
                # we calculate the similarity value between the edge in procedure and the edge in campaign
                edge_similarity_value = math.sqrt(source_similarity_value * dest_similarity_value) / distance
                # if source_similarity_value >=1.0 and dest_similarity_value >=1.0 and campaign_source != campaign_dest:
                #     edge_similarity_value = math.sqrt(edge_similarity_value)
                
                edge_alignment[k] = edge_similarity_value
                if verb_similarity is not None:
                    #if the verb is different, we decrease the edge similarity value
                    if edge_similarity_value > verb_similarity and verb_similarity < 1.0:
                        edge_alignment[k] = math.sqrt(edge_similarity_value * verb_similarity)
                        #punishment
                    # if the verb is the same, we increase the edge similarity value
                    if verb_similarity >= 1.0:
                        edge_alignment[k] = math.sqrt(edge_similarity_value * verb_similarity)
                    if edge_alignment[k] >= 1.0:
                        edge_alignment[k] = 1.0
                
                     # we save the edge similarity value into a dictionary
            #now we calculate the similarity value between the procedure and its similar subgraph in campaign
            _sub_graph_similarity = Alignment._sub_graph_alligment_score(_mapper, edge_alignment, procedure)
            if _sub_graph_similarity >= 1.0:
                _sub_graph_similarity = 1.0
            if _sub_graph_similarity >= max_value:
                # save the best subgraph
                max_value = _sub_graph_similarity
                max_combination = _mapper
                temp_mapper = {k:v for k,v in _mapper.items() if not (len(procedure.graph_nodes[k]["meta"]["label"])==1 and "ACTOR" in procedure.graph_nodes[k]["meta"]["label"])}
                if _sub_graph_similarity > Keys.MATCHING_THRESHOLD:
                    all_result.append((_sub_graph_similarity,temp_mapper, _mapper))
        if len(all_result) == 0:
            return max_value, max_combination
        all_result = sorted(all_result, key=lambda x: x[0], reverse=True)
        max_similarity = all_result[0][0]
        toprs = [rs for rs in all_result if (rs[0] == max_similarity or rs[0] > 0.95)] #0.95 means very good similarity
        if len(toprs) == 1:
            return max_similarity, max_combination
        else:
            toprs = sorted(toprs, key=lambda x: calculate_spanningarea(x[0], x[1]), reverse=True)
            return toprs[0][0], toprs[0][2]# 
        return max_value, max_combination
    @classmethod
    def node_alignment(cls, campaign: Campaign, sub_graph: list, procedure: Procedure, procedure_similarity= False):
        """ for each node in procedure, calculate the similarity between it and all nodes in campaign
            if sim_value > threshold, save these similar nodes into a dictionary
            dict[node_in_procedure] = [(node_in_campaign, sim_value), ...]
        """
        _node_similarity_score = dict()
        _node_similarity_matrix = dict()
        for k,v in procedure.graph_nodes.items():
            node2 = v["meta"] # node2 is from procedure

            if k not in _node_similarity_score:
                    _node_similarity_score[k] = list()
            if k not in _node_similarity_matrix:
                    _node_similarity_matrix[k] = dict()
            for node_id in sub_graph:
                node1 = campaign.graph_nodes[node_id]["meta"] # node1 is from campaign
                sim_value = Alignment._node_similarity_calculation(node1, node2, campaign, procedure, procedure_similarity)    
                if sim_value > 1.0:
                    sim_value = 1.0
                if sim_value > Keys.NODE_SIMILARITY_THRESHOLD:
                    _node_similarity_score[k].append((node_id, sim_value))
                    _node_similarity_matrix[k][node_id] = sim_value
            # if len(_node_similarity_score[k])== 0:
            #     if node2["text"] in malwares:
            #         node2["label"]= ["ACTOR"] # try to treat malware as attacker
            #         for node_id in sub_graph:
            #             node1 = campaign.graph_nodes[node_id]["meta"] # node1 is from campaign
            #             sim_value = Alignment._node_similarity_calculation(node1, node2, campaign, procedure, procedure_similarity)    
            #             if sim_value > Keys.NODE_SIMILARITY_THRESHOLD:
            #                 _node_similarity_score[k].append((node_id, sim_value))
            #                 _node_similarity_matrix[k][node_id] = sim_value
            if len(_node_similarity_score[k])== 0:
                    _node_similarity_score[k]= [-1]
        
        return _node_similarity_score, _node_similarity_matrix
    


    @classmethod
    def _node_similarity_calculation(cls, node1: dict, node2: dict, campaign: Campaign, procedure: Procedure, procedure_similarity= False):
        labels1 = node1["label"].copy()
        labels2 = node2["label"].copy()
        intersect_ = list(set(labels1).intersection(set(labels2)))
        label_similarity = 0
        if len(intersect_) > 0:
            label_similarity = Keys.LAMDA
        else:
            label_similarity = Keys.SOFT_LAMDA # If types are not matched, soft lamda is used
        if procedure_similarity and len(intersect_) == 0:
            return 0.0 # strict punishment for the similarity between different types in procedure alignment
        if len(intersect_) == 1:
            if intersect_[0] == "OTHER":
                label_similarity = Keys.SOFT_LAMDA
        if (("VULNERABILITY" in node1["label"] and len(node1["label"])==1 ) or ("VULNERABILITY" in node2["label"] and len(node2["label"])==1)) and len(intersect_) == 0:
            return 0.0 # pusnish the similarity between vulnerability and other types
        if (("ACTOR" in node1["label"] and len(node1["label"])==1 and node1["text"].lower() not in ["it","they"]) or ("ACTOR" in node2["label"] and len(node2["label"])==1 and node2["text"].lower() not in ["it","they"])) and len(intersect_) == 0:
            return 0.0
        if (("USER" in node1["label"] and len(node1["label"])==1 ) or ("USER" in node2["label"] and len(node2["label"])==1)) and len(intersect_) == 0:
            return 0.0


        if "texts" in node1:
            list1 = campaign.get_true_text(texts = node1["texts"])
        else:
            list1 = [campaign.get_true_text(text = node1["text"])]
        # some special text "abc.docx" is not similar to "document" by in fact the same thing
        if "texts" in node2:
            list2 = procedure.get_true_text(texts = node2["texts"])
        else:
            list2 = [procedure.get_true_text(text = node2["text"])]

        if procedure_similarity and "FUNCTION" in node1["label"] and "FUNCTION" in node2["label"]: # compare the similarity between two procedures
            set1 = set(list1)
            set2 = set(list2)
            malwares_1 = set1.intersection(set(malwares))
            malwares_2 = set2.intersection(set(malwares))
            if len(malwares_1) > 0 and len(malwares_2) > 0:
                return 1.0
    
        if Keys.BERT_SIM_ENABLE:
            context_similarity =  bert_similarity.get_similarity(list2, list1)
            # string_similarity = Alignment.get_stringSet_similarity(list1, list2)
            # max_similarity = 0.6 * context_similarity+ 0.4 * string_similarity #more weight on context similarity
            string_similarity = 0.0 # test without string levenshtein similarity
            max_similarity = max(context_similarity, string_similarity)
        else:
            context_similarity = 0.0
            string_similarity = Alignment.get_stringSet_similarity(list1, list2)
            max_similarity = max(context_similarity, string_similarity)
        
        return_value = label_similarity+ (1- label_similarity)* max_similarity # weighted average
        if return_value > 0.8:
            return return_value
        if ("ACTOR" in node1["label"] and len(node1["label"])==1 and node1["text"].lower() not in ["it","they"]) and ("ACTOR" in node2["label"] and len(node2["label"])==1 and node2["text"].lower() not in ["it","they"]):
            if return_value < 0.95:
                return 0.95 #scale up actor similarity, implicit actor in action should be tolerated
        if ("USER" in node1["label"] and len(node1["label"])==1 ) and ("USER" in node2["label"] and len(node2["label"])==1):
            if return_value < 0.8 and return_value > 0.5:
                return return_value + 0.2
        if ("VULNERABILITY" in node1["label"] and len(node1["label"])==1 ) and ("VULNERABILITY" in node2["label"] and len(node2["label"])==1):
            if return_value < 0.8 and return_value > 0.5:
                return return_value + 0.2
        if ("REGISTRY" in node1["label"] and len(node1["label"])==1 ) and ("REGISTRY" in node2["label"] and len(node2["label"])==1):
            if return_value < 0.8 and return_value > 0.5:
                return return_value + 0.2
        return return_value
    
    
    @classmethod
    def _accumulate_node_similarity(cls, nodes_mapper, procedure: Procedure):
        similarity = 0.0
        _check = 0
        # count = 0
        for k,v in nodes_mapper.items():
            if v is None:
                continue
            _id = v[0]
            if "ACTOR" in procedure.graph_nodes[k]["meta"]["label"] and len(procedure.graph_nodes[k]["meta"]["label"])==1:
                _check += 1
                continue
            if "OTHER" in procedure.graph_nodes[k]["meta"]["label"] and len(procedure.graph_nodes[k]["meta"]["label"])==1:
                _check += 1
                continue 
            # count += 1
            similarity += v[1]
        normalize_factor = len(procedure.graph_nodes) - _check
        if normalize_factor == 0:
            normalize_factor = 1
        return similarity / normalize_factor#normalize by the number of nodes in procedure
    
    
    @classmethod
    def _accumulate_edge_similarity(cls, edges_mapper, procedure: Procedure):
        if len(edges_mapper) == 0:
            return 0.0
        similarity = 0.0
        for k,v in edges_mapper.items():
            similarity += v 
        return similarity / len(edges_mapper) #normalize by the number of edges in procedure
    
    
    @classmethod
    def _sub_graph_alligment_score(cls, nodes_mapper, edges_mapper, procedure: Procedure):
        node_similarity = cls._accumulate_node_similarity(nodes_mapper, procedure)
        edge_similarity = cls._accumulate_edge_similarity(edges_mapper, procedure)
        similarity = (node_similarity + edge_similarity)/2
        return similarity
    
    
    @classmethod
    def get_stringSet_similarity(cls, set_m, set_n) -> float:
        C2_pattern = r"\bC2\b"
        max_similarity = 0.0
        flag = False
        set_m = [m.lower() for m in set_m]
        set_n = [n.lower() for n in set_n]
        for m in set_m:
            # m_split= m.split()
            # m =m.lower()
            for n in set_n:
                # n = n.lower()
                # if re.search(C2_pattern, m, re.IGNORECASE) is not None and re.search(C2_pattern, n, re.IGNORECASE) is not None:
                #     flag = True
                # if "keylog" in m and "keylog" in n: # special case for keylogging
                #     return 1.0
                # n_split = n.split()
                # if len(m_split) > len(n_split):
                #     focus_parts = m_split[len(m_split)-len(n_split):]
                #     focus_phrase = " ".join(focus_parts)
                #     sim1 = Levenshtein.ratio(focus_phrase, n)
                # else:
                #     focus_parts = n_split[len(n_split)-len(m_split):]
                #     focus_phrase = " ".join(focus_parts)
                #     sim1 = Levenshtein.ratio(m, focus_phrase)
                sim2 = Levenshtein.ratio(m, n)
                # similarity = (sim1 + sim2) / 2
                similarity = sim2
                if similarity > max_similarity:
                    max_similarity = similarity
                # max_similarity = max_similarity if max_similarity > similarity else similarity
        # if flag and max_similarity < 0.85: #scale up the similarity if both m and n contain C2
        #     return 0.85
        return max_similarity
    

    @classmethod
    def alignment_localization(cls, combination, procedure: Procedure, features: list = None, id2order:dict = None, campaign: Campaign = None):
        #this is used for locating the alignment, however, we only do simple heuristic here to prevent overhead during alignment
        # we record the meta data for future decoding
        if combination is None:
            return -1
        if len(features) == 0:
            return -1
        good_features = [f for f in features if f not in ["ACTOR", "OTHER"]]
        good_nodes = []
        sent_indexes = []
        combine_ids  = []
        mapper =  []
        # first = -1
        for k,v in combination.items():
            labels = procedure.graph_nodes[k]["meta"]["label"].copy()
            if v is None:
                continue
            _id = v[0]
            if "ACTOR" in labels and len(labels) == 1:
                continue
            if "OTHER" in labels and len(labels) == 1:
                continue
            # intersect_ = list(set(good_features).intersection(set(labels)))
            # if len(intersect_) > 0:
            good_nodes.append(id2order[_id]) # save all the node_id
            sent_indexes.append(campaign.graph_nodes[_id]["meta"]["sent_index"]) # save all the sent_index
            combine_ids.append(_id)
            _data = dict()
            _data["source"] = procedure.graph_nodes[k]["meta"]
            _data["dest"] = campaign.graph_nodes[_id]["meta"]
            mapper.append(_data)
                # if "texts" in campaign.graph_nodes[_id]["meta"]:
                #     phrases.extend(campaign.graph_nodes[_id]["meta"]["texts"])
                # else:
                #     phrases.append(campaign.graph_nodes[_id]["meta"]["text"])
        # for k,v in combination.items():
        #         labels = procedure.graph_nodes[k]["meta"]["label"].copy()
        #     # labels.extend(procedure.graph_nodes[k]["meta"]["heuristic"])
        #         if v is None:
        #             continue
        #         _id = v[0]
        #         if "ACTOR" in labels and len(labels) == 1:
        #             continue # w
        #         if "OTHER" in labels and len(labels) == 1:
        #             continue
        #         for g in good_features:
        #             if g in labels:
        #                 if first == -1:
        #                     first = id2order[_id]
        #                     break
        #         if first != -1:
        #             break
        # if first == -1 and len(good_nodes) > 0:
        #     first = min(good_nodes)
        if len(good_nodes) == 0:
            return -1
        # input_list = sorted(good_nodes)
        # middle = float(len(input_list))/2
        # if middle % 2 != 0:
        #     return input_list[int(middle - .5)]
        # else:
        #     return input_list[int(middle-1)]
        # return round(10 *statistics.fmean(good_nodes))
        return max(good_nodes), good_nodes , sent_indexes, combine_ids, mapper

            
    @classmethod
    def _get_id(cls, combination, procedure: Procedure, features: list = None):
        if combination is None:
            return -1
        if len(features) == 0:
            return -1
        good_features = [f for f in features if f not in ["ACTOR", "OTHER"]]
        good_nodes = []
        for k,v in combination.items():
            labels = procedure.graph_nodes[k]["meta"]["label"].copy()
            # labels.extend(procedure.graph_nodes[k]["meta"]["heuristic"])
            if v is None:
                continue
            _id = v[0]
            if "ACTOR" in labels and len(labels) == 1:
                continue # w
            if "OTHER" in labels and len(labels) == 1:
                continue
            intersect_ = list(set(good_features).intersection(set(labels)))
            if len(intersect_) > 0:
                good_nodes.append(_id)
            else:
                continue
        if len(good_nodes) == 0:
            return -1
        # input_list = sorted(good_nodes)
        # middle = float(len(input_list))/2
        # if middle % 2 != 0:
        #     return input_list[int(middle - .5)]
        # else:
        #     return input_list[int(middle-1)]
        return min(good_nodes)


    



    @classmethod
    def all_alignment_multiprocess(cls, campaign:Campaign, procedures: dict, techniques: dict):
        #preparing the bert similarity
        if Keys.MULTI_PROCESSING:
            if not hasattr(campaign, "bert_path"):
                bert_sim_path = os.path.join(Keys.CONTEXT_SIMILARITY_PATH, f"{campaign.id}.pkl")
            else:
                bert_sim_path = campaign.bert_path
        else:
            bert_sim_path = os.path.join(Keys.CONTEXT_SIMILARITY_PATH,"all.pkl")
        # if os.path.exists(bert_sim_path):         
        #     try:
        #         print("loading bert similarity")
        #         bert_similarity = CosineSimilarity.from_pickle(bert_sim_path)
        #     except:
        #         print("calculate bert similarity")   
        #         bert_similarity = CosineSimilarity()
        #         procedures_phrases = get_procedure_phrases(procedures)
        #         campaign_phrases = campaign.phrases
        #         bert_similarity.compute_range(procedures_phrases,campaign_phrases )
        #         bert_similarity.to_pickle(bert_sim_path)
        
        final_result = dict()
        futures = []
        keys = list(procedures.keys())
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESS) as executor:
            max_numer_of_procedures = len(procedures)
            for i in range(0, max_numer_of_procedures, NUM_PROCEDURES_PER_PROCESS):
                start = i
                end = i+NUM_PROCEDURES_PER_PROCESS if i+NUM_PROCEDURES_PER_PROCESS < max_numer_of_procedures else max_numer_of_procedures
                procedure_id_slice = keys[start:end]
                procedure_slice = [procedures[k] for k in procedure_id_slice]

                futures.append(executor.submit(alignment_with_range, campaign, procedure_slice, techniques, bert_sim_path))
            #wait for all processes to finish
            for result in as_completed(futures):
                rs = result.result()
                if rs is None:
                    continue
                for k,v in rs.items():
                    if k not in final_result:
                        final_result[k] = list()
                    final_result[k].extend(v)         
        campaign.mapper = final_result

    @classmethod
    def all_alignment_big_campaign_multiprocess(cls, bigcampaign:BigCampaign, procedures: dict, techniques: dict):
        #preparing the bert similarity
        if Keys.MULTI_PROCESSING:
            if not hasattr(bigcampaign, "bert_path"):
                bert_sim_path = os.path.join(Keys.CONTEXT_SIMILARITY_PATH, f"{bigcampaign.id}.pkl")
            else:
                bert_sim_path = bigcampaign.bert_path
        else:
            bert_sim_path = os.path.join(Keys.CONTEXT_SIMILARITY_PATH,"all.pkl")
        # if os.path.exists(bert_sim_path):         
        #     try:
        #         print("loading bert similarity")
        #         bert_similarity = CosineSimilarity.from_pickle(bert_sim_path)
        #     except:
        #         print("calculate bert similarity")   
        #         bert_similarity = CosineSimilarity()
        #         procedures_phrases = get_procedure_phrases(procedures)
        #         campaign_phrases = campaign.phrases
        #         bert_similarity.compute_range(procedures_phrases,campaign_phrases )
        #         bert_similarity.to_pickle(bert_sim_path)
        for campaign in bigcampaign.data:
            final_result = dict()
            futures = []
            keys = list(procedures.keys())
            with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESS) as executor:
                max_numer_of_procedures = len(procedures)
                for i in range(0, max_numer_of_procedures, NUM_PROCEDURES_PER_PROCESS):
                    start = i
                    end = i+NUM_PROCEDURES_PER_PROCESS if i+NUM_PROCEDURES_PER_PROCESS < max_numer_of_procedures else max_numer_of_procedures
                    procedure_id_slice = keys[start:end]
                    procedure_slice = [procedures[k] for k in procedure_id_slice]

                    futures.append(executor.submit(alignment_with_range, campaign, procedure_slice, techniques, bert_sim_path))
                #wait for all processes to finish
                for result in as_completed(futures):
                    rs = result.result()
                    if rs is None:
                        continue
                    for k,v in rs.items():
                        if k not in final_result:
                            final_result[k] = list()
                        final_result[k].extend(v)
            end = time.time()            
            campaign.mapper = final_result
        bigcampaign.mapper_gathering()
    # single process version for debugging
    @classmethod
    def all_alignment_sequential(cls, campaign:Campaign, procedures: dict, techniques: dict):
        #preparing the bert similarity
        bert_sim_path = os.path.join(Keys.CONTEXT_SIMILARITY_PATH, f"{campaign.id}.pkl")
        assert os.path.exists(bert_sim_path), "Bert similarity file does not exist"
        global bert_similarity
        bert_similarity = CosineSimilarity.from_pickle(bert_sim_path)

        
        final_result = dict()
        futures = []
        keys = list(procedures.keys())
        max_numer_of_procedures = len(procedures)
        for i in range(0, max_numer_of_procedures, NUM_PROCEDURES_PER_PROCESS):
                start = i
                end = i+NUM_PROCEDURES_PER_PROCESS if i+NUM_PROCEDURES_PER_PROCESS < max_numer_of_procedures else max_numer_of_procedures
                procedure_id_slice = keys[start:end]
                procedure_slice = [procedures[k] for k in procedure_id_slice]

                rs = alignment_with_range(campaign, procedure_slice, techniques, bert_sim_path)
                if rs is None:
                    continue
                for k,v in rs.items():
                    if k not in final_result:
                        final_result[k] = list()
                    final_result[k].extend(v)
        end = time.time()            
        campaign.mapper = final_result
    
    @classmethod
    def all_alignment_sequential_big_campaign_sequential(cls, bigcampaign:BigCampaign, procedures: dict, techniques: dict):
        #preparing the bert similarity

        bert_sim_path = os.path.join(Keys.CONTEXT_SIMILARITY_PATH,"all.pkl")
        assert os.path.exists(bert_sim_path), "Bert similarity file does not exist"

        print("loading bert similarity")
        global bert_similarity
        bert_similarity = CosineSimilarity.from_pickle(bert_sim_path)

        
        for campaign in bigcampaign.data:
            final_result = dict()
            keys = list(procedures.keys())
            max_numer_of_procedures = len(procedures)
            for i in range(0, max_numer_of_procedures, NUM_PROCEDURES_PER_PROCESS):
                start = i
                end = i+NUM_PROCEDURES_PER_PROCESS if i+NUM_PROCEDURES_PER_PROCESS < max_numer_of_procedures else max_numer_of_procedures
                procedure_id_slice = keys[start:end]
                procedure_slice = [procedures[k] for k in procedure_id_slice]

                rs = alignment_with_range(campaign, procedure_slice, techniques, bert_sim_path)
                if rs is None:
                    continue
                for k,v in rs.items():
                    if k not in final_result:
                        final_result[k] = list()
                    final_result[k].extend(v)          
            campaign.mapper = final_result
            
        bigcampaign.mapper_gathering()

    
    @classmethod
    def campaign_technique_alignment(cls, campaign:Campaign,  techniques: dict,id2order:dict):
        result = dict()
        uniques = []
        nodes = list(campaign.graph_nodes.values())
        for k,v in techniques.items():
            technique = v
            if len(technique.best_phrases) == 0:
                continue
            flag = False
            for node in nodes:
                id_ = node["id"]
                if "texts" in node["meta"]:
                    texts = node["meta"]["texts"]
                else:
                    texts = [node["meta"]["text"]]
                if check_texts_similarity_simple(technique.best_phrases, texts):
                    
                    node_id = id2order[id_]
                    if technique.id not in uniques:
                        temp_data = dict()
                        temp_data["techID"] = technique.id
                        temp_data["techName"] = technique.tech_name
                        temp_data["value"] = 1.0
                        temp_data["min_id"] = node_id
                        temp_data["location"] = node_id
                        temp_data["order_ids"] = [node_id]
                        temp_data["sent_indexes"] = [node["meta"]["sent_index"]]
                        uniques.append(technique.id)
                        if "TA0042" in technique.tactics or "TA0043" in technique.tactics:
                            node_id = 0 # move pre to the first
                        if node_id not in result:
                            result[node_id] = list()
                    
                        result[node_id].append(temp_data)
        return result
    @classmethod
    def bigcampaign_technique_alignment(cls, bigcampaign:BigCampaign, techniques: dict):
        
        start = 0
        for campaign in bigcampaign.data:
            tech_alginment_rs = dict()
            if not hasattr(campaign, "nodeID_2_order") :
                campaign.nodeID_2_order = {}
                node_ids = list(campaign.graph_nodes.keys())
                node_ids = sorted(node_ids)
                for i in range(len(node_ids)):
                    campaign.nodeID_2_order[node_ids[i]] = i
            rs = cls.campaign_technique_alignment(campaign, techniques, campaign.nodeID_2_order)
            for k,v in rs.items():
                if k not in tech_alginment_rs:
                    tech_alginment_rs[k] = list()

                tech_alginment_rs[k].extend(v)
            campaign.tech_alignment = tech_alginment_rs
        bigcampaign.tech_mapper_gathering()
        return bigcampaign.tech_alignment