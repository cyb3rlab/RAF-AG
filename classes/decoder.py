from classes.campaign import Campaign
import collections
from keys import Keys
import itertools
import json
import statistics
from modules import *
from classes.cosine_similarity import CosineSimilarity
from mitre_attack import *
procedure_mapper = proID_techID
from language_models import *
from mitre_attack import MitreAttack
must_be_focus= r"\b(email|keylogger|privilege|credential)"
def phrase_ranking(phrase):
    # if re.search(must_be_focus, phrase):
    #     return 0.5
    if phrase.lower() in ["any", "anyone", "anything", "anywhere","that","this","these","those"]:
        return 4
    doc = nlp(phrase)
    tokens = [t for t in doc]
    if len(tokens) == 1 and tokens[0].pos_ == "PRON":
        return 4 #proper noun
    flag = False
    for t in tokens:
        if t.pos_ == "PROPN":
            flag = True
            break
        for c in t.text:
            if c.isupper():
                flag = True
                break
    if flag:
        return 1
    return 2 
def is_acceptable(tech:dict, remember:list):
    flag = True
    if len(remember) == 0:
        return flag
    if "." in tech["techID"]:
        return flag # this is a detail technique, we may keep it
    for remembered in remember:
        if abs(tech["location"] - remembered["location"]) >5 :#too far away, ignore
            continue
        if tech["techID"].split(".")[0] == remembered["techID"].split(".")[0]:#same technique tree
            flag = False
            break
    
    return flag
def heuristic_criteria_localization(mapper, entity = "dest", flag = True, order_ids = [], technnique_name:str = None, consine_object:CosineSimilarity= None):
    technnique_name = technnique_name.replace("_"," ").lower()
    #this will analyze the list of phrase and rank them based on their specificity
    ranking = []
    if len(mapper) == 1:
        return max(order_ids)
    for i in range(0, len(mapper)):
        dest = mapper[i][entity]
        source = mapper[i]["source"]
        if len(dest["label"]) == 1 and "REGISTRY" in dest["label"]:
            ranking.append((i,0)) #best priority
            continue
        if len(dest["label"]) == 1 and "VULNERABILITY" in dest["label"]:
            ranking.append((i,0)) #best priority
            continue
        if "verbs" in source and "verbs" in dest:
            verbs1 = source["verbs"]
            verbs2 = dest["verbs"]
            if not check_strong_verb_similarity_mismatch(verbs1, verbs2):
                ranking.append((i,0.2))
                continue
        if "verbs" in dest:
            verbs = dest["verbs"]
            if check_if_verb_is_strong(verbs,strong_verb_group = ["delete","mimic","schedule","reboot","hide","prevent","encode compress","decode","strong execute","exfiltrate","user_action"]):
                ranking.append((i,0.5))#second best priority
                continue
        if len(dest["label"]) == 1 and "DATA" in dest["label"]:
            ranking.append((i,2)) #we treat data as normal entity
            continue

        if "texts" in dest:
            texts = dest["texts"]
            rank_scores = [phrase_ranking(text) for text in texts]
            rank_score = min(rank_scores)
            ranking.append((i,rank_score))
        else:
            text = dest["text"]
            rank_score =  phrase_ranking(text)
            ranking.append((i,rank_score))
    ranking = sorted(ranking, key=lambda x: x[1])
    rank_scores = [r[1] for r in ranking]
    if len(set(rank_scores)) == 1: #all same ranking
        final_index = max(order_ids) # chose the largest order id
        return final_index
    min_score = ranking[0][1] # there are different ranks, we will choose the best one
    top_ranking_ = [order_ids[r[0]] for r in ranking if r[1] == min_score]# if there are multiple phrases with the same rank, we will collect them all
    # new_ranks = []
    # for t in top_ranking_:
    #     dest = mapper[t]["dest"]
    #     phrase = dest["text"]
    #     if "verbs" in dest:
    #         phrase = dest["verbs"][0] + " " + phrase
    #     cosine_similarity = consine_object.get_similarity_raw(technnique_name, phrase)
    #     new_ranks.append((t,cosine_similarity))
    # new_ranks = sorted(new_ranks, key=lambda x: x[1], reverse=True)
    # top_index = new_ranks[0][0]
    # final_index = order_ids[top_index]
    final_index = max(top_ranking_) #among top best ranks, we will choose the one that appear later
    return final_index
def relative_ranking(input_:float, spanning:list=[]):
    distance = 1
    spanning =list(set(spanning))
    #the aim is to punish the input if it spans too far away from the mean
    if len(spanning) == 0:
        distance = 1.5
    else:
        # min_ = min(spanning)
        # fmean = statistics.fmean(spanning)
        # max_ = max(spanning)
        distance = statistics.pstdev(spanning)
        if distance < 1:
            distance = 1
    
    return input_/distance

def check_path(path):
    for i in range(0, len(path)-1):
        j = i+1
        if not check_pairs(path[i], path[j]):
            return False
    return True
def check_pairs(source, dest):
    source_tac = tech_tac_mapper[source]
    dest_tac = tech_tac_mapper[dest]
    combinations = itertools.product(source_tac, dest_tac)
    for c in combinations:
        id = c[0] + "__" + c[1]
        if id in heuristic_tactic_combinations:
            return True
    return False
def _best_similarity( v:list):
        _temp = sorted(v, key=lambda x: x[1], reverse=True)
        if len(_temp) == 0:
            return None
        if _temp[0][1] < Keys.MATCHING_THRESHOLD:
            return None
        return procedure_mapper[_temp[0][0]]["tech_id"]
def top_best_similarity_with_threshold( v:list, threshold:float = Keys.MATCHING_THRESHOLD,similar_procedure = True):
        new_v = []
        if similar_procedure:
            for item in v:
                new_v.append(item)
                id_ = item[0]
                score = item[1]
                if id_ in similar_mapper:
                    for k,v_ in similar_mapper[id_].items():
                        new_score = score * v_
                        new_v.append((k,new_score))
        _temp = sorted(new_v, key=lambda x: x[1], reverse=True)
        if len(_temp) == 0:
            return None
        rs = [item[0] for item in _temp if item[1] >= threshold]
        return rs
def top_best_similarity( v:list, top_k = 2, similar_procedure = True):
        #We will add similar procedure to the list of procedure
        new_v = []
        if similar_procedure:
            for item in v:
                new_v.append(item)
                id_ = item[0]
                score = item[1]
                if id_ in similar_mapper:
                    for k,v_ in similar_mapper[id_].items():
                        new_score = score * v_
                        new_v.append((k,new_score))
        else:
            new_v = v
        _temp = sorted(new_v, key=lambda x: x[1], reverse=True)
        if len(_temp) == 0:
            return None
        rs = [item[0] for item in _temp if item[1] >= Keys.MATCHING_THRESHOLD]
        if top_k == -1:
            return rs
        if top_k < len(rs):
            return rs[0:top_k]
        else:
            return rs
        
from networkx import DiGraph
import networkx as nx
def refined_path(path):
    # we do not want duplated techniques inside a path?
    visited_techiques = []
    new_path = []
    for p in path:
        if "Root" in p:
            continue
        id_ = p.split("__")[1]
        if proID_techID[id_]["tech_id"].split(".")[0] not in visited_techiques:
            visited_techiques.append(proID_techID[id_]["tech_id"].split(".")[0])
            new_path.append(proID_techID[id_]["tech_id"])
    return new_path

def refined_path2(path):
    flag = False
    visited_tactics = []
        # we do not want duplated techniques inside a path?
    visited_techiques = []
    new_path = []
    for p in path:
        if "Root" in p:
            continue
        id_ = p.split("__")[1]
        tech_id = proID_techID[id_]["tech_id"]
        tactic_ids = tech_tac_mapper[tech_id]
        if "TA0008" in tactic_ids: # if there is "lateral movement" tactic, we allows duplicated techniques after that
            flag = True
        #check if there is new tactic appear
        if len(set(tactic_ids).intersection(set(visited_tactics)))!= len (tactic_ids) or flag == True:
            visited_tactics.extend(tactic_ids)
            new_path.append(proID_techID[id_]["tech_id"])
    return new_path
class Decoder():
    @classmethod
    def pure_decoding(cls, mapper: dict,top_k = 10): 
        #this function is only for showing result
        result = {}
       
        mapper = collections.OrderedDict(sorted(mapper.items()))
        new_mapper = {}
        for k, v in mapper.items():
            mapper[k] = top_best_similarity(v, top_k = top_k, similar_procedure = False)
        
        for k, v in mapper.items():
            if k not in new_mapper:
                new_mapper[k] = []
            for item in v:
                new_mapper[k].append(procedure_mapper[item])
        for k, v in new_mapper.items():
            new_mapper[k] = list(set(v))
        return new_mapper

    @classmethod
    def remove_PRE(cls, techniques):
        rs = []
        for t in techniques:
            platform =  MitreAttack.get_platforms(t)
            if len(platform) == 1 and platform[0] == "PRE":
                continue
            rs.append(t)
        if len(rs) == 0:
            return techniques
        return rs

    @classmethod
    def uniqueness_ensuring(cls, mapper: dict, criteria = "heuristic"):
        cosine_object = CosineSimilarity()
        #change the default value of the mapper to meet criteria
 
        temp= {}
        for k, v in mapper.items():
            for v_ in v:

                if criteria == "min":
                    index = min(v_["order_ids"])
                if criteria == "max":
                    index = max(v_["order_ids"])
                if criteria == "heuristic":
                    index = heuristic_criteria_localization(v_["phrases"], order_ids = v_["order_ids"],consine_object=cosine_object, technnique_name =MitreAttack.get_technique_name(v_["techID"]))
                v_["location"] = index
                if index not in temp:
                    temp[index] = []
                temp[index].append(v_)
        mapper = temp
        #remove procedure examples that span too large area
        new_mapper = {}
        for k, v in mapper.items():
            for item in v:
                # stdv = statistics.pstdev(item["order_ids"])
                # if stdv >5 and item["value"] < 0.95:
                #     continue
                if k not in new_mapper:
                    new_mapper[k] = []
                new_mapper[k].append(item)      

        mapper = new_mapper

        #count the apperance of each technique at each location, key is technique id

        new_mapper2 = {}
        for k,v in mapper.items():
            unique = []
            v_list = []
            for item in v:
                if item["techID"] not in new_mapper2:
                    new_mapper2[item["techID"]] = []
                new_mapper2[item["techID"]].append(item)

        print()
        # sort and find the crowded location
        new_mapper3 = {}
        for k,v in new_mapper2.items():
            total = 0
            total_weight = 0
            for item in v:
                total += item["value"]* item["location"]
                total_weight += item["value"]
            mean = total/total_weight
            v2 = sorted(v, key=lambda x: abs(x["location"]-mean), reverse=False)
            min_distance = abs(v2[0]["location"]-mean)
            v3 = [v_ for v_ in v2 if abs(v_["location"]-mean) == min_distance]
            if len(v3) == 1:
                new_mapper3[k] = v3[0]
            else:
                v4 =sorted(v3, key=lambda x: relative_ranking(x["value"],x["order_ids"]), reverse=True)
                new_mapper3[k] = v4[0]
        print()
        # only keep the best procedure for each technique at each location 
        # good_mapper = {}
        # for k,v in mapper.items():
        #     for item in v:
        #         if item["techID"] in new_mapper2 and k == new_mapper2[item["techID"]][0] and len(new_mapper2[item["techID"]]) == 1:
                    
        #             if item["techID"] not in good_mapper:
        #                 good_mapper[item["techID"]] = []
        #             good_mapper[item["techID"]].append(item)
        
        # for k,v in good_mapper.items():
        #     good_mapper[k] = sorted(v, key=lambda x: x["value"], reverse=True)
        #     good_mapper[k] = good_mapper[k][0:1]


        # bad_tech = [k for k,v in new_mapper2.items() if len(v) > 1]
        # bad_mapper = {}
        # for k,v in mapper.items():
        #     for item in v:
        #         if item["techID"] in bad_tech and k in new_mapper2[item["techID"]]:
        #             if item["techID"] not in bad_mapper:
        #                 bad_mapper[item["techID"]] = []
        #             bad_mapper[item["techID"]].append(item)
                
        
        # for k, v in bad_mapper.items():
        #     bad_mapper[k] = sorted(v, key=lambda x: relative_ranking(x["value"],x["order_ids"]), reverse=True)
        #     bad_mapper[k] = bad_mapper[k][0:1]         
        # for k,v in bad_mapper.items():
        #     good_mapper[k] = v
        # print()



        # print() # now we will consider the best location for each teachniques that are not in good location

        # print()
        # for k, v in new_mapper.items():
        #     new_mapper[k] = v[0:1]
        # print()
        good_mapper = new_mapper3
        location_mapper = {}
        for k,v in good_mapper.items():
            key = v["location"]
            if key not in location_mapper:
                location_mapper[key] = []
            location_mapper[key].append(v)
        
        location_mapper = collections.OrderedDict(sorted(location_mapper.items()))
        print()
        return location_mapper

    @classmethod
    def attack_path_decoding(cls, procedure_alignment_mapper: dict,topk = 1, matching_threshold = Keys.DECODING_MATCHING_THRESHOLD, relax:bool = True, criteria="heuristic",recode=True, tech_alignment_mapper:dict= None):
        mapper = procedure_alignment_mapper
        original_mapper = mapper.copy()
        mapper = {int(k): v for k, v in mapper.items() if len(v) > 0}
        temp = {}
        for k,v in mapper.items():
            v_ = []
            for item in v:
                if item["techID"] == "T1041":
                    print()
                fmean = statistics.fmean(item["order_ids"])
                sent_std= statistics.pstdev(item["sent_indexes"])
                min_id = min(item["order_ids"])
                if abs(min_id - fmean) >= 10 or sent_std >= 2.0:# under 3 sentences
                    continue
                if item['value'] >= 1.0:
                    item['value'] = 1.0
                if item['value'] < matching_threshold:
                    continue
                v_.append(item)
            temp[k] = v_

        mapper = temp
        mapper = collections.OrderedDict(sorted(mapper.items()))
        for k, v in mapper.items():
            mapper[k] = sorted(v, key=lambda x: x['value'], reverse=True)
        
        mapper = Decoder.uniqueness_ensuring(mapper, criteria)
        
        new_mapper = {}
        count = 0
        # length_threshold = math.floor(len(mapper)/3)
        for k,v in mapper.items():
            if len(v) == 0:
                continue
            count += 1
            if k not in new_mapper:
                new_mapper[k] = []
            for item in v:
                # fmean = statistics.fmean(item["order_ids"])
                # if abs(item['min_id'] - fmean) >= 10:
                #     continue #remove item that has span too large, these are not good sub-graphs since they connect cybersecurity entities too far away
                
                platform =  MitreAttack.get_platforms(item['techID'])
                if len(platform) == 1 and platform[0] == "PRE":# we are bringing the PRE technique to the begining of the attack path
                        if 0 not in new_mapper:
                            new_mapper[0] = []
                        new_mapper[0].append(item)
                        continue
                new_mapper[k].append(item)
        # print()
        #ensure unique techniques
        # uniques = []
        # mapper = new_mapper.copy()
        # new_mapper = {}
        # for k, v in mapper.items():
        #     if len(v) == 0:
        #         continue
        #     if k not in new_mapper:
        #         new_mapper[k] = []
        #     if len(v) == 1:
        #         new_mapper[k].append(v[0])
        #         uniques.append(v[0]['techID'])
        #         continue
        #     for item in v:
        #         # if  item[1] < 0.95:
        #         #     continue
        #         tech_id = item['techID']
        #         if tech_id in uniques:
        #             continue
        #         uniques.append(tech_id)
        #         new_mapper[k].append(item)
        unique_tech = [item["techID"]  for k,v in new_mapper.items() for item in v]
        if tech_alignment_mapper and matching_threshold != 1.0:
            for k,v in tech_alignment_mapper.items():
                k = int(k)
                if k == 0 and k in new_mapper:
                    continue# do not add pre to already added 
                if k not in new_mapper:
                    new_mapper[k] = []
                
                v[0]["value"] = float(v[0]["value"])
                new_mapper[k].append(v[0])
        #best decoding
        techniques = []
        mapper = new_mapper.copy()
        mapper = collections.OrderedDict(sorted(mapper.items()))
        for k, v in mapper.items():
            mapper[k] = sorted(v, key=lambda x: x['value'], reverse=True)
        best_mapper = {}
        indexes = list(mapper.keys())
        for i in range(0, len(indexes)):
            technique_list = mapper[indexes[i]]
            if len(technique_list) == 0:
                continue
            best_tech= technique_list[0]
            remembered_tech = []
            for j in range(i-3, i): # we will look back 3 steps
                if j < 0:
                    continue
                if indexes[j] in best_mapper:
                    remembered_tech.append(best_mapper[indexes[j]][0])
            if is_acceptable(best_tech, remembered_tech):
                best_mapper[indexes[i]] = [best_tech]
            else:
                if len(technique_list) >= 2:
                    best_mapper[indexes[i]] = [technique_list[1]]

        # for k, v in mapper.items():
        #     if len(v) == 0:
        #         continue
        #     if v[0]["value"] >= matching_threshold:
        #         best_mapper[k] = v[0:1]
        #     else:
        #         best_mapper[k] = []


        #top k decoding 
        topk_mapper = {}
        for k, v in mapper.items():
            if len(v) == 0:
                continue
            _v= []
            for i in range(0, len(v)):
                if i < topk:
                    _v.append(v[i])
            topk_mapper[k] = _v
            _good_list = []
            for item in topk_mapper[k]:
                if item["value"] >= matching_threshold:
                    _good_list.append(item)
            topk_mapper[k] = _good_list
        # unique_tech_topk = [item["techID"]  for k,v in topk_mapper.items() for item in v]
        # unique_tech_best = [item["techID"] for k,v in best_mapper.items() for item in v ]
        # if tech_alignment_mapper:
        #     for k,v in tech_alignment_mapper.items():
        #         k = int(k)
        #         if k not in topk_mapper and v[0]["techID"] not in unique_tech_topk:
        #             topk_mapper[k] = []
        #         if  v[0]["techID"] not in unique_tech_topk and len(topk_mapper[k]) <= topk:
        #             topk_mapper[k].append(v[0])
        #         if k not in best_mapper and v[0]["techID"] not in unique_tech_best:
        #             best_mapper[k] = []
        #         if v[0]["techID"] not in unique_tech_best and len(best_mapper[k]) == 0 :
        #             best_mapper[k].append(v[0])
        # topk_mapper = collections.OrderedDict(sorted(topk_mapper.items()))
        # best_mapper = collections.OrderedDict(sorted(best_mapper.items()))
        final_path = list()
        if relax:
            new_mapper2 = topk_mapper
        else:
            new_mapper2 = best_mapper
        for k,v in new_mapper2.items():
            if len(v) == 0:
                continue
            if len(v) == 1:
                final_path.append(v[0]["techID"])
            if len(v) >= 2:
                source = v[0]
                target = v[1]
                if source["techID"] == target["techID"]:
                    final_path.append(source["techID"])
                    continue
                source_order_ids = source["order_ids"]
                target_order_ids = target["order_ids"]
                source_mean_id = statistics.mean(source_order_ids)
                target_mean_id = statistics.mean(target_order_ids)
                if source_mean_id <= target_mean_id:
                    final_path.append(source["techID"])
                    final_path.append(target["techID"])
                else:

                    final_path.append(target["techID"])
                    final_path.append(source["techID"])
        unique_final_path = []
        for p in final_path:
            if p not in unique_final_path:
                unique_final_path.append(p)
        topK_for_evaluation = {}
        for k,v in best_mapper.items():
            topK_for_evaluation[k] = [item["techID"] for item in v]
        if len(topK_for_evaluation) < 10 and recode:
            best_mapper, topk_mapper, unique_final_path, topK_for_evaluation = Decoder.attack_path_decoding(original_mapper, topk = topk, matching_threshold = matching_threshold - 0.05, relax = True, criteria=criteria, recode=False, tech_alignment_mapper=tech_alignment_mapper)
        
        return best_mapper, topk_mapper, unique_final_path, topK_for_evaluation
    
    
    

    @classmethod
    def platform_connection(cls, association_data, platform_data):
        return None


   