from classes.campaign import Campaign
from classes.big_campaign import BigCampaign
from classes.procedure import Procedure
from classes.technique import Technique
from classes.alignment_multiprocessing import Alignment
from classes.cosine_similarity import CosineSimilarity

from classes.decoder import Decoder
from keys import Keys
import json
import jsonlines
import pandas as pd
import concurrent.futures
campaigns_dir = Keys.CAMPAIGN_PATH
campaigns_output_dir = campaigns_dir + "/output"
campaigns_image_dir = campaigns_dir + "/images"
campaigns_input_dir = campaigns_dir + "/input"
campaigns_procedure_alignment_dir = campaigns_dir + "/procedure_alignment"
campaigns_tech_alignment_dir = campaigns_dir + "/tech_alignment"
campaigns_decoding_result = campaigns_dir + "/decoding_result"
campaigns_sequence_techniques = campaigns_dir + "/sequence_techniques"
campaigns_bert= campaigns_dir + "/USE_cosine"
procedures_dir = Keys.PROCEDURE_PATH
procedures_output_dir = procedures_dir + "/output"
procedures_output_file = procedures_dir + "/analyzed_procedure.jsonl"
procedures_image_dir = procedures_dir + "/images"
procedures_text_dir = procedures_dir + "/txt"
procedure_input_dir = procedures_dir + "/input/procedures.csv"
procedure_deduplication_dir = procedures_dir + "/deduplication"
tech_dir = Keys.TECHNIQUE_PATH
tech_json_dir = tech_dir + "/json"
import timeit
from classes.decoder import Decoder
import os
import math
import pickle
import multiprocessing as mp
from modules import *
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
def get_procedure_phrases( procedures:dict):
        procedure_phrases = []
        for k,v in procedures.items():
            procedure_phrases.extend(v.phrases)
        return list(set(procedure_phrases))
def generate_procedure(procedures, i:int):
        tech_id = procedures.loc[i, "tech_id"]
        procedure_id = procedures.loc[i, "id"]
        text = procedures.loc[i, "description"]
        error_id = ""
        new_id = tech_id
        if Keys.IMAGE_GENERATION:
                image_path = os.path.join(procedures_image_dir, f"{procedure_id}.png")
        else:
                image_path = ""
        _procedure = Procedure(text=text,tech_id=tech_id,procedure_id = procedure_id, special_id= new_id, image_path=image_path)
        none_entity_nodes = _procedure.remove_none_entity_node()
        if len(_procedure.graph_nodes) == 0 or len(_procedure.graph_edges) == 0:
                error_id = _procedure.id
        else:
            try:
            
                _procedure.to_json(os.path.join(procedures_output_dir, _procedure.id + ".json"))
            except:
                error_id= _procedure.id
        # return none_entity_nodes, error_id

        
class Manager():
    def __init__(self, campaign_from_0 = True, procedure_from_0 = False, technique_from_0 = False,techniue_alignment_from_0 = True, matching_from_0 = True, multiprocessing = False,context_similarity_from0 = True, do_procedure_deduplication = False) -> None:
        self.procedures = dict()
        self.techniques = dict()
        is_knowledge_loaded = False
        # self.campaigns = []
        self.big_campaigns = []
        self.multiprocessing = multiprocessing
        time_recoder  = dict()
        # read campaigns from pure text file
        if campaign_from_0:
                print("read campaign from text")
                # self.analyze_campaign_from_text(campaigns_input_dir)
                # self.write_cp_to_json()
                time1 = timeit.default_timer()
                self.analyze_big_campaign(campaigns_input_dir)               #change to big campaign
                # self.write_big_cp_to_jsonl()
                time2 = timeit.default_timer()
                time_recoder["campaign_analyzing"] = time2 - time1

        else:
                # self.load_campaigns_from_json(campaigns_output_dir)
                self.load_big_campaigns_from_jsonl(campaigns_output_dir)

        if procedure_from_0:
                # if self.multiprocessing:
                #     print("read procedure from text in parallel")
                #     self.read_procedures_from_text_multiprocess(procedure_input_dir)
                # else:
                print("read procedure from text")
                # since we need to access GPU for bert, we need to read procedure one by one to be faster
                time1 = timeit.default_timer()
                self.analyze_procedures_from_text(procedure_input_dir)
                time2 = timeit.default_timer()
                time_recoder["procedure_analyzing"] = time2 - time1
                self.generate_procedure_jsonl(procedures_output_dir, procedures_output_file)
                self.load_procedures_from_json(load_from_jsonl = True)
                is_knowledge_loaded = True
                # self.write_pro_to_json()
                # we do not need to write because we already write in the analyze function


        else:
                if not os.path.exists(procedures_output_file):
                    self.generate_procedure_jsonl(procedure_deduplication_dir, procedures_output_file)
                self.load_procedures_from_json(load_from_jsonl = True)
       

        if do_procedure_deduplication:
            print("start procedure deduplication")
            time1 = timeit.default_timer()
            self.procedure_deduplication(saved_dir=procedure_deduplication_dir)
            time2 = timeit.default_timer()
            time_recoder["procedure_deduplication"] = time2 - time1
            self.generate_procedure_jsonl(procedure_deduplication_dir, procedures_output_file)
            self.load_procedures_from_json(load_from_jsonl = True)
            is_knowledge_loaded = True
        if technique_from_0:
                if len(self.procedures) == 0:
                        self.load_procedures_from_json(load_from_jsonl = True)
                print("create technique group and extract techniques features")
                self.materialize_tech(procedure_input_dir)
        else:
                self.load_techniques_from_json(tech_json_dir) #
        


        if context_similarity_from0:
            time1 = timeit.default_timer()
            self.generate_bert_object()
            time2 = timeit.default_timer()
            time_recoder["cosine_similarity"] = time2 - time1
        # self.compress_data()
        if techniue_alignment_from_0:
            if len(self.big_campaigns) == 0:
                # self.load_campaigns_from_json(campaigns_output_dir)
                self.load_big_campaigns_from_jsonl(campaigns_output_dir)
            if len(self.techniques) == 0:
                self.load_techniques_from_json(tech_json_dir)
            for campaign in self.big_campaigns:
                print("start aligning techniques for this campaign "+campaign.id)
                tech_alignmt_rs = Alignment.bigcampaign_technique_alignment(campaign, self.techniques)
                with open(os.path.join(campaigns_tech_alignment_dir, campaign.id + ".json"), "w") as f:
                    json.dump(tech_alignmt_rs, f, indent=4)
        # self.load_from_json() 
        print(f"==========>load {len(self.big_campaigns)} campaigns, \n==========> {len(self.procedures)} procedures,\n==========> {len(self.techniques)} techniques")
        if matching_from_0:
            if len(self.big_campaigns) == 0:
                # self.load_campaigns_from_json(campaigns_output_dir)
                self.load_big_campaigns_from_jsonl(campaigns_output_dir)
            if len(self.procedures) == 0 or not is_knowledge_loaded:
                self.load_procedures_from_json(load_from_jsonl = True)
            if len(self.techniques) == 0 or not is_knowledge_loaded:
                self.load_techniques_from_json(tech_json_dir)
            self.procedures = { k:v for k,v in self.procedures.items() if len(v.graph_nodes) > 1}
            print(f"==========>load {len(self.big_campaigns)} campaigns, \n==========> {len(self.procedures)} procedures,\n==========> {len(self.techniques)} techniques")
            print("start alignment")
            # self.procedure_matching()
            time1 = timeit.default_timer()
            self.big_procedure_matching()
            time2 = timeit.default_timer()
            time_recoder["graph_alignment"] = time2 - time1

        print("start decoding")
        time1 = timeit.default_timer()
        self.report_decoding()
        time2 = timeit.default_timer()
        time_recoder["decoding"] = time2 - time1
        with open("time_recoder.json", "w") as f:
            json.dump(time_recoder, f, indent=4)

    

    def generate_bert_object(self):
        if len(self.big_campaigns) > 0:
            if Keys.MULTI_PROCESSING:
                for campaign in self.big_campaigns:
                    bert_sim_path = os.path.join(campaigns_bert, f"{campaign.id}.pkl")
                    print("calculating bert similarity model")
                    bert_similarity = CosineSimilarity()
                    procedures_phrases = get_procedure_phrases(self.procedures)
                    campaign_phrases = campaign.phrases
                    bert_similarity.compute_range(procedures_phrases,campaign_phrases )
                    bert_similarity.to_pickle(bert_sim_path)
                    campaign.bert_path = bert_sim_path
            else: #no multiprocessing, we stack all of phrases into a big one
                bert_sim_path = os.path.join(campaigns_bert, f"all.pkl")
                if os.path.exists(bert_sim_path):
                    bert_similarity = CosineSimilarity.from_pickle(bert_sim_path)
                else:
                    bert_similarity = CosineSimilarity()
                procedures_phrases = get_procedure_phrases(self.procedures)
                campaign_phrases = []
                for campaign in self.big_campaigns:
                    campaign_phrases.extend(campaign.phrases)
                campaign_phrases = list(set(campaign_phrases))
                bert_similarity.compute_range(procedures_phrases,campaign_phrases )
                bert_similarity.to_pickle(bert_sim_path)
        # if len(self.campaigns) > 0 and len(self.big_campaigns) == 0:
        #     for  campaign in self.campaigns:
        #         bert_sim_path = os.path.join(campaigns_bert, f"{campaign.id}.pkl")
        #         print("calculating bert similarity model")
        #         bert_similarity = CosineSimilarity()
        #         procedures_phrases = get_procedure_phrases(self.procedures)
        #         campaign_phrases = campaign.phrases
        #         bert_similarity.compute_range(procedures_phrases,campaign_phrases )
        #         bert_similarity.to_pickle(bert_sim_path)
        #         campaign.bert_path = bert_sim_path




    
    def materialize_tech(self, path:str):
        """
        Generate procedure group from the given metadata
        """
        df = pd.read_csv(path)
        techniques = dict()
        for i in range(0,len(df)):
            tech_id = df.loc[i, "tech_id"]
            procedure_id = df.loc[i, "id"]
            locations = df.loc[i, "platform"].replace("[", "").replace("]", "").replace("\'","").split(", ")
            if tech_id not in techniques:
                techniques[tech_id] = {"tech_id": tech_id, "locations": locations, "procedures": [procedure_id]}
            else:
                techniques[tech_id]["procedures"].append(procedure_id)

        techs = dict()
        for k,v in techniques.items():
            tech_id = v["tech_id"]
            locations = v["locations"]
            _procedures = v["procedures"]
            # if tech_id == "T1566.001":
            #     print(1)
            te = Technique(tech_id, locations, _procedures)
        
            te.add_procedures(self.procedures)
            te.to_json(os.path.join(tech_json_dir,te.id+".json"))
            if te.id in techs:
                continue
            else:
                techs[te.id] = te
        self.techniques = techs

    def get_important_techniques(self):
        picked_tactics = Keys.TACTICS
        if len(picked_tactics) == 0:
            return []
        picked_techniques = []
        for k,v in tech_tac_mapper.items():
            if len(set(v).intersection(set(picked_tactics))) > 0:
                        picked_techniques.append(k)
        return picked_techniques

    def read_procedures_from_text_multiprocess(self, path: str):
        """
        Read procedures from the given path
        """
        num_processes = Keys.NUM_PROCESSES if self.multiprocessing else 1
        none_entity_nodes = []
        error_ids = []
        futures = []
        procedures = pd.read_csv(path)
        important_techniques = self.get_important_techniques()
        with concurrent.futures.ProcessPoolExecutor(max_workers= num_processes) as executor:
            for i in range(0,len(procedures)):
                if len(important_techniques)==0:
                    futures.append(executor.submit(generate_procedure, procedures, i))
                if len(important_techniques)>0:
                    if procedures.loc[i, "tech_id"] not in important_techniques:
                        continue
                    futures.append(executor.submit(generate_procedure, procedures, i))



    def compress_data(self):
        shutil.make_archive('data/compressed/p_out', 'zip', root_dir='data/procedure/output')
        shutil.make_archive('data/compressed/c_out', 'zip', root_dir='data/campaign/output')
        shutil.make_archive('data/compressed/t_out', 'zip', root_dir='data/Techniques/json')

    def analyze_procedures_from_text(self, path: str):
        """
        Read procedures from the given path
        """
        procedures = pd.read_csv(path)
        for i in range(0,len(procedures)):
            generate_procedure(procedures, i)

    
    def analyze_campaign_from_text(self, path: str):
        """
        Read campaign from the given path
        """
        files = os.listdir(path)
        for index in range(0,  len(files)):
            file = files[index]
            if file.endswith(".txt"):
                print(file)
                with open(os.path.join(path,file), "r") as f:
                    if Keys.IMAGE_GENERATION:
                        image_path = os.path.join(campaigns_image_dir, f"{file[0:-4]}.png")
                    else:
                        image_path = ""
                    campaign = Campaign(f.read(), file[0:-4], image_path=image_path)
                    self.campaigns.append(campaign)
                    campaign.to_json(os.path.join(campaigns_output_dir, campaign.id + ".json"))

    def analyze_big_campaign(self,dir):
        files = os.listdir(dir)
        for index in range(0,  len(files)):
            file = files[index]
            if file.endswith(".txt"):
                campaign_id = file[0:-4]
                print("start analyzing report :"+campaign_id),
                big_campaign = BigCampaign(os.path.join(dir, file), campaign_id)
                if len(big_campaign.data) == 0:
                    continue
                big_campaign.to_jsonl(os.path.join(campaigns_output_dir, campaign_id + ".jsonl"))
                self.big_campaigns.append(big_campaign)
            if file.endswith(".json"):
                campaign_id = file[0:-5]
                print("start analyzing report :"+campaign_id),
                big_campaign = BigCampaign(os.path.join(dir, file), campaign_id)
                if len(big_campaign.data) == 0:
                    continue
                big_campaign.to_jsonl(os.path.join(campaigns_output_dir, campaign_id + ".jsonl"))
                self.big_campaigns.append(big_campaign)

    def demo_procedure_matching(self):
        Alignment.all_alignment(self.campaigns[0],self.procedures, self.techniques)

    def save_campaign_matching_result(self):
        for campaign in self.campaigns:
            #save matching result for this campaign for later use
            campaign.to_pickle(os.path.join(campaigns_procedure_alignment_dir, campaign.id + ".pkl"))  
    def load_campaign_matching_result(self):
        files = os.listdir(campaigns_procedure_alignment_dir)
        self.campaigns = []
        for f in files:
            file_path = os.path.join(campaigns_procedure_alignment_dir, f)
            if file_path.endswith(".pkl"):
            #load matching result for this campaign for later use
                campaign = pickle.load(open(file_path, "rb"))
                self.campaigns.append(campaign)
    
    def procedure_matching(self):
        #todo: flatten the big campaign or update the alignment function to accept big campaign
        for campaign in self.campaigns:
            print("start analyzing this report "+campaign.id)
            if self.multiprocessing:
                #multi process verion
                Alignment.all_alignment_multiprocess(campaign,self.procedures, self.techniques)
            else:
                #single process version  
                Alignment.all_alignment_sequential(campaign,self.procedures, self.techniques)
                
            # print(f"Sequence techniques for {campaign.id} is :\n ", sequence_techniques)
            # campaign.sequence_techniques = sequence_techniques
            
            # with open(os.path.join(campaigns_sequence_techniques, campaign.id + ".json"), "w") as f:
            #     json.dump(sequence_techniques, f, indent=4)
            mapper_ = campaign.mapper
            file_path = os.path.join(campaigns_procedure_alignment_dir, campaign.id + ".pkl")
            
            with open(file_path, 'wb') as handle:
                pickle.dump(mapper_, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("done")
    def big_procedure_matching(self):
                 #todo: flatten the big campaign or update the alignment function to accept big campaign
        for campaign in self.big_campaigns:
            print("start analyzing this report "+campaign.id)
            if self.multiprocessing:
                #multi process verion
                Alignment.all_alignment_big_campaign_multiprocess(campaign,self.procedures, self.techniques)
            else:
                #single process version  
                Alignment.all_alignment_sequential_big_campaign_sequential(campaign,self.procedures, self.techniques)
            
            # print(f"Sequence techniques for {campaign.id} is :\n ", sequence_techniques)
            # campaign.sequence_techniques = sequence_techniques
            
            # with open(os.path.join(campaigns_sequence_techniques, campaign.id + ".json"), "w") as f:
            #     json.dump(sequence_techniques, f, indent=4)
            if campaign.mapper is None:
                campaign.mapper_gathering()
            mapper_ = campaign.mapper
            file_path = os.path.join(campaigns_procedure_alignment_dir, campaign.id + ".json")
            with open(file_path, "w") as f:
                json.dump(mapper_, f, indent=4)
            # with open(file_path, 'wb') as handle:
            #     pickle.dump(mapper_, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(file_path, 'rb') as handle:
            #     mapper_2 = pickle.load(handle)
            # print(mapper_2 == mapper_)    
        print("done")

    def report_decoding(self,matching_result_dir:str = campaigns_procedure_alignment_dir, tech_alignment_dir:str = campaigns_tech_alignment_dir, saved_decoding_dir:str = campaigns_decoding_result):
        mappers = dict()
        files = os.listdir(matching_result_dir)
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(matching_result_dir, file)
                with open(file_path, 'r') as handle:
                    mapper = json.load(handle)
                    id_ = file[0:-5]
                    if id_ not in mappers:
                        mappers[id_] = {}
                    mappers[id_]["procedure_alignment"] = mapper
        files = os.listdir(tech_alignment_dir)
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(tech_alignment_dir, file)
                with open(file_path, 'r') as handle:
                    mapper = json.load(handle)
                    id_ = file[0:-5]
                    if id_ not in mappers:
                        mappers[id_] = {}
                    mappers[id_]["tech_alignment"] = mapper
        for k,v in mappers.items():
            print("start decoding this report "+k)
            # mapper1 = Decoder.pure_decoding(v)
            mapper1, mapper2, final_path, top_k_path = Decoder.attack_path_decoding(v["procedure_alignment"],matching_threshold= Keys.DECODING_MATCHING_THRESHOLD,relax =Keys.DECODING_RELAXING, criteria = Keys.DECODING_CRITERIA,tech_alignment_mapper=v["tech_alignment"],topk = Keys.DECODING_TOP_K,recode=Keys.DECODING_RECODE)
            data = {"best": mapper1, "k2": mapper2, "full_path" : final_path, "top_k_path": top_k_path}
            saved_path = os.path.join(saved_decoding_dir, k + ".json")
            with open(saved_path, "w") as f:
                json.dump(data, f, indent=4)
        print("done")
    
    def statistics_calculation(self, saved_decoding_dir:str = campaigns_decoding_result):
        files = os.listdir(saved_decoding_dir)
        data = []
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(saved_decoding_dir, file), "r") as f:
                    json_object = json.load(f)
                    pure = json_object["pure"]

    # def alignment_decoding(self):
    #     # each alignment result from campaign will be decoded into sequences of techniques. Pair of techniques will be recorded in association_record
    #     for campaign in self.campaigns:
    #         print("start decoding this report "+campaign.id)
    #         self.association_record, all_paths = Decoder.greedy_decoding(campaign, association_record=self.association_record)
    #         # print(f"Sequence techniques for {campaign.id} is :\n ", all_paths)
    #     #save the paths, this one mostly for demonstration purpose
    #     with open("test_paths.json", "w") as f:
    #         json.dump(all_paths, f, indent=4)
    #     #save the association record for later use
    #     self.write_association()
    def write_cp_to_json(self):
        for campaign in self.campaigns:
            campaign.to_json(os.path.join(campaigns_output_dir, campaign.id + ".json"))
    def write_big_cp_to_jsonl(self):
        for campaign in self.big_campaigns:
            campaign.to_jsonl(os.path.join(campaigns_output_dir, campaign.id + ".jsonl"))
    def write_pro_to_json(self):
        for k,v in self.procedures.items():
            v.to_json(os.path.join(procedures_output_dir, v.id + ".json"))
    def write_to_json(self):
        """
        Write the campaign to json
        """
        for campaign in self.campaigns:
            campaign.to_json(os.path.join(campaigns_output_dir, campaign.id + ".json"))
        
        for k,v in self.procedures.items():
             v.to_json(os.path.join(procedures_output_dir, v.id + ".json"))

    def load_from_json(self):
        """
        Load the campaign from json
        """
        self.load_campaigns_from_json(campaigns_output_dir)
        self.load_procedures_from_json(procedures_output_dir)
    

    def load_procedures_from_json(self, path: str = procedure_deduplication_dir, load_from_jsonl = False):
        """
        Load procedures from the given path
        """
        self.procedures = dict()
        
        if load_from_jsonl:
            try:
                with jsonlines.open(procedures_output_file, "r") as reader:
                    for line in reader.iter():
                        procedure = Procedure()
                        procedure.from_json(json_object = line)
                        if len(procedure.graph_nodes) > 1:
                            self.procedures[procedure.id] = procedure
                        # self.procedures[procedure.id] = procedure
                        
            except:
                files = os.listdir(path)
                for file in files:
                    if file.endswith(".json"):
                        procedure = Procedure()
                        procedure.from_json(path = os.path.join(path, file))
                        if len(procedure.graph_nodes) > 1:
                            self.procedures[procedure.id] = procedure
                        # self.procedures[procedure.id] = procedure
        # print(f"load {len(self.procedures)} procedures")
        # del_list = []
        # for k, v in self.procedures.items():
        #     if len(v.graph_nodes) <=1:
        #         del_list.append(k)
        # for k in del_list:
        #     del self.procedures[k]
        # print(f"load {len(self.procedures)} procedures")
   
    def generate_procedure_jsonl(self, input_dir: str ="", output_file:str = ""):
        files = os.listdir(input_dir)
        data = []
        for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(input_dir, file), "r") as f:
                        json_object = json.load(f)
                        # if "special_phrases" not in json_object:
                        #     print(file)
                        data.append(json_object)
        with jsonlines.open(output_file, mode ='w') as writer:
            writer.write_all(data)
   
    def load_techniques_from_json(self, path: str):
        """
        Load procedure groups from the given path
        """
        self.techniques = dict()
        files = os.listdir(path)
        for file in files:
            if file.endswith(".json") and file.startswith("T"):
                technique = Technique.from_json(os.path.join(path, file))
                self.techniques[technique .id] = technique 
    
    
    def load_campaigns_from_json(self, path: str):
        """
        Load campaigns from the given path
        """
        self.campaigns = []
        files = os.listdir(path)
        for file in files:
            if file.endswith(".json"):
                campaign = Campaign()
                campaign.from_json(os.path.join(path, file))
                self.campaigns.append(campaign)


    def load_big_campaigns_from_jsonl(self, path: str):
        """
        Load campaigns from the given path
        """
        self.big_campaigns = []
        files = os.listdir(path)
        for file in files:
            if file.endswith(".jsonl"):
                campaign_id = file[0:-6]
                campaign = BigCampaign()
                campaign.from_jsonl(os.path.join(path, file), campaign_id)
                self.big_campaigns.append(campaign)




    def load_campaign_and_calculate(self, input_file:str):
        campaign = pickle.load(open(input_file, "rb"))
        self.association_record, sequence_techniques = Alignment.finalize_result(campaign, association_record=self.association_record)
        print(f"Sequence techniques for {campaign.id} is :\n ", sequence_techniques)



    def translate_matching_rs_to_excel(self, ms, name):
        data= list()
        for k, v in ms.items():
            procedure_id = v[0]
            similarity_value = v[1]
            procedure = self.procedures[procedure_id]
            text = procedure.text
            procedure.draw(os.path.join(procedures_image_dir, f"{procedure_id}.png"))
            data.append({"start_index": k, "id": v[0], "similarity": similarity_value, "text": text})
        
        df = pd.DataFrame(data)
        df.to_excel(f"{name}.xlsx", index=False)
    


    def procedure_deduplication(self, saved_dir:str = procedure_deduplication_dir):
        count = 0
        marking_dict = dict()
        for t in self.techniques.values():
            temp_dict = {}
            tech_procedures = t.procedures
            for p in tech_procedures:
                procedure = self.procedures.get(p, None)
                if procedure is None:
                    continue
                num_nodes = len(procedure.graph_nodes)
                if num_nodes not in temp_dict:
                    temp_dict[num_nodes] = [procedure.id]
                else:
                    temp_dict[num_nodes].append(procedure.id)
            print()
            for k,v in temp_dict.items():
                if len(v) > 1:
                    for i in range(0, len(v)-1):
                        if not marking_dict.get(v[i], True):
                                continue
                        for j in range(i+1, len(v)):
                            if not marking_dict.get(v[j], True):
                                continue #this procedure has been copied to another procedure
                            flag, combination = self.procedure_similarity(v[i], v[j])
                            count += 1
                            if flag:
                                if v[i] not in marking_dict:
                                    marking_dict[v[i]] = True
                                if v[j] not in marking_dict:
                                    marking_dict[v[j]] = False
                                self.procedure_accumulation(v[i], v[j], combination)
                                #copy v[j] to v[i]
            for p in tech_procedures:
                procedure = self.procedures.get(p, None)
                if procedure is None:
                    continue
                if procedure.id not in marking_dict or marking_dict[procedure.id]:
                    procedure.to_json(os.path.join(saved_dir, procedure.id + ".json"),reverse_text=False)

    
    def procedure_accumulation(self,procedure_id1, procedure_id2, combination):
        #update self.procedures[procedure_id1]
        procedure1 = self.procedures[procedure_id1]
        procedure2 = self.procedures[procedure_id2]
        keys = list(procedure1.graph_nodes.keys())
        for i in range(0, len(keys)):
            key1 = keys[i]
            if "texts" not in procedure1.graph_nodes[key1]["meta"]:
                procedure1.graph_nodes[key1]["meta"]["texts"] = [procedure1.graph_nodes[key1]["meta"]["text"]]
            if "verbs" not in procedure1.graph_nodes[key1]["meta"]:
                procedure1.graph_nodes[key1]["meta"]["verbs"] = []
            if combination[key1] is None:
                continue
            key2 = combination[key1][0]
            if "texts" in procedure2.graph_nodes[key2]["meta"]:
                procedure1.graph_nodes[key1]["meta"]["texts"].extend(procedure2.graph_nodes[key2]["meta"]["texts"])
            else:
                procedure1.graph_nodes[key1]["meta"]["texts"].append(procedure2.graph_nodes[key2]["meta"]["text"])
            procedure1.graph_nodes[key1]["meta"]["label"].extend(procedure2.graph_nodes[key2]["meta"]["label"])

            procedure1.graph_nodes[key1]["meta"]["label"] = list(set(procedure1.graph_nodes[key1]["meta"]["label"]))
            procedure1.graph_nodes[key1]["meta"]["texts"] = list(set(procedure1.graph_nodes[key1]["meta"]["texts"]))
            if "verbs" in procedure2.graph_nodes[key2]["meta"]:
                procedure1.graph_nodes[key1]["meta"]["verbs"].extend(procedure2.graph_nodes[key2]["meta"]["verbs"])
                procedure1.graph_nodes[key1]["meta"]["verbs"] = list(set(procedure1.graph_nodes[key1]["meta"]["verbs"]))
        _mapper = combination
        #add verbs
        for k,v in procedure1.graph_edges.items():
                # we will check each edge in procedure
                # we get edge source and dest node
            procedure_source= v["source"]
            procedure_dest = v["dest"]
            verb = v["verb"]
            if "verbs" in v:
                verbs = v["verbs"]
            else:
                verbs = [verb]
            if procedure_source not in procedure1.graph_nodes or procedure_dest not in procedure1.graph_nodes:
                    continue
            try:
                    if _mapper[procedure_source] is None or _mapper[procedure_dest] is None:
                        # None meaning that this procedure node does not have a similar node in campaign
                        continue
            except:

                    continue
            procedure2_source = _mapper[procedure_source][0]
            procedure2_dest = _mapper[procedure_dest][0]
            verb2 = None

            id2 = str(procedure2_source) + "_" + str(procedure2_dest)
            id2_ = str(procedure2_dest) + "_" + str(procedure2_source)
            if id2 in procedure2.graph_edges:
                verb2 = procedure2.graph_edges[id2]["verb"]
            if id2_ in procedure2.graph_edges:
                verb2 = procedure2.graph_edges[id2_]["verb"]


            if verb2 is None:
                continue
            else:
                verbs.append(verb2)
            v["verbs"] = list(set(verbs))
        return True
    
    def procedure_similarity(self, procedure_id1, procedure_id2):
        procedure1 = self.procedures[procedure_id1]
        procedure2 = self.procedures[procedure_id2]
        sub_graph = list(procedure2.graph_nodes.keys())
        # calculate bert similarity here
        similarity_, combination = Alignment.procedure_graph_alignment(procedure2, sub_graph, procedure1)
        if similarity_ > 0.95: # these two procedures are pretty similar
            return True, combination
        return False, None


