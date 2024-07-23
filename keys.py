import psutil
import math
class Keys():
    LAMDA = 0.4 # put more focus on the label
    SOFT_LAMDA = 0.3 # put more focus on the text
    NODE_SIMILARITY_THRESHOLD = 0.8
    MATCHING_THRESHOLD = 0.8
    DECODING_RECODE  = True
    DECODING_MATCHING_THRESHOLD = 0.87
    DECODING_RELAXING = True
    STRONG_VERB_GROUP = ["delete","mimic","schedule","reboot","hide","prevent","encode compress","decode","exfiltrate","command","reboot","user_action","proxy","damge","search analyze","install develop","change"]
    DECODING_CRITERIA = "heuristic"
    DISTANCE_FACTOR_PER_SENTENCE = 0.3
    CAMPAIGN_PATH = r"data/campaign"
    PROCEDURE_PATH = r"data/procedure"
    TECHNIQUE_PATH = r"data/Techniques"
    CONTEXT_SIMILARITY_PATH = r"data/campaign/USE_cosine"
    FIXING_PATTERN = r"data/patterns/fix_pattern.json"
    TOP_VALUE = 1
    DECODING_TOP_K = 1
    SPECIAL_DIR_PATTERN = r"data/patterns/special_dir.json"
    MALWARE_LIST = r"data/meta data/malware.json"
    IMAGE_GENERATION = False
    PRE_ASSOCIATION_FILE = r"data/meta data/rel_annotations_main.xlsx"
    SIMILAR_PROCEDURE_FILE = r"data/meta data/similar_procedures.json"
    # TACTICS = ["TA0001","TA0002","TA0003","TA0004","TA0003","TA0011","TA0010","TA0009","TA0007"]
    TACTICS = [] # [] mean all tactics
    NER_MODEL = r"data/saved_ner_model/model-best/"
    REMOVE_WORDS =r"data/meta data/remove_words.json"
    #we use the max number of physical cpu cores to run the program, always -1
    #reduce by half
    NUM_PROCESSES = math.floor(psutil.cpu_count(logical=False)/2)
    NUM_WORK_PER_PROCESS = 1000
    BERT_SIM_ENABLE = True
    MULTI_PROCESSING = False
    ACTOR_TOLERATE_DISTANCE = 1.0
    VERB_DIFF_PUNISHMENT = 0.5
    VERB_DIFF_SEVERVE_PUNISHMENT = 0.3
    VERB_DIFF_SOFT_PUNISHMENT = 0.8
    ENABLE_BIG_CAMPAIGN = False
    LABEL2ID = {
    "OTHER": 0,
    "DATA": 1,
    "DIRECTORY": 2,
    "ENCRYPTION": 3,
    "FUNCTION": 4,
    "NETWORK": 5,
    "COMPONENT": 6,
    "REGISTRY": 7,
    "USER": 8,
    "VULNERABILITY": 9,
    "ACTOR": 10,
    "ABSTAIN": -1
    }
    ID2LABEL = {
    0: "OTHER",
    1: "DATA",
    2: "DIRECTORY",
    3: "ENCRYPTION",
    4: "FUNCTION",
    5: "NETWORK",
    6: "COMPONENT",
    7: "REGISTRY",
    8: "USER",
    9: "VULNERABILITY",
    10: "ACTOR"
    }
