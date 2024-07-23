import pandas as df
import Levenshtein
import json
import pandas as pd
from keys import Keys
import Levenshtein
import re
import Levenshtein
import re
def regex_match(phrase:str, pattern: str):
    #phrase now is a root of the word
    #keyword now has len(split()) > 1
    if re.search(pattern, phrase, re.IGNORECASE):
        return True

def regex_match2(phrase:str, pattern: str):
    #phrase now is a root of the word
    #keyword now has len(split()) > 1
    ms = re.search(pattern, phrase, re.IGNORECASE)
    if ms:
        return ms.group(0)
    return False
def regex_patterns(phrase:str, patterns):
    for pattern in patterns:
        if regex_match(phrase, pattern):
            return pattern
    return False


def regex_patterns2(phrase:str, patterns):
    for pattern in patterns:
        ms = regex_match2(phrase, pattern)
        if ms:
            return ms
    return False
#special search would do the same but have x2 important weights in majority voting
def special_search(phrase:str, keywords: str):
    return keyword_search(phrase, keywords)
def extract_match(phrase:str, keyword: str, rate=0.9):
    if Levenshtein.ratio(phrase, keyword) > rate:
        return True

def keyword_search(phrase:str, keywords: str):

    phrase = phrase.lower()
    split = phrase.split()

    for keyword in keywords:

        keyword = keyword.lower()
        keyword_split = keyword.split()

        if len(split) == len(keyword_split):
            if len(split) == 1:
                if extract_match(phrase, keyword, rate=0.95):
                    return keyword
            else:
                if extract_match(phrase, keyword):
                    return keyword
        if len(split) > len(keyword_split):
            focus_parts = split[len(split)-len(keyword_split):]
            focus_phrase = " ".join(focus_parts)
            if len(keyword_split) == 1:
                if extract_match(focus_phrase, keyword, rate=0.95):
                    return keyword
            else:
                if extract_match(focus_phrase, keyword):
                    return keyword

    return False






#x is a row in the dataframe
label2id = Keys.LABEL2ID
id2label = Keys.ID2LABEL
ABSTAIN = -1
class Entity():
    def __init__(self, file_name,label) -> None:
        with open(file_name, "r") as f:
            data = json.load(f)
        self.regex= []
        self.special = []
        self.strong_special = []
        self.strong_regex = []
        self.keywords = []
        self.label = label

        for key in data.keys():
            if "strong_regex" in key:
                self.strong_regex.extend(data[key])
            if "strong_special" in key:
                self.strong_special.extend(data[key])
            if "regex" in key:
                self.regex.extend(data[key])
            elif "special" in key:
                self.special.extend(data[key])
            else:
                self.keywords.extend(data[key])
        self.keywords = list(set(self.keywords))
        
        self.regex = list(set(self.regex))
        self.special = list(set(self.special))
        self.special = sorted(self.special, key=len, reverse=True)

    def recognize_special(self, phrase):
        rs= special_search(phrase, self.special)
        if rs:
                return self.label
        return ABSTAIN
    def recognize_regex(self, phrase):
        ms= regex_patterns(phrase, self.regex)
        if ms: 
            return self.label
        return ABSTAIN
    def recognize_regex2(self, phrase):
        ms= regex_patterns2(phrase, self.regex)
        if ms: 
            return self.label, ms
        return ABSTAIN
    def recognize_keyword(self, phrase):
        rs= keyword_search(phrase, self.keywords)
        if rs:
                return self.label
        return ABSTAIN
    def recognize(self, phrase):
        
        if special_search(phrase, self.special):
                return self.label
        if regex_patterns(phrase, self.regex):
            return self.label
        
        if keyword_search(phrase, self.keywords):
            return self.label
        return ABSTAIN
    
    def absolute(self, text):
        result  = []
        for sp in self.strong_special:
            m = re.search(re.escape(sp), text)
            if m:
                start = m.start(0)
                end = m.end(0)
                before = start -1
                if before >= 0:
                    if text[before].isalnum():
                        continue
                after = end +1
                if after < len(text):
                    if text[after].isalnum():
                        continue
                result.append((m.group(0),m.start(0), m.end(0), Keys.ID2LABEL[self.label]))
                break

        for rg in self.strong_regex:
            matches = re.finditer(rg, text,re.IGNORECASE)
            for m in matches:
                result.append((m.group(0),m.start(0), m.end(0), Keys.ID2LABEL[self.label]))
        return result
        
actor = Entity(r"data/dictionarydata/Actor.json", label=label2id["ACTOR"]) 
data = Entity(r"data/dictionarydata/Data.json", label= label2id["DATA"])
directory = Entity(r"data/dictionarydata/Dir.json", label=label2id["DIRECTORY"])
encryption = Entity(r"data/dictionarydata/Encryption.json", label=label2id["ENCRYPTION"])
function = Entity(r"data/dictionarydata/Functionality.json", label=label2id["FUNCTION"])
network = Entity(r"data/dictionarydata/Networking.json", label=label2id["NETWORK"])
component = Entity(r"data/dictionarydata/OS-Service-App.json", label=label2id["COMPONENT"])
registry = Entity(r"data/dictionarydata/RegistryDLL.json", label=label2id["REGISTRY"])
user = Entity(r"data/dictionarydata/User.json", label=label2id["USER"])
vulnerability = Entity(r"data/dictionarydata/Vulnerability.json", label=label2id["VULNERABILITY"])
other = Entity(r"data/dictionarydata/Other.json", label=label2id["OTHER"])
entities = [directory,registry,network,vulnerability]
full_entities = [actor,data,directory,encryption,function,network,component,registry,user,vulnerability,other]

from snorkel.labeling import PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel,MajorityLabelVoter
import json


@labeling_function()
def lf_Actor(x):
    phrase = x.phrase
    return_ = actor.recognize(phrase)
    return return_

@labeling_function()
def lf_Actor_special(x):
    phrase = x.phrase
    return_ = actor.recognize_special(phrase)
    return return_

@labeling_function()
def lf_Actor_regex(x):
    phrase = x.phrase
    return_ = actor.recognize_regex(phrase)
    return return_
@labeling_function()
def lf_Actor_keyword(x):
    phrase = x.phrase
    return_ = actor.recognize_keyword(phrase)
    return return_
    

@labeling_function()
def lf_Data(x):
    phrase = x.phrase
    return_ = data.recognize(phrase)
    return return_
@labeling_function()
def lf_directory(x):
    phrase = x.phrase
    return_ = directory.recognize(phrase)
    return return_

@labeling_function()
def lf_encrypt(x):
    phrase = x.phrase
    return_ = encryption.recognize(phrase)
    return return_

@labeling_function()
def lf_function(x):
    phrase = x.phrase
    return_ = function.recognize(phrase)
    return return_

@labeling_function()
def lf_network(x):
    phrase = x.phrase
    return_ = network.recognize(phrase)
    return return_

@labeling_function()
def lf_component(x):
    phrase = x.phrase
    return_ = component.recognize(phrase)
    return return_


@labeling_function()
def lf_registry(x):
    phrase = x.phrase
    return_ = registry.recognize(phrase)
    return return_


@labeling_function()
def lf_user(x):
    phrase = x.phrase
    return_ = user.recognize(phrase)
    return return_

@labeling_function()
def lf_vulnerability(x):
    phrase = x.phrase
    return_ = vulnerability.recognize(phrase)
    return return_

@labeling_function()
def lf_other(x):
    phrase = x.phrase
    return_ = other.recognize(phrase)
    return return_


@labeling_function()
def lf_Data_special(x):
    phrase = x.phrase
    return_ = data.recognize_special(phrase)
    return return_
@labeling_function()
def lf_directory_special(x):
    phrase = x.phrase
    if "DIRECTORY" in phrase:
        return directory.label
    return_ = directory.recognize_special(phrase)
    return return_

@labeling_function()
def lf_encrypt_special(x):
    phrase = x.phrase
    return_ = encryption.recognize_special(phrase)
    return return_

@labeling_function()
def lf_function_special(x):
    phrase = x.phrase
    return_ = function.recognize_special(phrase)
    return return_

@labeling_function()
def lf_network_special(x):
    phrase = x.phrase
    return_ = network.recognize_special(phrase)
    return return_

@labeling_function()
def lf_component_special(x):
    phrase = x.phrase
    return_ = component.recognize_special(phrase)
    return return_


@labeling_function()
def lf_registry_special(x):
    phrase = x.phrase
    if "REGISTRY" in phrase:
        return registry.label
    return_ = registry.recognize_special(phrase)
    return return_


@labeling_function()
def lf_user_special(x):
    phrase = x.phrase
    return_ = user.recognize_special(phrase)
    return return_

@labeling_function()
def lf_vulnerability_special(x):
    phrase = x.phrase
    return_ = vulnerability.recognize_special(phrase)
    return return_

@labeling_function()
def lf_other_special(x):
    phrase = x.phrase
    return_ = other.recognize_special(phrase)
    return return_



@labeling_function()
def lf_Actor_special2(x):
    phrase = x.phrase
    return_ = actor.recognize_special(phrase)
    return return_
@labeling_function()
def lf_Data_special2(x):
    phrase = x.phrase
    return_ = data.recognize_special(phrase)
    return return_
@labeling_function()
def lf_directory_special2(x):
    phrase = x.phrase
    return_ = directory.recognize_special(phrase)
    return return_

@labeling_function()
def lf_encrypt_special2(x):
    phrase = x.phrase
    return_ = encryption.recognize_special(phrase)
    return return_

@labeling_function()
def lf_function_special2(x):
    phrase = x.phrase
    return_ = function.recognize_special(phrase)
    return return_

@labeling_function()
def lf_network_special2(x):
    phrase = x.phrase
    return_ = network.recognize_special(phrase)
    return return_

@labeling_function()
def lf_component_special2(x):
    phrase = x.phrase
    return_ = component.recognize_special(phrase)
    return return_


@labeling_function()
def lf_registry_special2(x):
    phrase = x.phrase
    return_ = registry.recognize_special(phrase)
    return return_


@labeling_function()
def lf_user_special2(x):
    phrase = x.phrase
    return_ = user.recognize_special(phrase)
    return return_

@labeling_function()
def lf_vulnerability_special2(x):
    phrase = x.phrase
    return_ = vulnerability.recognize_special(phrase)
    return return_

@labeling_function()
def lf_other_special2(x):
    phrase = x.phrase
    return_ = other.recognize_special(phrase)
    return return_

@labeling_function()
def lf_Data_regex(x):
    phrase = x.phrase
    return_ = data.recognize_regex(phrase)
    return return_
ignore_dir1 = r"^(HKEY_CLASSES_ROOT|KCU|HKCR|HKLM|HKCU|HKU|HKEY_LOCAL_MACHINE|HKEY_CURRENT_USER|HKEY_CURRENT_USERS|HKCR|HKEY_USERS|KEY_CURRENT_USER|Registry)+"
ignore_dir2 = r"^(schtasks|odbcconf|net|netstat|sudo|dir|mmc|rundll32|chmod|cmd|wmic)(\.exe)?[\s]+"
@labeling_function()
def lf_directory_regex(x):
    phrase = x.phrase
    if re.search(ignore_dir1, phrase, re.IGNORECASE):
        return ABSTAIN
    if re.search(ignore_dir2, phrase, re.IGNORECASE):
        return ABSTAIN
    if phrase.endswith(".exe"):
        return ABSTAIN
    if "DIRECTORY" in phrase:
        return directory.label
    return_ = directory.recognize_regex(phrase)
    return return_

@labeling_function()
def lf_encrypt_regex(x):
    phrase = x.phrase
    return_ = encryption.recognize_regex(phrase)
    return return_

@labeling_function()
def lf_function_regex(x):
    phrase = x.phrase
    return_ = function.recognize_regex(phrase)
    return return_

@labeling_function()
def lf_network_regex(x):
    phrase = x.phrase
    return_ = network.recognize_regex(phrase)
    return return_

@labeling_function()
def lf_component_regex(x):
    phrase = x.phrase
    return_ = component.recognize_regex(phrase)
    return return_


@labeling_function()
def lf_registry_regex(x):
    phrase = x.phrase
    return_ = registry.recognize_regex(phrase)
    return return_


@labeling_function()
def lf_user_regex(x):
    phrase = x.phrase
    return_ = user.recognize_regex(phrase)
    return return_

@labeling_function()
def lf_vulnerability_regex(x):
    phrase = x.phrase
    return_ = vulnerability.recognize_regex(phrase)
    return return_

@labeling_function()
def lf_other_regex(x):
    phrase = x.phrase
    return_ = other.recognize_regex(phrase)
    return return_

data_ignore_ = r"(file|attachment|document|item)[s]?$"
@labeling_function()
def lf_Data_keyword(x):
    phrase = x.phrase
    if re.search(data_ignore_, phrase, re.IGNORECASE):
        if function.recognize_regex(phrase)!= ABSTAIN or registry.recognize_regex(phrase)!= ABSTAIN or encryption.recognize_regex(phrase)!= ABSTAIN:
            return ABSTAIN
    if phrase.endswith("user") or phrase.endswith("users"):
        return ABSTAIN
    return_ = data.recognize_keyword(phrase)
    return return_
@labeling_function()
def lf_directory_keyword(x):
    phrase = x.phrase
    if "DIRECTORY" in phrase:
        return directory.label
    return_ = directory.recognize_keyword(phrase)
    return return_

@labeling_function()
def lf_encrypt_keyword(x):
    phrase = x.phrase
    return_ = encryption.recognize_keyword(phrase)
    return return_

@labeling_function()
def lf_function_keyword(x):
    phrase = x.phrase
    return_ = function.recognize_keyword(phrase)
    return return_

@labeling_function()
def lf_network_keyword(x):
    phrase = x.phrase
    return_ = network.recognize_keyword(phrase)
    return return_

@labeling_function()
def lf_component_keyword(x):
    phrase = x.phrase
    return_ = component.recognize_keyword(phrase)
    return return_


@labeling_function()
def lf_registry_keyword(x):
    phrase = x.phrase
    if "REGISTRY" in phrase:
        return registry.label
    return_ = registry.recognize_keyword(phrase)
    return return_


@labeling_function()
def lf_user_keyword(x):
    phrase = x.phrase
    return_ = user.recognize_keyword(phrase)
    return return_

@labeling_function()
def lf_vulnerability_keyword(x):
    phrase = x.phrase
    return_ = vulnerability.recognize_keyword(phrase)
    return return_

@labeling_function()
def lf_other_keyword(x):
    phrase = x.phrase
    return_ = other.recognize_keyword(phrase)
    return return_
# from snorkel.preprocess import preprocessor
# from BERT_model import *
# @preprocessor(memoize=True)
# def ner(x):
#     text = x.sentence
#     ner = classifier(text)
#     ner.append({'entity': 'O-NONE', 'score': 0, 'index': 0, 'word': 'exe', 'start': 0, 'end': 0})
#     x.entities = reconstruct_entity(ner)
#     return x

# @labeling_function(pre=[ner])
# def lf_hf_actor(x):
#     sentence = x.sentence
#     entities = x.entities
#     phrase = x.phrase
#     start = x.start
#     end = x.end
#     for e in entities:
#         e_start = e[0]
#         e_end = e[1]
#         e_label = e[2]
#         phrase2 = sentence[e_start:e_end]
#         if Levenshtein.ratio(phrase, phrase2) > 0.9 and e_label == "APT":
#             return Keys.LABEL2ID["ACTOR"]
#     return ABSTAIN

# @labeling_function(pre=[ner])
# def lf_hf_component(x):
#     sentence = x.sentence
#     entities = x.entities
#     phrase = x.phrase
#     start = x.start
#     end = x.end
#     for e in entities:
#         e_start = e[0]
#         e_end = e[1]
#         e_label = e[2]
#         phrase2 = sentence[e_start:e_end]
#         if Levenshtein.ratio(phrase, phrase2) > 0.9 and e_label == "OS":
#             return Keys.LABEL2ID["COMPONENT"]
#     return ABSTAIN

# @labeling_function(pre=[ner])
# def lf_hf_commu(x):
#     sentence = x.sentence
#     entities = x.entities
#     phrase = x.phrase
#     start = x.start
#     end = x.end
#     for e in entities:
#         e_start = e[0]
#         e_end = e[1]
#         e_label = e[2]
#         phrase2 = sentence[e_start:e_end]
#         if Levenshtein.ratio(phrase, phrase2) > 0.9 and e_label == "EMAIL":
#             return Keys.LABEL2ID["NETWORK"]
#     return ABSTAIN

# @labeling_function(pre=[ner])
# def lf_hf_data(x):
#     sentence = x.sentence
#     entities = x.entities
#     phrase = x.phrase
#     start = x.start
#     end = x.end
#     for e in entities:
#         e_start = e[0]
#         e_end = e[1]
#         e_label = e[2]
#         phrase2 = sentence[e_start:e_end]
#         if Levenshtein.ratio(phrase, phrase2) > 0.9 and e_label in ["LOC","TIME","MD5","SHA1","SHA2"]:
#             return Keys.LABEL2ID["DATA"]
#     return ABSTAIN

# @labeling_function(pre=[ner])
# def lf_hf_func(x):
#     sentence = x.sentence
#     entities = x.entities
#     phrase = x.phrase
#     start = x.start
#     end = x.end
#     for e in entities:
#         e_start = e[0]
#         e_end = e[1]
#         e_label = e[2]
#         phrase2 = sentence[e_start:e_end]
#         if Levenshtein.ratio(phrase, phrase2) > 0.9 and e_label in ["ACT","MAL","TOOL"]:
#             return Keys.LABEL2ID["FUNCTION"]
#     return ABSTAIN




# @labeling_function(pre=[ner])
# def lf_hf_network(x):
#     sentence = x.sentence
#     entities = x.entities
#     phrase = x.phrase
#     start = x.start
#     end = x.end
#     for e in entities:
#         e_start = e[0]
#         e_end = e[1]
#         e_label = e[2]
#         phrase2 = sentence[e_start:e_end]
#         if Levenshtein.ratio(phrase, phrase2) > 0.9 and e_label in ["IP","PROT"]:
#             return Keys.LABEL2ID["NETWORK"]
#     return ABSTAIN

# @labeling_function(pre=[ner])
# def lf_hf_encryption(x):
#     sentence = x.sentence
#     entities = x.entities
#     phrase = x.phrase
#     start = x.start
#     end = x.end
#     for e in entities:
#         e_start = e[0]
#         e_end = e[1]
#         e_label = e[2]
#         phrase2 = sentence[e_start:e_end]
#         if Levenshtein.ratio(phrase, phrase2) > 0.9 and e_label in ["ENCR"]:
#             return Keys.LABEL2ID["ENCRYPTION"]
#     return ABSTAIN

# @labeling_function(pre=[ner])
# def lf_hf_vulnerability(x):
#     sentence = x.sentence
#     entities = x.entities
#     phrase = x.phrase
#     start = x.start
#     end = x.end
#     for e in entities:
#         e_start = e[0]
#         e_end = e[1]
#         e_label = e[2]
#         phrase2 = sentence[e_start:e_end]
#         if Levenshtein.ratio(phrase, phrase2) > 0.9 and e_label in ["VULID","VULNAME"]:
#             return Keys.LABEL2ID["VULNERABILITY"]
#     return ABSTAIN

#dulicate special and regex to allow more weight
function_dict = {
    "0": [lf_other_special2,lf_other_special,lf_other_regex,lf_other_keyword], 
    "1": [lf_Data_special2,lf_Data_special,lf_Data_regex,lf_Data_keyword],
    "2": [lf_directory_special2,lf_directory_special,lf_directory_regex,lf_directory_keyword],
    "3": [lf_encrypt_special2,lf_encrypt_special,lf_encrypt_regex,lf_encrypt_keyword],
    "4": [lf_function_special2,lf_function_special,lf_function_regex,lf_function_keyword],
    "5": [lf_network_special2,lf_network_special,lf_network_regex,lf_network_keyword],
    "6": [lf_component_special2,lf_component_special,lf_component_regex,lf_component_keyword],
    "7": [lf_registry_special2,lf_registry_special,lf_registry_regex,lf_registry_keyword],
    "8": [lf_user_special2,lf_user_special,lf_user_regex,lf_user_keyword],
    "9": [lf_vulnerability_special2,lf_vulnerability_special,lf_vulnerability_regex,lf_vulnerability_keyword],
    "10": [lf_Actor_special2,lf_Actor_special,lf_Actor_regex,lf_Actor_keyword]
}

lfs = []
for key in function_dict.keys():
    lfs.extend(function_dict[key])

applier = PandasLFApplier(lfs)
label_model = MajorityLabelVoter(cardinality=11, verbose=False)
# label_model.load(Keys.LABEL_MODEL)

def delete_ENTITY(text):
    entity_pattern = r"\b(ENTITY)[0-9]+\b"
    return re.sub(entity_pattern, "", text).strip()

def convert_to_csv(sentence_obj):
    data =[]
    seen = []
    for svo in sentence_obj:
        sub, verb, obj = svo
        if not isinstance(sub, str):
            if sub["text"] not in seen:
                seen.append(sub["text"])
                data.append({"phrase":delete_ENTITY(sub["text"]), "label":0, "start":sub["start"], "end":sub["end"]})
        if obj["text"] not in seen:
            seen.append(obj["text"])
            data.append({"phrase":delete_ENTITY(obj["text"]), "label":0, "start":obj["start"], "end":obj["end"]})

    df = pd.DataFrame(data)
    return df

def heuristic_extract(sentence_objs):
    if len(sentence_objs) == 0:
        return []
    df = convert_to_csv(sentence_objs)
    L_df = applier.apply(df)
    labels = label_model.predict(L_df, tie_break_policy="abstain")
    df["label"] = labels
    entities = []
    for index, row in df.iterrows():
        phrase = row["phrase"]
        if len(phrase.split()) ==1 and "ENTITY" in phrase:
            continue
        if row["label"] != ABSTAIN:         
            label = Keys.ID2LABEL[row["label"]]
        else:
            label = "OTHER"
        label = heuristic_rules(phrase, label)
        entities.append({"text":phrase, "label":label})
    return entities


def heuristic_extract_(input_df):
    df = input_df.copy()
    L_df = applier.apply(df)
    labels = label_model.predict(L_df, tie_break_policy="abstain")
    df["label"] = labels
    entities = []
    for index, row in df.iterrows():
        phrase = row["phrase"]
        ID = row["ID"]
        if row["label"] != ABSTAIN:

            label = Keys.ID2LABEL[row["label"]]
        else:
            label = "OTHER"
        # this is the place for some heuristic rules that 

        label = heuristic_rules(phrase, label)
        entities.append({"ID": ID,"text":phrase, "label":label})
    return entities



def heuristic_rules(phrase, original_label):
    if phrase == "":
        return original_label
    if "REGISTRY" in phrase:
        return "REGISTRY"
    if "ENCRYPTION" in phrase:
        return "ENCRYPTION"
    if "DIRECTORY" in phrase:
        return "DIRECTORY"
    if "NETWORK" in phrase:
        return "NETWORK"
    if "FUNCTION" in phrase:
        return "FUNCTION"
    if "DATA" in phrase:
        return "DATA"
    if "threat actor" in phrase.lower():
        return "ACTOR"
    p_split = phrase.split()
    if "actor" in p_split[-1].lower():
        return "ACTOR"
    if "attacker" in p_split[-1].lower():
        return "ACTOR"
    if "vunerabili" in phrase.lower():
        return "VULNERABILITY"
    return original_label


def replace_special_entities(text):
    
    rs = []
    for entity in entities:
        rs.extend(entity.absolute(text))
    rs = sorted(rs, key=lambda x: x[2]-x[1], reverse=True)
    mark = [1]* len(rs)
    for i in range(0, len(rs)-1):
        for j in range(i+1, len(rs)):
            if rs[i][1] <= rs[j][1] and rs[i][2] >= rs[j][2]:
                mark[j] = 0
    _rs = [rs[i] for i in range(0, len(rs)) if mark[i] == 1]
    return _rs