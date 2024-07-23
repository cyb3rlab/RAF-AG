import re
from modules import malwares
from keys import Keys
import unicodedata
from language_models import nlp
from modules import common_fixing_pattern
special_dir_list = Keys.SPECIAL_DIR_PATTERN
import json
with open(special_dir_list) as f:
    special_dir_data = json.load(f)
from pyinflect import getInflection
NEGATIONS = {"no", "not", "n't", "never", "none", "without","neither","nor"}
Verb_forms = {"VB","VBD","VBG","VBN","VBP","VBZ"}


def capitalize(line):
    return ' '.join(s[:1].upper() + s[1:] for s in line.split(' '))





def homogenization(stri):
    for k, v in special_dir_data.items():
        for value in v:
            if value.lower() in stri.lower():
                stri = stri.replace(value, k)
    return stri

cc_list =['ip:* ', 'c&c server','c&c servers', 'c&c','command and control', 'command and control server','command and control servers','command and control (C2)','command and control (C2) server','command and control (C2) servers', 'c2 server','c2 servers' 'c2', 'candc server','candc servers' 'candc', 'cc server','cc servers', 'command & control server', 'command & controle server', 'command & control', 'CandC server', 'CandC servers','CandC', 'CC server','CC servers', 'Command & Controle server', 'Command & Controle servers', 'Command & Controle', 'CnC','command & control servers', 'command & controle servers',]

def CـC(text):
    pattern = sorted(cc_list, key=len, reverse=True)
    for i in pattern:
        text = re.sub(re.escape(i), "C2", text, flags=re.IGNORECASE)
        
    text = text.replace("C2 (C2)", "C2")
    text = text.replace("C2 C2", "C2")
    return text




                                                                               
substitutions = {"has been observed": "is","has been seen": "is","has been known to": "can","is known to":"can","has enumerated": "enumerated","look for":"find","%Application Data%": "%ApplicationData%","exfil’d":"exfiltrated","exfil ":"exfiltrate ","exfil'ing":"exfiltrating","side load":"side-load",
    "Launch Daemon":"LaunchDaemon","launch daemon":"launchdaemon", "look like": "as","known as":"named","with the help of": "by", "in order to": "to", "in order for": "for", "on the basis of": "based on",
                 "on account of": "because of", "in spite of": "despite", "with regard to": "about", "with respect to": "about",
                 "with respect for": "for", "in front of": "before",
                 "in accordance with": "according to", "with a view to": "to", "in case of": "if", "in the event of": "if",
                  "for the sake of": "for", "on the grounds that": "because", "on the ground that": "because",
                 "on condition that": "if", "in the absence of": "without", "for the benefit of": "for", "for the sake of": "for",
                 "on the basis that": "because", "in the case of": "for", "in the event that": "if", "in the light of": "because",
                 "in the course of": "during", "in the form of": "such as", "in the shape of": "such as", "in the manner of": "such as",
                 "in the style of": "such as", "in the process of": "while", "in the interests of": "for", "in the vicinity of": "near",
                 "in the neighborhood of": "near", "in the neighbourhood of": "near", "in the direction of": "toward",
                 "in the neighborhood of": "near", "in the neighbourhood of": "near", "in the direction of": "toward",
                 "in the event of": "if", "in the event that": "if", "in the event of": "if", "in the event that": "if",
                 "through the medium of": "by", "by means of": "by", "by dint of": "by", "by virtue of": "by",
                 " via": " by", "with a view to": "to", "with a view toward": "to", "with a view towards": "to",
                 "through":"by",", for example,":", such as",", including":", such as", ", for instance,":", such as", "e.g.,":"such as", "click on": "click",
                 "actor-controlled":"malicious", "actor-created": "malicious", "actor-generated": "malicious", "as soon as":"when","as long as":"while", "at the same time as": "when","time zone":"timezone","time-zone":"timezone","in tandem with":"with","in parallel with":"with","in conjunction with":"with","in combination with":"with",
                 "at the time that": "when", "at the moment that": "when", "at the point that": "when", "at the point in time that": "when", "at the time":"when", "back to": "to",
                  "leverage":"use","leveraged":"used","leveraging":"using","NT Authority":"NTAuthority", "NT AUTHORITY":"NTAuthority",
                 "decrypt stings":"decrypt strings","Registry ket":"Registry key","Windows 11":"Windows11", "Windows XP":"WindowsXP","Windows 8":"Windows8","Windows 10":"Windows10","Windows 7":"Windows7","Windows NT":"WindowsNT","leveraged": "used", "leverages":"uses","leveraging": "using", "leverage":"use", "a lot of":"many", "lots of":"many", "brute force": "bruteforce","brute forced": "bruteforced","brute forcing": "bruteforcing",
                 "has the ability to": "can","have the ability to": "can","have the capacity to": "can","has the capacity to": "can", "is able to": "can", "is capable to": "can","are able to": "can", "are capable to": "can",
                 "are capable of": "can","is capable of": "can", "is able of": "can","are able of": "can", "(s)": "s","(S)": "s", "for use as": "as", "for use in": "in", "for use on": "on", "for use within": "within", "for use during":"during","for use of": "of", "for use with":"with", "to act as": "as","to use for": "for", "to use in": "in","to use as": "as", "to act on": "on", "a series of":"", "a list of": "", "a sequence of":"", "a number of":"", "a variety of":"", "a range of":"", "a set of":"", "a group of":"", "a collection of":"", "a combination of":"", "a pair of":"", "a couple of":""}
                #       "for execution of": "to execute", "for execution": "to execute","for persistence":"to persist","to ensure its persistence":"to persist",
                #  "to establish persistence":"to persist", "maintain persistence": "persist", "as part of its persistence":"to persist", "has established persistence": "has persisted","has added persistence": "has persisted","to ensure persistence":"to persist", "to gain persistence":"to persist", "to enable persistence mechanisms": "to persist", "as a persistence mechanism": "to persist"}
def should_fix(sent):
    doc = nlp(sent)
    first = doc[0]
    flag = False
    sents = list(doc.sents)
    if len(sents) == 0:
        return False
    sent = sents[0]

    for t in doc:
        if t.dep_ in ["nsubj", "nsubjpass", "csubj", "csubjpass"]:
            flag = True
    if (first.pos_ in ["VERB","NOUN"] or first.lemma_ in ["browse","move","instal"]) and not flag: #there is verb but no subject
        return True
    return False

def handling_substitutions(text):
    for k,v in substitutions.items():
        text = text.replace(k, v)
    return text



def remove_non_ascii(text):
    return text.encode('ascii', errors='ignore').decode()
removed = ["any","many", "various", "every", "own", "our", "their", "his","her", "its","my", "a", "the", "an", "these", "those"]

def remove_common_word(text):
    split = text.split(" ")
    remains= [s for s in split if s.lower() not in removed]
    return " ".join(remains)

def find_main_entity(input_text:str):
    pattern = common_fixing_pattern["mitre_subject"]
    for r in pattern:
            matches = re.finditer(r, input_text)
            entities = [m.group(0) for m in matches]
    return entities[0].replace("[", "").replace("]", "")
def remove_explicit_entity(input_text:str):
    abundant_pattern = [r"^During.*?\(https://attack.mitre.org/campaigns/[\S]*\)\,",r"^For.*?\(https://attack.mitre.org/campaigns/[\S]*\)\,"]
    for ab in abundant_pattern:
        input_text = re.sub(ab, "", input_text)
    data = []
    pattern = common_fixing_pattern["mitre_entity"]
    for r in pattern:
            matches = re.finditer(r, input_text)
            entities = [m.group(0) for m in matches]
            for entity in entities:
                        entity = entity.replace("))", ")")
                        if "software" in entity:
                            data.append({"entity":entity, "type":"Software"})
                        if "groups" in entity:
                            data.append({"entity":entity, "type":"Group"})
                        if "campaigns" in entity:  
                            data.append({"entity":entity, "type":"Campaign"})
                        if "techniques" in entity:
                            data.append({"entity":entity, "type":"Technique"})
                        if "tactics" in entity:
                            data.append({"entity":entity, "type":"Tactics"})
                        main_entity = find_main_entity(entity)
                        # if main_entity in malwares:
                        #     main_entity = "malware"
                        input_text = input_text.replace(entity, main_entity, 1)

    return input_text

def remove_link_and_citations(input_text):
    input_text = re.sub("[\n\t\r]+", " ",input_text).replace("  ", " ")
    citation_pattern = common_fixing_pattern["citation"]
    for i in citation_pattern:
            input_text = re.sub(i, "", input_text)

    return input_text
def fix_unicode(text):
    text = re.sub(u"(\u2018|\u2019)", "'", text)
    text = re.sub(u"(\u201c|\u201d)", '"', text)
    text = re.sub(u"\u2013", "-", text)
    text = re.sub(u"\u2014", "-", text)
    text = re.sub(u"\u2026", "...", text)
    text = re.sub(u"\uf09f", " ", text)
    text = re.sub(u"\u202E", "", text)
    text = re.sub(u"\ufeff", "", text)
    return text

pattern = r"^(\u2022|\u25e6|\u2218)"
pattern2 = r"(\u2022|\u25e6|\u2218)"
def fix_enumeration(text):
    text = unicodedata.normalize("NFKD",text)
    splits = text.split("\n")
    new_splits = []
    for i in range(0, len(splits)):
        sent = splits[i].strip()
        result = re.search(pattern, sent)
        if result:
            newsent = re.sub(pattern2, "", sent).strip()
            _split = newsent.split(" ")
            if len(_split) > 2 and should_fix(newsent.lower()):

                prenewsent =  newsent[0].lower() + newsent[1:]
                new_sent = subject_elipsis(prenewsent)
                if new_sent != prenewsent:
                    new_splits.append(new_sent.strip())
                else:
                    verb = None
                    if _split[0].endswith("s"):
                        verb = _split[0][:-1].lower()
                    if verb:
                        try:
                                newverb = getInflection(verb,"VB")[0]
                                new_sent = "Attacker can " + newverb + " " + " ".join(_split[1:])
                                new_splits.append(new_sent.strip())
                        except:
                            new_splits.append(sent)
                    else:
                        new_splits.append(sent)
            else:   
                new_splits.append(sent)
        else:
            new_splits.append(sent)
    return_ = "\n".join(new_splits)
    # return_ = re.sub(r"[\n]+", "\n", return_)
    return return_.strip()
def remove_before_after(text):
    re_pattern = r"\b(before|after).*?(,|\.|\n)"
    text = re.sub(re_pattern, "", text)
    return text
# def experiment_coref(text):
#     doc = nlp(text)
#     coref = nlp.add_pipe("experimental_coref")
# # This usually happens under the hood
#     processed = coref(doc)
#     return processed._.coref_resolved
def coref_text(text):
    if "coreferee" not in nlp.pipe_names:
            print("adding coreferee to pipeline")
            nlp.add_pipe("coreferee")

    doc = nlp(text)
    tokens = [t for t in doc]
    replacement = [""]*len(tokens)
    coref= doc._.coref_chains
    for chain in coref:
        main_index = chain[chain.most_specific_mention_index]
        if len(main_index) > 1:
            continue
        main = doc[main_index[0]]
        main_start = -1

        for n in doc.noun_chunks:
            if n.root.i == main.i:
                main_start = n.start_char
                main_end = n.end_char
                break
        if main_start == -1:
            main_start = main.idx
            main_end = main.idx + len(main.text)
        main_text = text[main_start:main_end]
        if "\\" in main_text:
            continue
        if main_text.lower().startswith("a "):
            main_text = main_text.replace("a ", "the ")
        if main_text.lower().startswith("an "):
            main_text = main_text.replace("an ", "the ")
        # main_start = main.idx
        # main_end = main.idx + len(main.text)
        for mention in chain:
            if len(mention) > 1:
                continue
            if mention[0] == main_index[0]:
                continue
            replaced = doc[mention[0]]
            if replaced.text.lower() not in ["it","he","she","him","her","they","them"]:
                continue
            replacement[replaced.i] = main_text
    new_text = ""
    for i in range(0, len(tokens)):
        if replacement[i] != "":
            new_text += replacement[i]
        else:
            new_text += tokens[i].text
        if bool(tokens[i].whitespace_):
            new_text += " "
    return new_text
def coref_resolution(text):
    texts = text.split("\n")
    new_texts = []
    for t in texts:
        new_texts.append(coref_text(t))
    text =  "\n".join(new_texts)
    doc = nlp(text)
    sents = list(doc.sents)
    new_texts = []
    for sent in sents:
        new_texts.append(coref_text(sent.text))
    return " ".join(new_texts)
def subject_elipsis(text:str):
    doc = nlp(text)

    sents = list(doc.sents)
    if len(sents) == 0:
        return text
    sent = sents[0]
    token = sent.root
    flag = False
    if token.pos_ != "VERB":
         return text
    if hasattr(token, "lefts"):# if the token has lefts
        for t in token.lefts:
            if t.dep_ in ["nsubj", "nsubjpass", "csubj", "csubjpass"]:# if the lefts has nsubj
                flag = True
                break
    
    if not flag and token.i == 0:
        verb = token.lemma_
        new_sent = "Attacker can " + verb +" "+ doc[token.i+1:].text
        return new_sent
    return text
functions = ['fix_unicode','remove_link_and_citations','remove_explicit_entity','delete_brackets', 'pass2acti', 'coref_', 'wild_card_extansions', 'try_to', 'is_capable_of', 'ellipsis_subject']
functions_dict ={
    'fix_unicode': fix_unicode,
    'homogenization': homogenization,
    'CـC': CـC,
    'handling_substitutions': handling_substitutions,
    'remove_non_ascii': remove_non_ascii,
    'remove_common_word': remove_common_word,
    'remove_link_and_citations': remove_link_and_citations,
    'remove_explicit_entity': remove_explicit_entity,
    'fix_enumeration': fix_enumeration,
    'coref_': coref_resolution,
    'ellipsis_subject': subject_elipsis

}

def text_preprocessing(txt, paragraph_functions= ['fix_unicode',"handling_substitutions",  "CـC", "homogenization", 'ellipsis_subject'], flag = True):
    txt = fix_enumeration(txt)
    mitre_specific = ['remove_link_and_citations','remove_explicit_entity']
    for func in mitre_specific:
        txt = functions_dict[func](txt)
    if flag:
        txt = coref_resolution(txt)

    
    new_txt = ""

    doc = nlp(txt)
    sentences = list(doc.sents)
    for sent in sentences:
        new_sent_text = sentence_processing(sent.text, False, paragraph_functions,flag = flag)
        new_txt += new_sent_text + " "
    return new_txt.replace("  ", " ").strip()

def sentence_processing(sent,exhaustive=False, paragraph_functions= ['fix_unicode',"handling_substitutions",  "CـC", "homogenization",'ellipsis_subject'], flag = True):
    # if not flag:
    #     sent = remove_before_after(sent)
    for func in paragraph_functions:
        sent = functions_dict[func](sent)
    return sent

