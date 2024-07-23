from spacy import displacy
from language_models import *
from classes.subject_verb_object_extract import findSVOs

def sentence_view(sentence, port = 5000):
    doc = nlp(sentence)
    displacy.serve(doc, style="dep", port = port)


def action_extraction_per_sentence(sentence:str="", _doc = None):
    sentence.strip(" ")
    assert sentence != "" or _doc != None, "sentence or doc must be not empty"
    if _doc != None:
        doc = _doc
    else:
        doc = nlp(sentence)

    svos, chains = findSVOs(doc)
    temp_dict = dict()
    for x in svos:
        _key = x[0]+ " " + x[1]["text"] + " " + x[2]["text"] if isinstance(x[0],str) else x[0]["text"]+ " " + x[1]["text"] + " " + x[2]["text"]
        temp_dict[_key] = x
    mylist = list(temp_dict.values())
    return doc, mylist, chains

def split_clauses(doc):
    
    seen = set() # keep track of covered words
    chunks = []
    for sent in doc.sents:
        heads = [cc for cc in sent.root.children if cc.dep_ == 'advcl']

    for head in heads:
        words = [ww for ww in head.subtree]
        for word in words:
            seen.add(word)
        chunk = (' '.join([ww.text for ww in words]))
        chunks.append(  chunk )

    unseen = [ww for ww in sent if ww not in seen]
    chunk = ' '.join([ww.text for ww in unseen])
    chunks.append(  chunk )

    return chunks
