from keys import Keys
import spacy
import os

try:
    ner = spacy.load(Keys.NER_MODEL)

    print("load ner model successfully")
except:
    newpath = os. getcwd() +"/"+ Keys.NER_MODEL
    ner = spacy.load(newpath)
    print("load ner model successfully")
def ner_extract(text):
        doc = ner(text)
        ents = doc.ents
        entities = list()
        for ent in ents:
            data = dict()
            data["start"] = ent.start_char
            data["end"] = ent.end_char
            data["text"] = ent.text
            data["label"] = ent.label_
            entities.append(data)
        return entities