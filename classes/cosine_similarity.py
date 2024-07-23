from evaluate import load
# bertscore = load("bertscore")
from keys import Keys
import pickle
import json
from tqdm import tqdm
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# model = gensim.models.KeyedVectors.load_word2vec_format(Keys.WORD2VEC, binary=True)
import tensorflow_hub as hub
import os
os.environ["TFHUB_CACHE_DIR"] = "./data/tf_hub"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5") 
class CosineSimilarity:
    # def __init__(self, model_name = "xlnet-large-cased"):
    #     self.f1 = {}
    #     self.model_name = model_name
    def __init__(self):
        self.f1 = {}
       
    @classmethod
    def from_pickle(cls, file_name):
        with open(file_name, "rb") as f:
            data = pickle.load(f)
            object = cls()
            object.f1 = data
            return object
    def to_pickle(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.f1, f)
    def add(self, id1, id2, f1):
        if id1 not in self.f1:
            self.f1[id1] = {}
        self.f1[id1][id2] = f1
        if id2 not in self.f1:
            self.f1[id2] = {}
        self.f1[id2][id1] = f1
    
    #get f1 score between phrases
    def getf1(self, id1, id2):
        try:
            if id1 in self.f1:
                return self.f1[id1][id2]
            if id2 in self.f1:
                return self.f1[id2][id1]
        except:
            # return self.compute([id1], [id2])
            # because we calculate the similarity before hand and save it in a file
            # so if we can't find the similarity between two phrases, return 0. we do not want to load new model herer
            return 0
        return 0
        
    # #compute f1 score between procedures
    # def compute(self, input_texts, target_texts):
    #     f1 = []
    #     rs = bertscore.compute(predictions=input_texts, references=target_texts, model_type=self.model_name, nthreads=8 ,lang="en", device = None, batch_size=16)
    #     f1 = rs["f1"]
    #     for i in range(len(input_texts)):
    #         self.add(input_texts[i], target_texts[i], f1[i])
    

    # def compute_range2(self, predictions, references, window_size = 100000, flag = True):
    #     if flag:
    #         combinations = list(itertools.product(predictions, references))
    #     else:
    #         combinations = list(zip(predictions, references))
    #     combinations2 = [c for c in combinations if self.getf1(c[0], c[1]) == 0]
    #     combinations = combinations2
    #     for i in tqdm(range(0, len(combinations), window_size)):
    #             end = i + window_size if i + window_size < len(combinations) else len(combinations)
    #             data1 =[k[0] for k in combinations[i:end]]
    #             data2 =[k[1] for k in combinations[i:end]]
    #             self.compute(data1, data2)
    
    def compute_range(self, predictions, references, window_size = 100000, flag = True):
        # predict_ = []
        # reference_ = {}
        # for p in predictions:
        #     predict_.append(p)
        #     for r in references:
        #         if self.getf1(p, r) == 0 and r not in reference_:  
        #             reference_[r] = 1
        # predictions = predict_
        # references = list(reference_.keys())
        predict_embed = np.array(model(predictions))
        reference_embed = np.array(model(references))
        if flag:
            combinations = list(itertools.product(predictions, references))
            embed_combinations = list(itertools.product(predict_embed, reference_embed))
        else:
            combinations = list(zip(predictions, references))
            embed_combinations = list(zip(predict_embed, reference_embed))
        # for i in tqdm(range(0, len(combinations))):
        #         if self.getf1(combinations[i][0], combinations[i][1]) != 0:
        #             continue
        #         value = cosine_similarity(embed_combinations[i][0].reshape(1,-1), embed_combinations[i][1].reshape(1,-1))[0][0]
        #         self.add(combinations[i][0], combinations[i][1], value)
        cosine_values = cosine_similarity(predict_embed, reference_embed)
        for i in range(len(predictions)):
            for j in range(len(references)):
                self.add(predictions[i], references[j], cosine_values[i][j])
    # def compute_range2(self, predictions, references, window_size = 100000, flag = True):
    #     # predict_ = []
    #     # reference_ = {}
    #     # for p in predictions:
    #     #     predict_.append(p)
    #     #     for r in references:
    #     #         if self.getf1(p, r) == 0 and r not in reference_:  
    #     #             reference_[r] = 1
    #     # predictions = predict_
    #     # references = list(reference_.keys())
    #     predict_embed = np.array(model(predictions))
    #     reference_embed = np.array(model(references))
    #     if flag:
    #         combinations = list(itertools.product(predictions, references))
    #         embed_combinations = list(itertools.product(predict_embed, reference_embed))
    #     else:
    #         combinations = list(zip(predictions, references))
    #         embed_combinations = list(zip(predict_embed, reference_embed))
    #     for i in tqdm(range(0, len(combinations))):
    #             if self.getf1(combinations[i][0], combinations[i][1]) != 0:
    #                 continue
    #             value = cosine_similarity(embed_combinations[i][0].reshape(1,-1), embed_combinations[i][1].reshape(1,-1))[0][0]
    #             self.add(combinations[i][0], combinations[i][1], value)
        # cosine_values = cosine_similarity(predict_embed, reference_embed)
        # for i in range(len(predictions)):
        #     for j in range(len(references)):
        #         self.add(predictions[i], references[j], cosine_values[i][j])
    # def cosine_sim(self, w1, w2):
    #     value = 0
    #     try:
    #         value = model.similarity(w1,w2)
    #     except:
    #         w1 = w1.split()[-1]
    #         w2 = w2.split()[-1]
    #         try:
    #             value = model.similarity(w1,w2)
    #         except:
    #             pass
    #     return value

    def get_similarity_raw(self, input1, input2):
        predictions = [input1]
        references = [input2]
        self.compute_range(predictions, references, window_size = 1, flag = False)
        return self.getf1(input1, input2)



    def get_similarity(self, input1, input2):
        max_f1 = 0.0
        for i in range(len(input1)):
            for j in range(len(input2)):
                f_value = self.getf1(input1[i], input2[j])
                if f_value > max_f1:
                    max_f1 = f_value
        return max_f1
    
