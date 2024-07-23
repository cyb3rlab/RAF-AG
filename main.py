import os
os.environ["TFHUB_CACHE_DIR"] = "./data/tf_hub"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from classes.managment import Manager
from multiprocessing import set_start_method
from classes.procedure import Procedure
from classes.finalizing import stat_Calculator
# from mitre_attack import MitreAttack
from modules import *
import fire

def main(campaign_from_0:bool=True, procedure_from_0:bool=False, technique_from_0:bool=False, techniue_alignment_from_0:bool=True,):
    Manager(campaign_from_0= campaign_from_0, procedure_from_0=procedure_from_0, technique_from_0= technique_from_0,techniue_alignment_from_0=techniue_alignment_from_0,
            matching_from_0=True, context_similarity_from0=True, multiprocessing=False,do_procedure_deduplication= False)

if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except:
        print("context already set")
    # Manager(campaign_from_0= True, procedure_from_0=True, technique_from_0= True,techniue_alignment_from_0=True,
    #         matching_from_0=True, context_similarity_from0=True, multiprocessing=False,do_procedure_deduplication=True)
    fire.Fire(main)
    # Manager(campaign_from_0= True, procedure_from_0=False, technique_from_0= False, techniue_alignment_from_0=True,
    #         matching_from_0= True, context_similarity_from0=True, multiprocessing=False,do_procedure_deduplication=False)
    # Manager(campaign_from_0= False, procedure_from_0=False, technique_from_0= False,
    #         matching_from_0=False, context_similarity_from0=False, multiprocessing=False,do_procedure_deduplication=False)
    # Manager(campaign_from_0= False, procedure_from_0=False, technique_from_0= False, techniue_alignment_from_0=False,
    #         matching_from_0=False, context_similarity_from0=True, multiprocessing=False,do_procedure_deduplication=False)
    # _decoding_dir = r"data/campaign/decoding_result"
    # stat_dir = r"data/statistics"
    # stat_cal = stat_Calculator(attack_path_dir=_decoding_dir, saved_dir=stat_dir)
    # stat_cal.calculate_frequency()
    # stat_cal.calculate_pair_propapility()
    # a = stat_cal.get_top_K_common_tech()
    # b = stat_cal.get_top_K_common_pair()

    # stat_cal.save()


