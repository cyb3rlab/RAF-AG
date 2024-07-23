from mitreattack.stix20 import MitreAttackData

class MitreAttack:
    type_id_map = {}
    id_type_map = {}
    MitreId_2_StixID = {}
    StixID_2_MitreId = {}
    sub_technique_map = {}
    mitre_attack_data = None
    @classmethod
    def initialize(cls, mitre_attack_file):
        cls.mitre_attack_data = MitreAttackData(mitre_attack_file)
        techniques = cls.mitre_attack_data.get_techniques(include_subtechniques=True, remove_revoked_deprecated=True)
        data = cls.mitre_attack_data.get_all_parent_techniques_of_all_subtechniques()
        for item in techniques:
            encoded_name = item.name
            
            cls.type_id_map[encoded_name] = item.external_references[0].external_id
            cls.id_type_map[item.external_references[0].external_id] = encoded_name
            cls.MitreId_2_StixID[item.external_references[0].external_id] = item.id
            cls.StixID_2_MitreId[item.id] = item.external_references[0].external_id
        for k,v in data.items():
            assert len(v) == 1
            technique = k
            mitre_id = cls.mitre_attack_data.get_attack_id(technique)
            # mitre_id = mitre_attack_data.get_technique_by_name(technique).external_references[0].external_id
            parent_technique = v[0]["object"].id
            parent_mitre_id = cls.mitre_attack_data.get_attack_id(parent_technique)
            cls.sub_technique_map[mitre_id] = parent_mitre_id


    @classmethod
    def get_techniques_by_tactic(cls, tactic):
        techniques = cls.mitre_attack_data.get_techniques_by_tactic(tactic, domain="enterprise", include_subtechniques=True, remove_revoked_deprecated=True)
        return [t.name for t in techniques]
    @classmethod
    def get_platforms(cls, tech_id):
        tech_obj = cls.mitre_attack_data.get_object_by_attack_id(tech_id, "attack-pattern")
        return tech_obj.x_mitre_platforms
    @classmethod
    def get_technique_name(cls, tech_id):#T1204 => "User Execution"
        return cls.id_type_map[tech_id]
    @classmethod
    def get_tactics(cls, tech_id):
        tech_obj = cls.mitre_attack_data.get_object_by_attack_id(tech_id, "attack-pattern")
        
        return [ kc["phase_name"] for kc in tech_obj.kill_chain_phases]

    @classmethod
    def get_tech_obj(cls, tech_id):
        
        tech_obj = cls.mitre_attack_data.get_object_by_attack_id(tech_id, "attack-pattern")
        return tech_obj

    @classmethod
    def is_sub_technique(cls, mitre_id):
        return mitre_id in cls.sub_technique_map
    

    @classmethod
    def is_parent_technique(cls, mitre_id):
        if mitre_id in cls.sub_technique_map.values():
            return True

    
    @classmethod
    def is_childless_parent(cls, mitre_id):
        """
        subtechnique map is a dict of {subtechnique: parent_technique}
        if mitre_id is a parent technique, then it should be in the values of subtechnique map
        if mitre_id is a subtechnique, then it should be in the keys of subtechnique map
        if mitre_id is a childless parent technique, then it should not be in the keys of the dict map or in the values of the dict
        """
        
        if mitre_id not in cls.sub_technique_map.values()  and mitre_id not in cls.sub_technique_map:
            return True
        return False
            

    @classmethod
    def get_parent_technique_id(cls, mitre_id):
        return cls.sub_technique_map[mitre_id]
    
    @classmethod
    def get_children_technique_ids(cls, mitre_id):
        children = []
        for k,v in cls.sub_technique_map.items():
            if v == mitre_id:
                children.append(k)
        return children
    
    @classmethod
    def get_procedure(cls, _id):
        procedures = []
        rel = cls.mitre_attack_data.get_objects_by_type("relationship")
        for r in rel:
           if r.target_ref == _id and r.relationship_type == "uses":
               procedures.append({"id":r.id , "description": r.description})
        return procedures
    

    @classmethod
    def get_group_software_campaign_url(cls):
        
        
        rel = cls.mitre_attack_data.get_objects_by_type("relationship")
        data = dict()
        for r in rel:
            url_data = list()
            if "external_references" not in r:
                continue
            if r.relationship_type == "uses" and r.target_ref.startswith("attack-pattern"):
               for u in r.external_references:
                   url_data.append(u.url)
            data[r.id] = url_data
        return data
    @classmethod
    def get_campaign_url(cls):
        rel = cls.mitre_attack_data.get_objects_by_type("relationship")
        data = dict()
        for r in rel:
            url_data = list()
            if "external_references" not in r:
                continue
            if r.relationship_type == "uses" and r.target_ref.startswith("attack-pattern") and r.source_ref.startswith("campaign"):
                for u in r.external_references:
                   url_data.append(u.url)
                data[r.id] = url_data
        return data

    @classmethod
    def get_campagin_url_for_techniques(cls, techniques: list = []):
        if len(techniques) == 0:
            return cls.get_campaign_url()
        
        rel = cls.mitre_attack_data.get_objects_by_type("relationship")
        data = dict()
        for r in rel:
            url_data = list()
            if "external_references" not in r:
                continue
            if r.relationship_type == "uses" and r.target_ref.startswith("attack-pattern") and r.source_ref.startswith("campaign"):
                
                for u in r.external_references:
                   url_data.append(u.url)
                data[r.id] = url_data
        return data
    @classmethod
    def get_all_url(cls):
        unique = []
        rel = cls.mitre_attack_data.get_objects_by_type("relationship")
        data = dict()
        for r in rel:
            url_data = list()
            if "external_references" not in r:
                continue
            if r.relationship_type == "uses":
                
                for u in r.external_references:
                   unique.append(u.url)
                   url_data.append(u.url)
                data[r.id] = url_data
        unique = list(set(unique))
        return unique
    
    @classmethod
    def get_procedures(cls):
        rel = cls.mitre_attack_data.get_objects_by_type("relationship")
        procedures = [r for r in rel if (r.relationship_type == "uses" and r.target_ref.startswith("attack-pattern"))]
        rows = []
        for p in procedures:
            _id = p.id
            # print(_id)
            description = p.description
            tech_id = cls.StixID_2_MitreId[p.target_ref]
            platform = cls.get_platforms(tech_id)
            rows.append( { "tech_id": tech_id, "platform": platform, "id": _id, "description": description})
        return rows


MitreAttack.initialize(r"data/enterprise-attack.json")