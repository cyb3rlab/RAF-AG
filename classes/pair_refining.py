from mitre_attack import *
import itertools
mobile_tactics = ["lateral-movement", "initial-access"]
platform_mapper ={}
platform_mapper["Windows"] = "Windows"
platform_mapper["Linux"] = "Linux"
platform_mapper["macOS"] = "macOS"
platform_mapper["Office 365"] = "CloudService"
platform_mapper["SaaS"] = "CloudService"
platform_mapper["IaaS"] = "CloudService"
platform_mapper["Azure AD"] = "CloudService"
platform_mapper["Google Workspace"] = "CloudService"
platform_mapper["Containers"] = "Containers"
platform_mapper["Network"] = "NetworkDevice"
platform_mapper["PRE"] = "WAN"
def is_mobile(input_tactic):
    """check the type of input technique whether it is fixed type or mobile type"""

    if input_tactic in mobile_tactics:
            return True
    return False

def get_platform(input):
    """get the platform of the input technique"""
    original_platforms = MitreAttack.get_platforms(input)
    convert_platforms = []
    for p in original_platforms:
        if p in platform_mapper:
            convert_platforms.append(platform_mapper[p])
    return list(set(convert_platforms))

def pair_refine(input):
    pairs =[]
    source, target = input
    source_tactics = MitreAttack.get_tactics(source)
    target_tactics = MitreAttack.get_tactics(target)
    combined_tactics = list(itertools.product(source_tactics, target_tactics))
    for c in combined_tactics:
        source_type = is_mobile(c[0])
        target_type = is_mobile(c[1])
        if not source_type and not target_type:
            new_pairs = fixed_refine(source, target)
            pairs.extend(new_pairs)
        if not source_type and target_type:
            new_pairs = fixed2mobile_refine(source, target)
            pairs.extend(new_pairs)
        if source_type and not target_type:
            new_pairs = mobile2fixed_refine(source, target)
            pairs.extend(new_pairs)
    unique_pairs = []
    unique_ids = []
    for p in pairs:
        id_ = p["source"] + "__" + p["target"]+ + "__" + p["source_platform"]+ "__" + p["target_platform"]
        if id_ not in unique_ids:
            unique_ids.append(id_)
            unique_pairs.append(p)
    return unique_pairs


def fixed_refine(source, target):
    pairs = []
    source_platforms = get_platform(source)
    target_platforms = get_platform(target)
    for platform in source_platforms:
        if platform in target_platforms:
            pairs.append({"source": source,"source_platform": platform, "target": target,"target_platform": "same_"+platform , "type": "refinement"})

    return pairs

def mobile2fixed_refine(source, target):
    """source is mobile, target is fixed, no change in platform since we already in new platform"""
    pairs = []
    source_platforms = get_platform(source)
    target_platforms = get_platform(target)
    for platform in source_platforms:
        if platform in target_platforms:
            pairs.append({"source": source,"source_platform": platform, "target": target,"target_platform": "same_"+platform , "type": "refinement"})
    return pairs

def fixed2mobile_refine(source, target):
    """source is fixed, target is mobile, platform will change"""
    pairs = []
    source_platforms = get_platform(source)
    target_platforms = get_platform(target)
    for platform in source_platforms:
        for platform2 in target_platforms:
            if platform2 == platform:
                pairs.append({"source": source,"source_platform": platform, "target": target,"target_platform": "other_" + platform2 , "type": "propagation"})
            else:
                pairs.append({"source": source,"source_platform": platform, "target": target,"target_platform": platform2 , "type": "propagation"})
    return pairs