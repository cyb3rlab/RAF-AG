
from language_models import nlp
import itertools
from modules import remove_words
# dependency markers for subjects
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
# dependency markers for objects
OBJECTS = {"dobj", "dative", "attr", "oprd","pobj"}
# POS tags that will break adjoining items
BREAKER_POS = {"CCONJ", "VERB"}
NOUNS = {"NOUN", "PROPN", "PRON", "X","NUM"}
NOUNS2 = {"NOUN", "PROPN", "X","NUM"}
# words that are negations
NEGATIONS = {"no", "not", "n't", "never", "none", "without","neither","nor"}
via_list = ["by", "via", "through","over", "after", "before"]
precede_list = ["by", "via", "through", "after", "once"] # the action precedes the main action, "by doing this, we can..."
successor_list = ["before"] # the action succeeds the main action, "before doing this, we need to..."


def generate_svo(sub_, verb_, obj_):
    sub= sub_
    verb = verb_
    verb_text = verb["text"]
    obj = obj_
    if verb_text in ["in","on","at","for","inside","of"]:
        return (sub, verb, obj)
    obj["verbs"] = [verb_text]
    return (sub, verb, obj)
    

np_dictionary = dict()
def contains_conj(depSet):
    return "and" in depSet or "or" in depSet or "nor" in depSet or \
           "but" in depSet or "yet" in depSet or "so" in depSet or "for" in depSet


# get subs joined by conjunctions
def _get_subs_from_conjunctions(subs):
    more_subs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ in NOUNS])
            if len(more_subs) > 0:
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs


# get objects joined by conjunctions
def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_objs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ in NOUNS])
            if len(more_objs) > 0:
                more_objs.extend(_get_objs_from_conjunctions(more_objs))
    return more_objs


# find sub dependencies
def _find_subs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ not in NOUNS and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ in SUBJECTS]
        if len(subs) > 0:
            verb_negated = _is_negated(head)
            subs.extend(_get_subs_from_conjunctions(subs))
            return subs, verb_negated
        elif head.head != head:
            return _find_subs(head)
    elif head.pos_ in NOUNS:
        return [head], _is_negated(tok)
    return [], False


# is the tok set's left or right negated?
def _is_negated(tok):
    if not hasattr(tok, "children"):
        return False
    parts = list(tok.children)
    for dep in parts:
        if dep.lower_ in NEGATIONS:
            return True
    return False


# get all the verbs on tokens with negation marker
def _find_svs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


# get grammatical objects for a given set of dependencies (including passive sentences)
def _get_objs_from_prepositions(deps, is_pas):
    objs = []
    for dep in deps:
        if dep.lemma_ in via_list:
            continue
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or (is_pas and dep.dep_ == "agent")):
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or
                         (tok.pos_ == "PRON" and tok.lower_ == "me") or
                         (is_pas and tok.dep_ == 'pobj')])
    return objs

# get objects from the dependencies using the attribute dependency
def _get_objs_from_attrs(deps, is_pas):
    for dep in deps:

        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(_get_objs_from_prepositions(rights, is_pas))
                    if len(objs) > 0:
                        return v, objs
    return None, None






# get all functional subjects adjacent to the verb passed in
def _get_all_subs(v):
    verb_negated = _is_negated(v)
    lefts = list(v.lefts)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    # subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS]

    if len(subs) > 0:
        _subs = []
        for sub in subs:
            _subs.extend(_get_conj_noun(sub))
        subs = _subs
    else:
        foundSubs, verb_negated = _find_subs(v)
        subs.extend(foundSubs)
    return subs, verb_negated


#allowing tobe words
def _is_non_aux_verb(tok):
    x = tok.pos_
    y = tok.dep_
    z = tok.tag_
    if tok.lemma_ in ["be"]:
        return True
    # print(tok,x,y,z)
    return tok.pos_ == "VERB" and (tok.dep_ != "aux" or tok.dep_ != "auxpass")


# return the verb to the right of this verb in a CCONJ relationship if applicable
# returns a tuple, first part True|False and second part the modified verb if True
def _right_of_verb_is_conj_verb(v):
    # rights is a generator
    rights = list(v.rights)

    # VERB CCONJ VERB (e.g. he beat and hurt me)
    if len(rights) > 1 and rights[0].pos_ == 'CCONJ':
        for tok in rights[1:]:
            if _is_non_aux_verb(tok):
                return True, tok

    return False, v


# get all objects for an active/passive sentence
def _get_all_objs(v, is_pas):
    # rights is a generator
    if not is_pas:
        rights = list(v.rights)
        objs = list()
        for tok in rights:
            if tok.dep_ in OBJECTS and tok.text != "to":
                objs.append(tok)
        # print(1)       
        index = v.i
        for tok in rights:
            if tok.dep_ =="prep" and  tok.i == index + 1 and tok.pos_ == "ADP" and tok.lemma_ not  in ["like", "as","include", "contain"]:
                if hasattr(tok, "rights"):
                    for child in tok.rights:
                        if child.dep_ in OBJECTS:
                                objs.append(child)
        objs = list(set(objs))
        _objs = []
        for obj in objs:
            _objs.extend(_get_conj_noun(obj))
        objs = _objs

    else: # passive
        objs = get_pure_gent_objs(v)
        index = v.i
        rights = list(v.rights)
        _objs = []
        for obj in objs:
            _objs.extend(_get_conj_noun(obj))
        objs = _objs
    potential_verb = v
    return potential_verb, objs


# return true if the sentence is passive - at he moment a sentence is assumed passive if it has an auxpass verb
def _is_passive(tokens):
    for tok in tokens:
        if tok.dep_ == "auxpass":
            return True
    return False

def _is_passive_verb(token):
    if not hasattr(token, 'lefts'):
        return False
    for l in token.lefts:
        if l.dep_ == "auxpass":
            return True
    return False
# resolve a 'that' where/if appropriate
def _get_that_resolution(toks):
    for tok in toks:
        if 'that' in [t.orth_ for t in tok.lefts]:
            return tok.head
    return toks


# simple stemmer using lemmas
def _get_lemma(word):
    tokens = nlp(word)
    if len(tokens) == 1:
        return tokens[0].lemma_
    return word

def _expand_verb(verb):
    if not hasattr(verb, 'rights'):
        return verb.lemma_
    index = verb.i
    for r in verb.rights:
        if r.dep_ == "prep" and r.pos_ == "ADP" and r.i == index+1:
            return verb.lemma_ + " " + r.lemma_
    return verb.lemma_
# print information for displaying all kinds of things of the parse tree
def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])

def winden(noun, doc):
    tokens  = [noun.i]
    stacks = []
    if hasattr(noun, "lefts"):
        for token in noun.lefts:
            tokens.append(token.i)
            stacks.append(token)
    while len(stacks) > 0:
        token = stacks.pop()
        if hasattr(token, "lefts"):
            for t in token.lefts:
                tokens.append(t.i)
                stacks.append(t)
        if hasattr(token, "rights"):
            for t in token.rights:
                tokens.append(t.i)
                stacks.append(t)
    tokens = list(set(tokens))
    tokens = sorted(tokens)
    tokens = [doc[i] for i in tokens]
    return tokens
# good_adj = [,"safe","remote","virtual","hidden","masqueraded","keylogged","signed","scheduled","legitimate","benign","encrypted","encoded","deleted","decoded","anti"]
def chunk_analysis(item, chunk,doc):
    data = dict()
    subtree = []
    flag = False
    for tok in chunk:
        if flag:
            subtree.append(tok)
            continue
        else:
            if (tok.pos_ == "ADV" and tok.dep_ == "advmod") or tok.pos_ == "DET":
                        continue
            if tok.text.lower() in remove_words:
                continue
            else:
                subtree.append(tok)
                flag = True
            # if (tok.pos_ in NOUNS2 or tok.pos_ == "PART" or tok.lemma_.startswith(".") or tok.dep_ in ["compound"] or tok.text[0].isupper()):
            #         if (tok.pos_ == "ADV" and tok.dep_ == "advmod") or tok.pos_ == "DET":
            #             continue
            #         subtree.append(tok)
            #         flag = True
    if len(subtree) == 0:
        data['text'] = item.text
        data["start"] = item.idx
        data["end"] = item.idx + len(item.text)
        data["id"] = item.i + 1
        np_dictionary[(item.i +1)] = data
        return data
    data["start"] = subtree[0].idx
    for index in range(0, len(subtree)):
                    s = subtree[index]
                    if s.pos_ == "PART" and s.text in ["'s","'"]:
                        if index + 1 < len(subtree):
                            data["start"] = subtree[index + 1].idx
                            break
                        else:
                            break
        
    data["end"] = subtree[-1].idx + len(subtree[-1].text)
    data['text'] = doc.char_span(data["start"], data["end"]).text
    data["id"] = item.i + 1
    np_dictionary[(item.i +1)] = data
    return data
# expand an obj / subj np using its chunk
def _expand_np(item, doc, flag=True):
    data = dict()
    if (item.i +1) in np_dictionary:
        return np_dictionary[(item.i +1)].copy()

    if flag:
        noun_chunks = list(doc.noun_chunks)
        for chunk in noun_chunks:
            if chunk.root.i == item.i:
                if chunk.text.endswith("initial access") or chunk.text.endswith("lateral movement"):
                    data["start"] = chunk.start_char
                    data["end"] = chunk.end_char
                    data['text'] = chunk.text
                    data["id"] = item.i + 1
                    np_dictionary[(item.i +1)] = data
                    return data
                data = chunk_analysis(item, chunk,doc)
                if len(data) > 0:
                    return data
        #this is the case the noun not match with any noun chunk       
        chunk = winden(item, doc)
        data = chunk_analysis(item, chunk,doc)
        if len(data) > 0:
            return data
    else:#default case
        data['text'] = item.text
        data["start"] = item.idx
        data["end"] = item.idx + len(item.text)
        data["id"] = item.i + 1
        np_dictionary[(item.i +1)] = data
        return data

def expand(item, tokens, visited):
    if item.lower_ == 'that':
        item = _get_that_resolution(tokens)

    parts = []

    if hasattr(item, 'lefts'):
        lefts = list(item.lefts)
        for part in lefts:
            if not part.lower_ in NEGATIONS:
                if part.pos_ in NOUNS and part.dep_ == "compound":
                    parts.insert(0, part)

    parts.append(item)

    if hasattr(item, 'rights'):
        for part in item.rights:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                if part.pos_ in NOUNS and part.dep_ == "compound":
                    parts.append(part)

    if hasattr(parts[-1], 'rights'):
        for item2 in parts[-1].rights:
            if item2.pos_ == "DET" or item2.pos_ == "NOUN":
                if item2.i not in visited:
                    visited.add(item2.i)
                    parts.extend(expand(item2, tokens, visited))
            break

    return parts


# convert a list of tokens to a string
def to_str(tokens):
    return ' '.join([item.text for item in tokens])

verb_pattern = ["embed",'make', 'try' ,'add','run','delete','remove', 'register','replicate', 'create','execute','modify','download','spread','conduct', 'copy','fork','write','read','retrieve','redirect','wipe','exfiltrate','install','deploy','clear', 'erase','drop','entrench','run', 'collect','write','locate','allocate','clone','use','perform','spawn','issue','set','clone','execute','launch','save', 'add','extract','get','inject','obtain', 'gather', 'download','beacon','place', 'navigate',  'compose','acquire','browse', 'perform', 'open', 'send',  'target',  'accept',  'receive', 'transfer',  'invoke', 'modifies', 'connect',  'communicate',  'post',  'propagate', 'terminate',  'monitor', 'attempt', 'generate',  'search', 'contain',  'hide',  'infect',  'append',  'close',  'check',"exploit"]
# find verbs and their subjects / objects to create SVOs, detect passive/active sentences
command_verbs = ["make","rely","command", "ask", "lure", "persuade","tell","advice","entice","invite","order","recommend","remind","require","suggest","urge","warn","beg","dare","encourage","expect","force","instruct","need","oblige","permit","request"]
users =["users", "user","host", "victim","victims", "target", "targets", "client", "clients", "customer", "customers", "person", "people" ]
def findSVOs(tokens):
    np_dictionary.clear() # clear the dictionary for new sentence
    svos = []
    _verbs = [tok for tok in tokens if _is_non_aux_verb(tok)]
    # @kia
    if _verbs == []:
        _verbs = [tok for tok in tokens if (tok.lemma_ in verb_pattern  or tok.lemma_ in command_verbs)]
        
    visited_np = list()  # recursion detection
    for verb in _verbs:
        svos.extend(_haveget_something_done(verb,tokens))
    for verb in _verbs:
        
        if verb.lemma_ in ["as","like","include","contain"] and verb.dep_ == "prep":
            continue
        is_pas = _is_passive_verb(verb)
        subs, verbNegated = _get_all_subs(verb)
        # if len(subs) == 0, we assume the verb is conj verb, so we need to find the head of the verb
        # verb, objs = _get_all_objs(verb, is_pas)
        if len(subs) == 0:
            head = verb

            # if verb.dep_ == "conj": we handle it normally
            if head.dep_ == "conj":
                while head.dep_ in ["conj"] and head.pos_ == "VERB":
                    head = head.head
                subs, verbNegated = _get_all_subs(head)

            else:
                if head.dep_ in ["advcl", "xcomp"] and not is_pas:
                    while head.dep_ in ["advcl", "xcomp"] and head.pos_ == "VERB" and head.head.i < head.i:
                        head = head.head
                    # _, _objs = _get_all_objs(head, is_pas)
                    subs, verbNegated = _get_all_subs(head) if head.lemma_ not in command_verbs else _get_all_subs(verb)

        verbs = []
        objs = []
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:

            verb, objs = _get_all_objs(verb, is_pas)

            
            verbs = [verb]
            if len(objs) == 0 and hasattr(verb, 'rights'):
                for r in verb.rights:
                    if r.pos_ == "VERB" and r.dep_ == "conj":
                        _, objs = _get_all_objs(r, is_pas)
                        verbs = [verb, r]
            # if is_pas:
            #     for v in verbs:
            #         subs.extend(get_example_subs(v)) #  when giving examples of the subject in case passive voice
            #via , by doing something

            if len(objs) > 0:
                    
                _svos = [subs, verbs, objs]
                products = list(itertools.product(*_svos))
                # all triplet in the products affected by the same verbNegated
                for p in products:
                    sub = _fix_sub(p[0])
                    v = p[1]
                    obj = p[2]
                    objNegated = _is_negated(obj)
                    if is_pas:  # reverse object / subject for passive
                            svos.append(generate_svo(_expand_np(obj, tokens),
                                         expand_verb(verb = v,negation =(verbNegated or objNegated), index = v.i ), _expand_np(sub, tokens)))                            
                    else:
                            svos.append(generate_svo(_expand_np(sub, tokens),
                                         expand_verb(verb = v,negation = (verbNegated or objNegated),index = v.i ), _expand_np(obj, tokens)))

            if is_pas and len(objs) == 0:  # passive with no object                   
                    for sub in subs:
                        for v in verbs:
                            if (sub.i +1) in np_dictionary:
                                continue
                            svos.append(generate_svo("ANY",
                                         expand_verb(verb =v,negation =verbNegated,index = v.i ), _expand_np(sub, tokens)))
        
            #the below svo is happening after the main action
        if len(verbs) > 0:
                for _v in verbs:
                    verb_from_svos = _get_verb_fromto(_v, tokens)
                    svos.extend(verb_from_svos)
                    x_svos = get_SVO_from_x_comp(_v, tokens)
                    svos.extend(x_svos)

        if len(objs) > 0:
                noun_fromto_svos = _get_svos_noun_from_to(objs, tokens,verbNegated )
                svos.extend(noun_fromto_svos)
    for v in _verbs:
        by_svos = via_handling(v,tokens)
        svos.extend(by_svos)
    _new_svos = get_acl_and_preposition(tokens)
    svos.extend(_new_svos)
    _new_svos = passive_advcl(tokens)
    svos.extend(_new_svos)
    noun_chunks = list(tokens.noun_chunks)
    
    for chunk in noun_chunks:
        svos.extend(based_np_split(chunk))
        svos.extend(get_tactical_svo(chunk))
    _new_svos = as_handling(tokens)
    svos.extend(_new_svos)
    svos.sort(key=lambda tup: tup[1]['index'])
    chains = get_conjuncted_np(noun_chunks)
    return svos, chains

noun_2_verb = {"exfiltration":"exfiltrate", "escalation":"escalate","execution":"execute","collection":"collect", "discovery":"discover","access":"access","evasion":"bypass"}
appropriate_noun = {"exfiltrate":"information","escalate":"privilege","execute":"command","collect":"data", "discover":"information","access":"credential","bypass":"defense"}
def get_tactical_svo(np):
    root = np.root
    sovs = []
    if np.text in ["execution","collection","discovery","evasion", "exfiltration","escalation"]:
        verb_text = noun_2_verb[np.text]
        obj_text = appropriate_noun[verb_text]
        obj = {"text": obj_text, "start":np.start_char, "end": np.start_char + len(np.text), "id": np.root.i }
        verb = expand_verb(verb=None,negation = False,index = root.i ,text = verb_text)
        sovs.append(generate_svo("ANY", verb, obj))
        return sovs
    tokens = [t for t in np if t.pos_ in NOUNS2]
    if len(tokens) < 2:
        return sovs
    text = np.text

    if np.text in ["credential access","defense evasion"] or root.text in ["exfiltration","escalation","execution","collection","discovery"]:
        if hasattr(root, 'lefts'):
            remainder =  text[0:len(text) - len(root.text)].strip()
            if len(remainder)==0:
                return sovs
            obj = {"text": remainder, "start":np.start_char, "end": np.start_char + len(remainder), "id": root.i }
            verb = expand_verb(verb=None,negation = False,index = root.i ,text = noun_2_verb[root.text])
            sovs.append(generate_svo("ANY", verb, obj))
    return sovs
def based_np_split(np):
    root = np.root
    text = np.text
    sovs = []
    if hasattr(root, 'lefts'):
        lefts = list(root.lefts)
        for l in lefts:
            if l.pos_ == "VERB" and l.text.endswith("based"):
                new_text = text.split("based")[0]
                if new_text.endswith("-"):
                    new_text = new_text[:-1]
                obj = {"text": new_text, "start": l.idx, "end": l.idx + len(new_text), "id": l.i + 1}
                verb = expand_verb(verb=l,negation = False,index = l.i ,text = "basedon")
                sub = {"text": np.text, "start": np.start_char, "end": np.end_char, "id": np.root.i + 1}
                sovs.append(generate_svo(sub, verb, obj))
    return sovs
def passive_advcl(tokens):
    toks = [tok for tok in tokens if tok.dep_ == "advcl" and tok.pos_ == "VERB"]
    svos = []
    for t in toks:
        is_pas = _is_passive_verb(t)
        if not is_pas:
            continue
        subs, verbNegated = _get_all_subs(t)
        if len(subs) > 0:
            continue
        if t.head.pos_ == "VERB":   
            subs, verbNegated = _get_all_subs(t.head)
        if len(subs) == 0:
            continue
        _, objs = _get_all_objs(t, is_pas)
        if is_pas and len(objs) > 0:
            verbs = [t]
            _svos = [subs, verbs, objs]
            products = list(itertools.product(*_svos))
                # all triplet in the products affected by the same verbNegated
            for p in products:
                    sub = _fix_sub(p[0])
                    v = p[1]
                    obj = p[2]
                    objNegated = _is_negated(obj)
                    if is_pas:  # reverse object / subject for passive
                            svos.append(generate_svo(_expand_np(obj, tokens),
                                         expand_verb(verb=v,negation = (verbNegated or objNegated),index = v.i ,text = v.lemma_), _expand_np(sub, tokens))) 
        if is_pas and len(objs) == 0:
            verbs = [t]
            objs =  subs
            _svos = [verbs, objs]
            products = list(itertools.product(*_svos))
            for p in products:
                v = p[0]
                obj = p[1]
                objNegated = _is_negated(obj)
                if (obj.i +1) in np_dictionary:
                    continue
                svos.append(generate_svo("ANY",
                                expand_verb(verb=v,text = v.lemma_,negation =(verbNegated or objNegated),index = v.i ), _expand_np(obj, tokens)))
    return svos

before = ["before"]
after = ["after", "once"]
def expand_verb(verb = None, negation= False, index = 0, text = "", flag_ = 0):
    if text == "":
        text = verb.lemma_
    flag = flag_
    if verb is not None:
        if hasattr(verb, 'lefts'):
            position = 0
            for r  in verb.lefts:
                if r.lower_ in before: # A could do it before B could do it, B event is after A event
                    position = 1
                if r.lower_ in after:#  A could do it after B could do it
                    position = -1
            if verb.dep_ == "advcl":
                real_position = verb.head.i # A event
                index = real_position + position if position != 0 else verb.i
        if verb.lemma_ in ["like", "include", "contain"] and verb.dep_ == "prep":
            flag = 1
        if verb.lemma_ == "as" and verb.dep_ == "prep":
            if hasattr(verb, 'lefts'):
                for r in verb.lefts:
                    if r.lemma_ == "such":
                        flag = 1
        if hasattr(verb, 'rights'):
            rights = list(verb.rights)
            for r in verb.rights:
                if r.dep_ == "advmod" and r.lemma_ in ["back", "into", "onto", "from", "away", "off", "out", "up", "down", "in", "over", "through", "across", "around", "behind", "between", "by", "for", "of", "on", "to", "with", "about", "against", "at", "before", "after"]:
                    text =  text + " " + r.lemma_
        if verb.dep_ == "acl" and verb.text in ["named", "called"]:
            flag = 1
        return {"negation": negation, "text": text, "index": index , "flag": flag}
    return {"negation": negation, "text": text, "index": index , "flag": flag}


def via_handling(verb, tokens):
    """handling via, via Ving"""
    if verb.dep_ == "conj" and verb.head.pos_ == "VERB":
            subs, verbNegated = _get_all_subs(verb.head)
    else:
            subs, verbNegated = _get_all_subs(verb)
    if len(subs) == 0:
        _sub = nlp("Attacker")[0]
        subs= [_sub]
    children = list(verb.children)
    svos = []
    position = 0
    for tok in children :
        if tok.dep_ != "prep" or tok.lower_ not in via_list:
            continue
        if tok.lower_ in precede_list:
            position = -1
        if tok.lower_ in successor_list:
            position = 1
        if hasattr(tok, 'rights'):
            for item in tok.rights:
                objs = []
                if item.pos_ == "VERB" and item.dep_ == "pcomp":
                    isConjVerb, conjV = _right_of_verb_is_conj_verb(item)
                    is_pas = _is_passive_verb(item)
                    
                    verbNegated = verbNegated or _is_negated(item) or _is_negated(tok) # "not by using email" or "by not using email"
                    
                    if isConjVerb:
                        v2, objs = _get_all_objs(conjV, is_pas)
                        verbs = [item, v2]
                        
                    else:
                        v, objs = _get_all_objs(item, is_pas)
                        verbs = [v]
                else:
                    
                    if  item.pos_ in NOUNS and item.dep_ == "pobj": #by email or via email
                        objs = _get_conj_noun(item)
                        if len(objs) == 0:
                            objs = [item]
                        _verb = nlp("use")[0]
                        # verbNegated = _is_negated(_verb)
                        verbs= [_verb]
                        is_pas = False
                if len(objs) > 0:
                        _svos = [subs, verbs, objs]
                        products = list(itertools.product(*_svos))
                        for p in products:
                            sub = _fix_sub(p[0])
                            v = p[1]                     
                            obj = p[2]
                            objNegated = _is_negated(obj)
                            if is_pas:  # reverse object / subject for passive
                                svos.append(generate_svo(_expand_np(obj, tokens),
                                         expand_verb(verb=v,text = v.lemma_,negation = (verbNegated or objNegated),index =( v.i+ position )), _expand_np(sub, tokens)))                            
                            else:
                                svos.append(generate_svo(_expand_np(sub, tokens),
                                         expand_verb(verb=v,text = v.lemma_,negation = (verbNegated or objNegated),index =( v.i+ position ) ), _expand_np(obj, tokens)))
    return svos  
def _fix_sub(sub):
    if sub.pos_ == "PRON" and sub.head.pos_ == "VERB" and sub.head.dep_ == "relcl":
        return sub.head.head
    else:
        return sub
        
def _get_conj_noun(noun):
    """
    Get all conj noun of a noun, including the original noun itself"""
    stacks = [noun]
    nouns = []
    while(stacks):
        _noun = stacks.pop()
        nouns.append(_noun)
        if hasattr(_noun, 'rights'):
            for t in _noun.rights:
                #handling and
                if t.dep_ == "conj" and  t.pos_ in NOUNS:
                    stacks.append(t)
                # hhandling such as or like
                # if t.dep_ == "prep" and t.lemma_ in ["as", "like", "include", "contain"] and t.pos_ in ["ADP","VERB"]:
                #     for r in t.rights:
                #         if r.pos_ in NOUNS and r.dep_ in OBJECTS:
                #             stacks.append(r)

                #handling appos
                if t.dep_ == "appos" and t.pos_ in NOUNS:
                    stacks.append(t)
                if t.dep_ == "acl" and t.pos_ == "VERB" and t.lemma_ in ["named", "called"]:
                    if hasattr(t, 'rights'):
                        for z in t.rights:
                            if z.dep_ == "oprd":
                                stacks.append(z)
    return nouns

def _get_verb_fromto(verb, tokens):
    """ A get something from someone
    => (somehintg, from, someone)
    This is the case "from" support the verb "get"
    """
    # if verb.lemma_ not in ["obtain", "extract", "send", "receive", "get", "acquire", "retrieve", "achieve", "attain", "gain", "collect", "gather","steal", "accumulate","exfiltrate", "add", "write" ]:
    #     return []
    if not hasattr(verb, 'rights'):
        return []
    svos = []

    main_negation = _is_negated(verb)
    if _is_passive_verb(verb):
        subs = [tok for tok in tokens if tok.dep_ == "nsubjpass" and tok.head == verb] # A is extracted from B , so new_sub is A
    else:
        subs = [tok for tok in tokens if tok.dep_ == "dobj" and tok.head == verb] # He extracts A from B, so new_sub is A
    _subs = []
    # we want to include the conjunct nouns too, "A and B are extracted from C"
    for s in subs:
        _subs.extend(_get_conj_noun(s))
    subs= _subs
    if len(subs) == 0:
        return []
    rights = list(verb.rights)
    # check if this verb has  to/ from prep # extract "from"/ "to"
    preps = [tok for tok in rights if tok.dep_ in ["prep","dative"] and tok.lower_ in ["to", "from"]]
    if len(preps) == 0:
        return []
    # if preps has pobj # extract "someone"
    for v in preps:
        verbs = [v]
        if not hasattr(v, 'rights'):
            continue
        _temp = [tok for tok in v.rights if tok.dep_ == "pobj" and tok.pos_ in NOUNS]
        _objs =[]
        for t in _temp:
            _objs.extend(_get_conj_noun(t))
        objs = _objs
        if len(objs) == 0:
            continue
        _svos = [subs, verbs, objs]
        products = list(itertools.product(*_svos))
        for p in products:
            sub = _fix_sub(p[0])
            v = p[1]
            verbNegated = _is_negated(v) # negation for the prep
            obj = p[2]
            objNegated = _is_negated(obj) # negation for the object
            negation = main_negation or verbNegated or objNegated
            # no reverse for this case
            svos.append(generate_svo(_expand_np(sub, tokens),
                                         expand_verb(verb=v,text = v.lemma_,negation = negation,index = v.i), _expand_np(obj, tokens)))                            


    return svos

def _get_svos_noun_from_to(objs, tokens, main_negation):
    """I get something from someone,
    => (something, from, someone)
    This is the case when "from" support directly to the Noun object instead of the verb
    """
    svos = []
    for obj in objs:
        if not hasattr(obj, 'rights'):
            continue

        sub= obj
        for r in obj.rights:

            if r.pos_ == "ADP" and r.dep_ == "prep" and r.lower_ in ["to", "from"]:
                v = r
                is_pas = False # passive not effect this case 
                verbNegated = _is_negated(v)
                for tok in r.rights:
                    if tok.dep_ == "pobj":
                        # we want to extend the conjunct nouns too, "C is extracted from A and B"
                        objs = _get_conj_noun(tok)
                        for obj in objs:
 
                            objNegated = _is_negated(obj)
                            negation = main_negation or verbNegated or objNegated
                            if is_pas:  # reverse object / subject for passive
                                svos.append(generate_svo(_expand_np(obj, tokens),
                                         expand_verb(verb=v,text = v.lemma_,negation = negation,index = v.i), _expand_np(sub, tokens)))                            
                            else:
                                svos.append(generate_svo(_expand_np(sub, tokens),
                                         expand_verb(verb=v,text = v.lemma_,negation = negation,index = v.i), _expand_np(obj, tokens))) 
    return svos

def _get_svos_noun_acl(_noun, tokens):
    svos = []
    noun = _noun
    if not hasattr(noun, 'rights'):
        return []
    for r in noun.rights:
            if r.pos_ == "VERB" and r.dep_ == "acl":
                is_pas = False
                if r.tag_ in ["VBN", "VBD"]:
                    is_pas = True
                else:
                    if r.tag_ in ["VBG", "VB"]:
                        is_pas = False
                
                _, objs =  _get_all_objs(r, False)
                verbs = [r]
                subjs = [noun]
                _svos = [subjs, verbs, objs]
                products = list(itertools.product(*_svos))
                for p in products:
                    sub = _fix_sub(p[0])
                    subNegated = _is_negated(sub)
                    v = p[1]
                    verbNegated = _is_negated(v)
                    obj = p[2]
                    objNegated = _is_negated(obj)
                    negation = verbNegated or objNegated or subNegated
                    verb_text = _expand_verb(v)
                    svos.append(generate_svo(_expand_np(sub, tokens),
                                     expand_verb(verb=v,text = verb_text,negation = negation,index = v.i), _expand_np(obj, tokens)))
                if is_pas: # if is passive and has agent
                    agent = get_pure_gent_objs(r)
                    if len(agent) > 0:
                        _svos = [agent, verbs, subjs]
                        products = list(itertools.product(*_svos))
                        for p in products:
                            sub = _fix_sub(p[0])
                            subNegated = _is_negated(sub)
                            v = p[1]
                            verbNegated = _is_negated(v)
                            obj = p[2]
                            objNegated = _is_negated(obj)
                            negation = verbNegated or objNegated or subNegated

                            svos.append(generate_svo(_expand_np(sub, tokens),
                                     expand_verb(verb=v,text = v.lemma_,negation = negation,index = v.i), _expand_np(obj, tokens)))
                    else:

                        negation = _is_negated(r) or _is_negated(noun)
                        if (noun.i +1) in np_dictionary:
                                continue
                        svos.append(generate_svo("ANY",
                                         expand_verb(verb = r,negation = negation,index = r.i), _expand_np(noun, tokens)))
    
    return svos
def _get_svo_from_noun_preposition(_noun, tokens):
    svos = []
    noun = _noun
    if noun.dep_ == "pobj" and noun.head.pos_ == "ADP":
        objs = [noun]
        verb = noun.head
        subjs = []
        if verb.dep_ == "prep" and verb.head.pos_ == "VERB":
            true_verb = verb.head
            if true_verb.dep_ == "conj":
                while true_verb.dep_ == "conj" and true_verb.head.pos_ == "VERB":
                    true_verb = true_verb.head
            if hasattr(true_verb, 'lefts'):# we only care about passive subjec here
                subjs = [tok for tok in true_verb.lefts if tok.dep_ in ["nsubjpass","nsubj"]]
        verbs = [verb]
        if len(subjs)>0:
            _svos = [subjs, verbs, objs]
            products = list(itertools.product(*_svos))
            for p in products:
                sub = p[0]
                subNegated = _is_negated(sub)
                v = p[1]
                verbNegated = _is_negated(v)
                obj = p[2]
                objNegated = _is_negated(obj)
                negation = verbNegated or objNegated or subNegated
                svos.append(generate_svo(_expand_np(sub, tokens),
                                     expand_verb(verb=v,text = v.lemma_,negation = negation,index = v.i), _expand_np(obj, tokens)))
    rights = list(noun.rights) if hasattr(noun, 'rights') else []
    for r in rights:
            if r.pos_ == "ADP" and r.dep_ == "prep":
                v,objs =  _get_all_objs(r, False)
                verbs = [r]
                subjs = [noun]
                _svos = [subjs, verbs, objs]
                products = list(itertools.product(*_svos))
                for p in products:
                    sub = _fix_sub(p[0])
                    subNegated = _is_negated(sub)
                    v = p[1]
                    verbNegated = _is_negated(v)
                    obj = p[2]
                    objNegated = _is_negated(obj)
                    negation = verbNegated or objNegated or subNegated

                    svos.append(generate_svo(_expand_np(sub, tokens),
                                     expand_verb(verb=v,text = v.lemma_,negation = negation,index = v.i), _expand_np(obj, tokens)))
    
    
    if noun.lemma_ in ["itself", "themselves"] and noun.dep_ == "dobj":
        tok = noun.head
        subs = [noun]
        if hasattr(tok, "rights"):
            for r in tok.rights:
                if r.pos_ == "ADP" and r.dep_ == "prep":
                    verbs = [r]
                    objs = [t for t in r.rights if t.dep_ in OBJECTS]
                    _objs = []
                    for o in objs:
                        _objs.extend(_get_conj_noun(o))
                    if len(_objs) == 0:
                        continue
                    _svos = [subs, verbs, _objs]
                    products = list(itertools.product(*_svos))
                    for p in products:
                        sub = p[0]
                        subNegated = _is_negated(sub)
                        v = p[1]
                        verbNegated = _is_negated(v)
                        obj = p[2]
                        objNegated = _is_negated(obj)
                        negation = verbNegated or objNegated or subNegated
                        svos.append(generate_svo(_expand_np(sub, tokens),expand_verb(verb=v,text = v.lemma_,negation= negation,index = v.i), _expand_np(obj, tokens)))
    

    if noun.dep_ == "dobj":
        subjs = [noun]
        verb = noun.head
        if hasattr(verb,"rights"):
            toks = [r for r in verb.rights if r.dep_ == "prep" and r.lemma_ not in ["as", "like", "include", "contain"]]
            for t in toks:
                verbs = [t]
                objs = [o for o in t.rights if o.dep_ in OBJECTS]
                _objs = []
                for o in objs:
                    _objs.extend(_get_conj_noun(o))
                if len(_objs) == 0:
                    continue
                _svos = [subjs, verbs, _objs]
                products = list(itertools.product(*_svos))
                for p in products:
                    sub = p[0]
                    subNegated = _is_negated(sub)
                    v = p[1]
                    verbNegated = _is_negated(v)
                    obj = p[2]
                    objNegated = _is_negated(obj)
                    negation = verbNegated or objNegated or subNegated
                    svos.append(generate_svo(_expand_np(sub, tokens),expand_verb(verb=v,text = v.lemma_,negation= negation,index = v.i), _expand_np(obj, tokens)))

    return svos



def get_SVOS_from_missing_entity(doc, missing_entity:dict=None):
    """
    after we map a entity to its represented string
    for example: "https://something.com/.../something" => "WebsiteA"
    Change the sentence to new one and find svos that related to "websiteA"
    Now we need to find the tok that tok.lower_ == "websitea"
    and find any SVOS that contain this tok
    The problem is sometime, this important entity are not included in the SVOS....
    """
    svos = []
    noun_chunks = list(doc.noun_chunks)
    text = missing_entity["text"]
    for chunk in noun_chunks:
            if text.lower() in chunk.text.lower():
                tok = chunk.root # get the main tok of the chunk
                if tok.dep_ in ["nsubj", "nsubjpass"]:
                    sub = tok
                    if not tok.head.pos_ == "VERB":
                        continue
                    verb = tok.head
                    is_passive = _is_passive_verb(verb)
                    if is_passive:
                        objs = get_agent_objs(verb)
                    else:
                        v, objs = _get_all_objs(verb,is_passive)
                    subs = [sub]
                    verbs = [verb]
                    verbNegated = _is_negated(verb)
                else:
                    if tok.dep_ in ["dobj", "pobj"]:
                        objs = [tok]
                        if not tok.head.pos_ == "VERB":
                            continue
                        verb = tok.head
                        is_passive = False
                        verbNegated, subs = get_sub_exhaustive(verb)
                        verbs = [verb]
                if len(objs) == 0 or len(subs) == 0:
                    continue
                _svos = [subs, verbs, objs]
                products = list(itertools.product(*_svos))
                for p in products:
                    sub = p[0]
                    v = p[1]                     
                    obj = p[2]
                    objNegated = _is_negated(obj)
                    if is_passive:  # reverse object / subject for passive
                            svos.append(generate_svo(_expand_np(obj, doc),
                                         expand_verb(verb=v,text = v.lemma_,negation =( verbNegated or objNegated),index = v.i), _expand_np(sub, doc)))                            
                    else:
                            svos.append(generate_svo(_expand_np(sub, doc),
                                       expand_verb(verb=v,text = v.lemma_,negation =( verbNegated or objNegated),index = v.i), _expand_np(obj, doc)))
    _svos =list()
    for s in svos:
        svo = dict()
        sub= svo[0]
        verb = svo[1]
        obj = svo[2]
        if verb.startswith("!"):
                continue
        svo["sub"] = sub
        svo["sub"]["entity"] = list()
        svo["verb"] = verb
        svo["obj"] = obj
        svo["obj"]["entity"] = list()
        e = missing_entity
        if e["start"] >= sub["start"] and e["end"] <= sub["end"]:
                    svo["sub"]["entity"].append(e)
        if e["start"] >= obj["start"] and e["end"] <= obj["end"]:
                    svo["obj"]["entity"].append(e)
        _svos.append(svo)
    return _svos
def get_agent_objs(verb):# when verb is passive
    _result = []
    if not hasattr(verb, 'rights'):
        return ["it"]
    
    for r in verb.rights:
        if r.dep_ == "agent" and r.pos_ == "ADP":
            if not hasattr(r, 'rights'):
                continue
            for tok in r.rights:
                if tok.dep_ == "pobj":
                    _result.extend(_get_conj_noun(tok))
            
    if _result == []:
        return ["it"]
    return _result
def get_pure_gent_objs(verb):# when verb is passive
    _result = []
    if not hasattr(verb, 'rights'):
        return ["it"]
    for r in verb.rights:
        if r.dep_ == "agent" and r.pos_ == "ADP":
            if not hasattr(r, 'rights'):
                continue
            for tok in r.rights:
                if tok.dep_ == "pobj":
                    _result.extend(_get_conj_noun(tok))
            
    return _result

def get_sub_exhaustive(verb):
    stacks = [verb]
    
    while len(stacks) > 0:
        verb = stacks.pop()
        if hasattr(verb, 'lefts'):   
            for l in verb.lefts:
                if l.pos_ in NOUNS and l.dep_ == ["nsubj", "nsubjpass"]:
                    return _is_negated(verb),_get_conj_noun(l)
        if verb.dep_ in ["advcl","xcomp","conj"] and verb.head.pos_ == "VERB":
            stacks.append(verb.head)

    return False, []



def get_SVO_from_x_comp(input_verb, doc):
    if input_verb.lemma_ not in command_verbs:
        return []
    main_negation = _is_negated(input_verb) #I do not ask him to do it
    svos = []
    is_pas = _is_passive_verb(input_verb)
    main_subs = []
    main_objs = []
    if not hasattr(input_verb, 'rights'):
        return []
    if hasattr(input_verb, 'lefts'):
        for l in input_verb.lefts:
            if l.dep_ in ["nsubj", "nsubjpass"] and l.pos_ in NOUNS:
                main_subs.extend(_get_conj_noun(l))
    if hasattr(input_verb, 'rights'):
        _, main_objs = _get_all_objs(input_verb,main_negation)
    xcomp_verb = [tok for tok in input_verb.rights if tok.dep_ in ["xcomp","ccomp","advcl"] and tok.pos_ == "VERB"]
    if len(xcomp_verb) == 0:
        for r in input_verb.rights:
            if r.dep_ == "prep" and r.pos_ == "ADP":
                xcomp_verb = [tok for tok in r.rights if tok.dep_ in ["pcomp"] and tok.pos_ == "VERB"]
    for xverb in xcomp_verb:
        xobjs =[]
        xsubs = []
        xcomp_negation = _is_negated(xverb)
        if hasattr(xverb, 'rights'):
            _,xobjs  = _get_all_objs(xverb,xcomp_negation)
        # "The reference of the subject is necessarily determined by an argument external to the xcomp "
    #    "(normally by the object of the next higher clause, if there is one, or else by the subject of the next higher clause)."

        if len(main_objs) == 0:
            xsubs = main_subs
        else:
            xsubs = main_objs
        verbs = [xverb]
        if len(xsubs) == 0 or len(xobjs) == 0:
            continue


        _svos = [xsubs, verbs, xobjs]
         # I ask him not to do it and I ask him to not do it
        negation = main_negation or xcomp_negation
        products = list(itertools.product(*_svos))
        for p in products:
            sub = p[0]
            v = p[1]                     
            obj = p[2]
            objNegated = _is_negated(obj)
            svos.append(generate_svo(_expand_np(sub, doc),
                                       expand_verb(verb=v,text = v.lemma_,negation =( negation or objNegated),index = v.i), _expand_np(obj, doc)))       
    
    
    return svos

def as_handling(tokens):
    svos = []
    toks = [tok for tok in tokens if tok.dep_ == "prep" and tok.lemma_ == "as"]
    for t in toks:
        objs = []
        _objs = []
        if hasattr(t, 'lefts'):
            for l in t.lefts:
                if l.lemma_ in ["such"]:
                    continue
        if hasattr(t, 'rights'):
            objs = [tok for tok in t.rights if tok.dep_ in OBJECTS]
            if len(objs) >0:
                objs = _get_conj_noun(objs[0])
        if t.head.pos_ == "VERB":
            verb = t.head
            is_pas = _is_passive_verb(verb)    
            if not is_pas:
                if hasattr(verb, 'rights'):
                    _objs = [tok for tok in verb.rights if tok.dep_ in OBJECTS]
            else:
                if hasattr(verb, 'lefts'):
                    _objs = [tok for tok in verb.lefts if tok.dep_ == "nsubjpass"]
            if len(_objs)== 0 and verb.dep_ == "acl" and verb.head.pos_ in NOUNS:
                _objs = [verb.head]
        if len(objs) > 0 and len(_objs) > 0:
            subjs = _objs
            verbs = [t]
            _svos = [subjs, verbs, objs]
            products = list(itertools.product(*_svos))
            for p in products:
                sub = p[0]
                v = p[1]                     
                obj = p[2]
                svos.append(generate_svo(_expand_np(sub, tokens),
                                     expand_verb(verb=v,text = v.lemma_,negation = False,index = v.i), _expand_np(obj, tokens)))
    
    return svos
def get_example_subs(v):
    """handling passive sentence with example"""
    assert _is_passive_verb(v) == True
    subjs = []
    if not hasattr(v, 'rights'):
        return []
    rights = list(v.rights)
    for tok in rights:
            if tok.dep_ == 'prep' and tok.lemma_ in ["like", "as","include", "contain"]: # like, such as, including,containing
                if hasattr(tok, 'rights'):
                    subjs.extend([t for t in tok.rights if t.dep_ in OBJECTS])
    
    _subjs = []
    for s in subjs:
        _subjs.extend(_get_conj_noun(s))
    _subjs = list(set(_subjs))
    return _subjs



def get_examples_cases(tokens):
    svos = []
    examples = [tok for tok in tokens if (tok.lemma_ in ["like", "as","include", "contain"] and tok.dep_ == "prep")]
    nouns =list()
    for e in examples:
        if hasattr(e, 'rights'):
            for tok in e.rights:
                if tok.dep_ in OBJECTS:
                    nouns.append(tok)
    _nouns = []
    for n in nouns:
        _nouns.extend(_get_conj_noun(n))
    
    return_nouns = []
    for n in _nouns:
        return_nouns.append(_expand_np(n, tokens))
    

    return return_nouns

def _get_svos_from_possessive_noun(_noun, tokens):
    svos = []
    noun = _noun
    if noun.dep_ == "poss" and noun.head.pos_ in NOUNS:
        subjs = _get_conj_noun(noun.head)
        objs = _get_conj_noun(noun)

        _svos = [subjs, objs]
        products = list(itertools.product(*_svos))
        for p in products:
            sub = _fix_sub(p[0])
            subNegated = _is_negated(sub)
            verbNegated = False
            obj = p[1]
            objNegated = _is_negated(obj)
            negation = verbNegated or objNegated or subNegated
            _expanded_sub = _expand_np(sub, tokens)
            _expanded_obj = _expand_np(obj, tokens, flag = False)
            if _expanded_sub["text"] == _expanded_obj["text"]:
                _expanded_sub = _expand_np(sub, tokens, flag = False)
                _expanded_obj = _expand_np(obj, tokens, flag = False)
            if _expanded_obj["text"].lower() in ["its", "his", "her", "their", "our", "my", "your"]:
                continue
            svos.append(generate_svo(_expanded_sub,
                                     expand_verb(text = "of",negation= negation,index = sub.i), _expanded_obj))
    


    return svos

def get_acl_and_preposition(toks):
    nouns = [tok for tok in toks if tok.pos_ in  NOUNS]
    svos = []
    for n in nouns:
        svos.extend(_get_svos_noun_acl(n, toks))
        svos.extend(_get_svo_from_noun_preposition(n, toks))
        svos.extend(_get_svos_from_possessive_noun(n, toks))
    return svos


def get_conjuncted_np(nps):
    chains = []
    for np in nps:
        index = np.root.i + 1
        if np.root.dep_ == "conj":
            source = np.root.head.i + 1
            chains.append((source, index))
    chains = sorted(chains, key=lambda x: x[1], reverse=True)
    return chains

def _haveget_something_done(verb,tokens):
    svos = []
    if verb.tag_ == "VBN" and verb.dep_  == "ccomp" and verb.head.pos_ == "VERB" and verb.head.lemma_ in ["have", "get"]:
        objs = []
        if hasattr(verb, 'lefts'):
            objs = [tok for tok in verb.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
        verbs = [verb]
        if len(objs) == 0:
            return []
        combinations = list(itertools.product(*[verbs,objs]))
        for c in combinations:
            v = c[0]
            obj = c[1]
            svos.append(generate_svo("ANY",expand_verb(verb =v,negation =False,index = v.i ), _expand_np(obj, tokens)))
    return svos