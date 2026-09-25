"""Candidate generation and blocking module for Business Entity Resolution."""

import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Set, Tuple

STOP_WORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'the', 'and', 'of', 'in', 'at',
    'group', 'services', 'service', 'solutions', 'solution', 'enterprises',
    'enterprise', 'center', 'centre', 'associates', 'holdings', 'holding',
    'consulting', 'management', 'international', 'systems', 'tech', 'technologies',
    'sarl', 'sasu', 'eurl', 'gmbh', 'sa', 'sas', 'sci', 'ste', 'societe',
    'pvtltd', 'india', 'us', 'usa', 'france', 'llp', 'pc', 'foundation'
}

ADDR_STOP_WORDS = {
    'near', 'opp', 'opposite', 'road', 'rd', 'street', 'st', 'lane', 'ln',
    'drive', 'dr', 'avenue', 'ave', 'boulevard', 'blvd', 'floor', 'fl',
    'unit', 'suite', 'ste', 'building', 'bldg', 'block', 'blk', 'house',
    'flat', 'plot', 'no', 'hno', 'sector', 'sec', 'nagar', 'colony',
    'california', 'texas', 'florida', 'new', 'york', 'null', 'nan'
}

def normalize_text(text: str) -> str:
    """Normalize text: strip accents, web domains, leading symbols, non-alphanumeric."""
    if not text or text == 'nan':
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'https?://|www\.', '', text)
    text = re.sub(r'\.(com|in|org|net|co|fr)\b', '', text)
    text = re.sub(r'^[<\-#@\s]+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return " ".join(text.split())

def extract_tokens(text: str, stop: set) -> List[str]:
    return [t for t in text.split() if t not in stop and len(t) > 1]

def extract_numbers(tokens: List[str]) -> Set[str]:
    nums = set()
    for t in tokens:
        if t.isdigit() and len(t) >= 2:
            nums.add(str(int(t)))
    return nums

def extract_name_keys(clean_name: str) -> List[Tuple]:
    """Generate multi-level name blocking keys."""
    tokens = [t for t in clean_name.split() if t not in STOP_WORDS and len(t) > 1]
    keys = []
    for t in tokens[:4]:
        keys.append(('tok', t))
    compact = "".join(tokens[:3])
    if len(compact) >= 5:
        keys.append(('compact', compact))
    clean_all = clean_name.replace(" ", "")
    if len(clean_all) >= 4:
        keys.append(('pref4', clean_all[:4]))
    return keys

def extract_address_keys(clean_addr: str) -> List[Tuple]:
    """Generate multi-level address blocking keys."""
    tokens = clean_addr.split()
    keys = []
    
    for i, tok in enumerate(tokens):
        if tok.isdigit() and len(tok) >= 2:
            norm_num = str(int(tok))
            adj_words = []
            if i + 1 < len(tokens) and not tokens[i+1].isdigit() and len(tokens[i+1]) > 2:
                adj_words.append(tokens[i+1])
            if i - 1 >= 0 and not tokens[i-1].isdigit() and len(tokens[i-1]) > 2:
                adj_words.append(tokens[i-1])
            for w in adj_words:
                if w not in ADDR_STOP_WORDS:
                    keys.append(('num_word', norm_num, w))
            if len(norm_num) >= 5:
                keys.append(('pin', norm_num))
                
    rare_words = [t for t in tokens if not t.isdigit() and len(t) >= 4 and t not in ADDR_STOP_WORDS]
    for w in rare_words[:3]:
        keys.append(('addr_tok', w))
        
    return keys

class FastBlocker:
    """Country-partitioned inverted index blocker."""
    
    def __init__(self):
        self.index = defaultdict(list)
        
    def index_candidates(self, candidates: Dict[str, dict]):
        for cid, cand in candidates.items():
            c = cand['country']
            for k in extract_name_keys(cand['clean_name']):
                self.index[(c, k)].append(cid)
            for k in extract_address_keys(cand['clean_addr']):
                self.index[(c, k)].append(cid)
                
    def get_candidates(self, s1_rec: dict, max_name_bucket: int = 150, max_addr_bucket: int = 100) -> Set[str]:
        c = s1_rec['country']
        cand_set = set()
        
        for k in extract_name_keys(s1_rec['clean_name']):
            b = self.index.get((c, k), [])
            if len(b) <= max_name_bucket:
                cand_set.update(b)
                
        for k in extract_address_keys(s1_rec['clean_addr']):
            b = self.index.get((c, k), [])
            max_b = 50 if k[0] == 'addr_tok' else max_addr_bucket
            if len(b) <= max_b:
                cand_set.update(b)
                
        return cand_set
