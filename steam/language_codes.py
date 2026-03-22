"""
ISO 639-1 <-> ISO 639-3 language code conversion.

Bridges translation APIs (ISO 639-1) with URIEL database (ISO 639-3).
"""

ISO_639_1_TO_3 = {
    # High-resource
    'en': 'eng', 'fr': 'fra', 'de': 'deu', 'it': 'ita',
    'es': 'spa', 'pt': 'por',
    # Medium-resource
    'pl': 'pol', 'nl': 'nld', 'ru': 'rus', 'hi': 'hin',
    'ko': 'kor', 'ja': 'jpn', 'zh-CN': 'zho', 'zh': 'zho',
    'ar': 'arb',
    # Low-resource
    'bn': 'ben', 'fa': 'pes', 'vi': 'vie', 'he': 'heb',
    'iw': 'heb', 'uk': 'ukr', 'ta': 'tam', 'th': 'tha',
    'tr': 'tur', 'id': 'ind', 'ms': 'msa', 'sw': 'swh',
    'ro': 'ron', 'cs': 'ces', 'sv': 'swe', 'da': 'dan',
    'no': 'nor', 'fi': 'fin', 'el': 'ell', 'hu': 'hun',
    'sk': 'slk', 'bg': 'bul', 'hr': 'hrv', 'sr': 'srp',
    'sl': 'slv', 'lt': 'lit', 'lv': 'lav', 'et': 'est',
    'ca': 'cat', 'eu': 'eus', 'gl': 'glg', 'af': 'afr',
    'is': 'isl', 'sq': 'sqi', 'ka': 'kat', 'hy': 'hye',
    'az': 'azj', 'uz': 'uzn', 'kk': 'kaz', 'mn': 'khk',
    'ur': 'urd', 'ne': 'nep', 'si': 'sin', 'my': 'mya',
    'km': 'khm', 'lo': 'lao', 'am': 'amh', 'ti': 'tir',
    'yo': 'yor', 'ig': 'ibo', 'zu': 'zul', 'xh': 'xho',
    'st': 'sot', 'sn': 'sna', 'ha': 'hau', 'mg': 'mlg',
    'so': 'som', 'te': 'tel', 'kn': 'kan', 'ml': 'mal',
    'mr': 'mar', 'gu': 'guj', 'pa': 'pan', 'or': 'ori',
    'as': 'asm', 'tg': 'tgk', 'be': 'bel', 'cy': 'cym',
    'gd': 'gla', 'ga': 'gle', 'co': 'cos', 'mt': 'mlt',
}

ISO_639_3_TO_1 = {v: k for k, v in ISO_639_1_TO_3.items()}


def iso1_to_iso3(code):
    code = code.lower()
    if code not in ISO_639_1_TO_3:
        raise ValueError(f"Unknown ISO 639-1 code: {code}")
    return ISO_639_1_TO_3[code]


def iso3_to_iso1(code):
    code = code.lower()
    if code not in ISO_639_3_TO_1:
        raise ValueError(f"Unknown or no ISO 639-1 equivalent for: {code}")
    return ISO_639_3_TO_1[code]


def is_valid_iso3(code):
    return code.lower() in ISO_639_3_TO_1
