import re

# Canonical mapping: raw name (any casing) → canonical name.
# Lookup is case-insensitive exact match first, then title-stripped exact match.
SPEAKER_MAP = {
    # ── SP Daniel Foo ─────────────────────────────────────────────────────────
    "Pastor Daniel Foo":            "SP Daniel Foo",
    "Ps Daniel Foo":                "SP Daniel Foo",
    "Senior Pastor Daniel Foo":     "SP Daniel Foo",
    "SP Dan Foo":                   "SP Daniel Foo",
    "SP Dani":                      "SP Daniel Foo",
    "Daniel Foo":                   "SP Daniel Foo",
    "Dan Foo":                      "SP Daniel Foo",
    "SP Daniel":                    "SP Daniel Foo",

    # ── SP Chua Seng Lee ──────────────────────────────────────────────────────
    "Senior Pastor Chua Seng Lee":  "SP Chua Seng Lee",
    "DSP Chua Seng Lee":            "SP Chua Seng Lee",
    "Elder DSP Chua Seng Lee":      "SP Chua Seng Lee",
    "SP Designate Chua Seng Lee":   "SP Chua Seng Lee",
    "Elder Chua Seng Lee":          "SP Chua Seng Lee",
    "SP Seng Lee":                  "SP Chua Seng Lee",
    "Ps Chua Seng Lee":             "SP Chua Seng Lee",
    "Chua Seng Lee":                "SP Chua Seng Lee",
    "Chua Seng":                    "SP Chua Seng Lee",
    "Chua":                         "SP Chua Seng Lee",
    "Csl":                          "SP Chua Seng Lee",
    "pcsl":                         "SP Chua Seng Lee",
    "Spd Chua Seng Lee":            "SP Chua Seng Lee",
    "Dps Chua Seng Lee":            "SP Chua Seng Lee",
    "Chua Seng Lee Member":         "SP Chua Seng Lee",
    "Faith By Chua Seng Lee":       "SP Chua Seng Lee",

    # ── Ps Edric Sng ──────────────────────────────────────────────────────────
    "DSP Edric Sng":                "Ps Edric Sng",
    "Elder Edric Sng":              "Ps Edric Sng",
    "Elder DSP Edric Sng":          "Ps Edric Sng",
    "DSP Elder Edric Sng":          "Ps Edric Sng",
    "Pastor Edric Sng":             "Ps Edric Sng",
    "Edric Sng":                    "Ps Edric Sng",
    "Edric":                        "Ps Edric Sng",
    "Edric Sng Part One":           "Ps Edric Sng",
    "Fire By Edric":                "Ps Edric Sng",

    # ── Ps Low Kok Guan ───────────────────────────────────────────────────────
    "DSP Low Kok Guan":             "Ps Low Kok Guan",
    "Pastor Lok Kok Guan":          "Ps Low Kok Guan",
    "Elder Low Kok Guan":           "Ps Low Kok Guan",
    "Pastor Low Kok Guan":          "Ps Low Kok Guan",
    "Low Kok Guan":                 "Ps Low Kok Guan",
    "Kok Guan":                     "Ps Low Kok Guan",

    # ── Elder Lok Vi Ming ─────────────────────────────────────────────────────
    "Elder Lok Vi Meng":            "Elder Lok Vi Ming",
    "Lok Vi Ming":                  "Elder Lok Vi Ming",
    "Eld Lok Vi Ming":              "Elder Lok Vi Ming",
    "Elvm":                         "Elder Lok Vi Ming",
    "Lvm Member":                   "Elder Lok Vi Ming",

    # ── Elder Ho Tuck Leh ─────────────────────────────────────────────────────
    "Ho Tuck Leh":                  "Elder Ho Tuck Leh",

    # ── Elder Goh Hock Chye ───────────────────────────────────────────────────
    "Goh Hock Chye":                "Elder Goh Hock Chye",
    "E Goh Hock Chye":              "Elder Goh Hock Chye",
    "Hock Chye":                    "Elder Goh Hock Chye",
    "By Eghc":                      "Elder Goh Hock Chye",
    "Faith By Goh Hock Chye":       "Elder Goh Hock Chye",

    # ── Elder Leon Oei ────────────────────────────────────────────────────────
    "Leon Oei":                     "Elder Leon Oei",

    # ── Elder David Foo ───────────────────────────────────────────────────────
    "CS David Foo":                 "Elder David Foo",
    "David Foo":                    "Elder David Foo",
    "Cell Supervisor, David Foo":   "Elder David Foo",
    "A Perfect God Study Of Jothams Life By Cs David Foo": "Elder David Foo",

    # ── Elder Mark Tan ────────────────────────────────────────────────────────
    "CS Mark Tan":                  "Elder Mark Tan",
    "Mark Tan":                     "Elder Mark Tan",
    "Mark T":                       "Elder Mark Tan",
    "Emark Tan":                    "Elder Mark Tan",

    # ── Ps Lawrence Chua ──────────────────────────────────────────────────────
    "Lawrence Chua":                "Ps Lawrence Chua",
    "Ps Lawrence":                  "Ps Lawrence Chua",
    "Senior Pastor Lawrence Chua":  "Ps Lawrence Chua",
    "SP Lawrence Chua":             "Ps Lawrence Chua",
    "SP LAWRENCE CHUA":             "Ps Lawrence Chua",
    "Lawrence":                     "Ps Lawrence Chua",

    # ── Ps Gary Koh ───────────────────────────────────────────────────────────
    "Pastor Gary Koh":              "Ps Gary Koh",
    "Gary Koh":                     "Ps Gary Koh",
    "Brother Gary Koh":             "Ps Gary Koh",
    "Mr Gary Koh":                  "Ps Gary Koh",
    "Gary & Joanna Koh":            "Ps Gary Koh",
    "Gary And Joanna Koh":          "Ps Gary Koh",

    # ── Ps Jeffrey Aw ─────────────────────────────────────────────────────────
    "Pastor Jeffrey Aw":            "Ps Jeffrey Aw",
    "Jeffrey Aw":                   "Ps Jeffrey Aw",
    "Jeff Aw":                      "Ps Jeffrey Aw",

    # ── Ps Andrew Tan ─────────────────────────────────────────────────────────
    "Pastor Andrew Tan":            "Ps Andrew Tan",
    "Andrew Tan":                   "Ps Andrew Tan",
    "Pastor Andrew Tan & Ps Sarah Khong": "Ps Andrew Tan",
    "Andrew Tan  Sarah Khong":      "Ps Andrew Tan",
    "Andrew Tan Lmembers Guide":    "Ps Andrew Tan",

    # ── Ps Darren Kuek ────────────────────────────────────────────────────────
    "Pastor Darren Kuek":           "Ps Darren Kuek",
    "Ps Darren":                    "Ps Darren Kuek",
    "Darren Kuek":                  "Ps Darren Kuek",
    "Darren":                       "Ps Darren Kuek",
    "Darren Kuek Echua Seng Lee":   "Ps Darren Kuek",

    # ── Ps Don Wong ───────────────────────────────────────────────────────────
    "Pastor Don Wong":              "Ps Don Wong",
    "Don Wong":                     "Ps Don Wong",

    # ── Ps Jason Teo ──────────────────────────────────────────────────────────
    "Pastor Jason Teo":             "Ps Jason Teo",
    "Jason Teo":                    "Ps Jason Teo",
    "Pastor Jaso":                  "Ps Jason Teo",

    # ── Ps Ng Hua Ken ─────────────────────────────────────────────────────────
    "Pastor Ng Hua Ken":            "Ps Ng Hua Ken",
    "Hua Ken":                      "Ps Ng Hua Ken",
    "Elder Ps Ng Hua Ken":          "Ps Ng Hua Ken",
    "Ng Hua Ken":                   "Ps Ng Hua Ken",
    "Ng H":                         "Ps Ng Hua Ken",
    "Ng Hua Ken Member":            "Ps Ng Hua Ken",
    "Ng Hua Ken Members Notes":     "Ps Ng Hua Ken",

    # ── Ps Paul Jeyachandran ──────────────────────────────────────────────────
    "Rev. Paul Jeyachandran":       "Ps Paul Jeyachandran",
    "Paul Jeyachandran":            "Ps Paul Jeyachandran",

    # ── Ps Eugene Seow ────────────────────────────────────────────────────────
    "Pastor Eugene Seow":           "Ps Eugene Seow",
    "Eugene Seow":                  "Ps Eugene Seow",

    # ── Ps Nicky Raiborde ─────────────────────────────────────────────────────
    "Nicky Raiborde":               "Ps Nicky Raiborde",
    "Ps Nicky S. Raiborde":         "Ps Nicky Raiborde",
    "Ps Nicky S Raiborde":          "Ps Nicky Raiborde",
    "Nicky S. Raiborde":            "Ps Nicky Raiborde",
    "Nicky S Raiborde":             "Ps Nicky Raiborde",
    "nicky s raiborde":             "Ps Nicky Raiborde",
    "Nicky":                        "Ps Nicky Raiborde",

    # ── Jeffrey Goh ───────────────────────────────────────────────────────────
    "Brother Jeffrey Goh":          "Jeffrey Goh",

    # ── Joseph Chean ──────────────────────────────────────────────────────────
    "Brother Joseph Chean":         "Joseph Chean",

    # ── Gurmit Singh ──────────────────────────────────────────────────────────
    "gurmit Singh":                 "Gurmit Singh",

    # ── Guest Speakers ────────────────────────────────────────────────────────
    "Watson":                       "Guest Speaker",
    "Ps Watson":                    "Guest Speaker",
    "Joey Bonifacio":               "Guest Speaker",
    "Joey Bonafacio":               "Guest Speaker",
    "Samuel Phun":                  "Guest Speaker",
    "Ps Samuel Phun":               "Guest Speaker",
    "Ps Joey Bonifacio":            "Guest Speaker",
    "Ernest Chow":                  "Guest Speaker",
    "Ps Ernest Chow":               "Guest Speaker",
    "Pastor Ernest Chow":           "Guest Speaker",
    "Erne":                         "Guest Speaker",
    "Ps Erne":                      "Guest Speaker",
    "Benny Ho":                     "Guest Speaker",
    "Pastor Benny Ho":              "Guest Speaker",
    "Ps Benny Ho":                  "Guest Speaker",
    "Daniel Koh":                   "Guest Speaker",
    "Ps Daniel Koh":                "Guest Speaker",
    "Lee Kuan Yew":                 "Guest Speaker",
    "Mr Lee Kuan Yew":              "Guest Speaker",
    "Josh McDowell":                "Guest Speaker",
    "Josh Mcdowell":                "Guest Speaker",
    "Josh D. & Dottie McDowell":    "Guest Speaker",
    "Prof Freddy Boey":             "Guest Speaker",
    "Rev Rick Seaward":             "Guest Speaker",
    "Reverend Rick Seaward":        "Guest Speaker",
    "Rev Rick Seward":              "Guest Speaker",
    "Ps Bill Wilson":               "Guest Speaker",
    "Pastor Craig Hill":            "Guest Speaker",
    "Dr Bill Bright":               "Guest Speaker",
    "Dr Victor Wong":               "Guest Speaker",
    "Dr Ng Liang Wei":              "Guest Speaker",
    "Ps William Wood":              "Guest Speaker",
    "Rev. Dr. Philip Huan":         "Guest Speaker",
    "REV. DR. PHILIP HUAN":         "Guest Speaker",
    "Ps Philip Huan":               "Guest Speaker",
    "Dr. Philip Huan":              "Guest Speaker",
    "Rev. Dr. Philip Huan, Rev. Jenni Ho-Huan": "Guest Speaker",
    "Rev David Ravenhill":          "Guest Speaker",
    "Ps David Ravenhill":           "Guest Speaker",
    "David Ravenhill":              "Guest Speaker",
    "Leonard Ravenhill":            "Guest Speaker",
    "Ravenhill":                    "Guest Speaker",
    "Rev Les Wheeldon":             "Guest Speaker",
    "Rev Daniel Wee":               "Guest Speaker",
    "Ps Daniel Wee":                "Guest Speaker",
    "Floyd McClung":                "Guest Speaker",
    "Dr Chester Kylstra":           "Guest Speaker",
    "Chester Kylstra":              "Guest Speaker",
    "Chester and Betsy Kylstra":    "Guest Speaker",
    "Dr Dan Brewster":              "Guest Speaker",
    "Dan Brewster":                 "Guest Speaker",
    "Dr Cassie Carstens":           "Guest Speaker",
    "Cassie Carstens":              "Guest Speaker",
    "Cassie Carsten":               "Guest Speaker",
    "Dr Ian J":                     "Guest Speaker",
    "Dr Ian Jagelman":              "Guest Speaker",
    "Ps Michael Ross Watson":       "Guest Speaker",
    "Michael Ross Watson":          "Guest Speaker",
    "Dr Chris Cheech":              "Guest Speaker",
    "Ps Jerry Chia":                "Guest Speaker",
    "Ps Jeff Chong":                "Guest Speaker",
    "Ps Henson Lim":                "Guest Speaker",
    "Pastor Henson Lim":            "Guest Speaker",
    "Henson Lim":                   "Guest Speaker",
    "Ps Hakan Gabrielsson":         "Guest Speaker",
    "Brother Hakan Gabrielsson":    "Guest Speaker",
    "GEORGE BARNA":                 "Guest Speaker",
    "MICHAEL NOVAK":                "Guest Speaker",
    "MARY MA":                      "Guest Speaker",
    "Billy Graham":                 "Guest Speaker",
    "David Pawson":                 "Guest Speaker",
    "David Pawson, Billy Graham, Ravi, Apostle Paul": "Guest Speaker",
    "Martin Luther":                "Guest Speaker",
    "AW Tozer":                     "Guest Speaker",
    "A.W. Tozer":                   "Guest Speaker",
    "Mr. Vuong Dinh Hue":           "Guest Speaker",
    "Pr Dr Chew Weng Chee":         "Guest Speaker",
    "Chew Weng Chee":               "Guest Speaker",
    "David Wong Kim":               "Guest Speaker",
    "William Loke":                 "Guest Speaker",
    "Eugene Shi":                   "Guest Speaker",
    "Blessing Campa":               "Guest Speaker",
    "Dwight L Lord":                "Guest Speaker",
    "Marie Tsuruda And Pierre Oosthuizen": "Guest Speaker",

    # ── Dr John Andrews ───────────────────────────────────────────────────────
    "DR JOHN ANDREWS":              "Dr John Andrews",
    "D R J O H N A N D R E W S":   "Dr John Andrews",

    # ── Multi-speaker / panel rows ────────────────────────────────────────────
    "Ps Low Kok Guan, Elder Lok Vi Ming, Ps Edric Sng, Elder Goh Hock Chye": "Unknown",
    "Ps Low Kok Guan, Elder Lok Vi Ming, DSP Edric Sng": "Unknown",

    # ── Unknown / unresolvable ────────────────────────────────────────────────
    "BBTC":                         "Unknown",
    "null":                         "Unknown",
    "Pastor [Name] (assuming Pastor is the speaker, actual name not found)": "Unknown",
    "Pastor":                       "Unknown",
    "Rev":                          "Unknown",
    "Senior Pastor":                "Unknown",
    "Senior Minister":              "Unknown",
    "Pastoral Word":                "Unknown",
    "Pastor's name not mentioned":  "Unknown",
    "Pastor T3":                    "Unknown",
    "PASTOR FRIEND":                "Unknown",
    "Elder y":                      "Unknown",
    "Elder G":                      "Unknown",
    "Elder Lo":                     "Unknown",
    "Elde":                         "Unknown",
    "KK":                           "Unknown",
    "Q.":                           "Unknown",
    "me":                           "Unknown",
    "Lee":                          "Unknown",
    "Mr. Lee":                      "Unknown",
    "Jerry":                        "Unknown",
    "James":                        "Unknown",
    "Jacob":                        "Unknown",
    "Paul":                         "Unknown",
    "JOSHUA":                       "Unknown",
    "Thomas Jefferson":             "Unknown",
    "This is Jesus":                "Unknown",
    "T T Jo S s th m T cr d W th re s c T ti M tr w p W w s se": "Unknown",
    "Simon Peter":                  "Unknown",
    "Simeon Peter":                 "Unknown",
    "Page The":                     "Unknown",
    "Pastor Abraham Jacob Joseph Joshua Moses": "Unknown",
    "Jephthah Fugitive | Fighter | Father": "Unknown",
    "KING ASA":                     "Unknown",
    "Commission F":                 "Unknown",
    "F_______-L________":           "Unknown",
    "Apostle Paul":                 "Unknown",
    "DR LUKE":                      "Unknown",
    "Abraham A Life and Legacy":    "Unknown",
    "Christian":                    "Unknown",
    "Video Wilson Foo":             "Unknown",
    "Rev Wil":                      "Unknown",
    "Wil":                          "Unknown",
    "Mr Jeffr":                     "Unknown",
    "Jeffr":                        "Unknown",
    "Hezekiah":                     "Unknown",
    "Ehud":                         "Unknown",
    "Lo":                           "Unknown",
    "Dan":                          "Unknown",
    "Jason":                        "Unknown",
    "Davi":                         "Unknown",
    "Ate":                          "Unknown",
    "Ds Ate":                       "Unknown",
    "Kols":                         "Unknown",
    "3 & 4 D":                      "Unknown",
    "Philippians":                  "Unknown",
    "Name":                         "Unknown",
    "Memberguide":                  "Unknown",
    "Members Copy":                 "Unknown",
    "Dps":                          "Unknown",
    "Dsp":                          "Unknown",
}

# Bible books and other non-person strings the LLM sometimes extracts as speaker
_GARBAGE_PATTERNS = [
    r'^\d',                          # starts with digit
    r'^[A-Z]_+',                     # single letter + underscores
    r'^(Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|'
    r'Samuel|Kings|Chronicles|Ezra|Nehemiah|Esther|Job|Psalms?|Proverbs|'
    r'Ecclesiastes|Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|'
    r'Amos|Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|'
    r'Malachi|Matthew|Mark|Luke|John|Acts|Romans|Corinthians|Galatians|'
    r'Ephesians|Philippians|Colossians|Thessalonians|Timothy|Titus|Philemon|'
    r'Hebrews|James|Peter|Jude|Revelation)$',
    r'^\b(Date|Topic|Theme|Introduction|By|Page|Verse|Name|Memberguide|Members Copy|Dps|Dsp)\b',
]

_TITLE_RE = re.compile(
    r'\b(SP|DSP|Ps|Pastor|Pasto|Elder|Dr\.?|Rev\.?|Reverend|Brother|Mr\.?|Ms\.?|Mrs\.?|Ds)\b',
    flags=re.IGNORECASE,
)


def normalize_speaker(name: str) -> str:
    if not name:
        return None

    name = name.strip().replace('\xa0', ' ')
    name = re.sub(r'\s+', ' ', name)  # collapse internal multiple spaces

    if name.lower() in ('none', 'null', 'unknown', '', 'n/a'):
        return None

    # 1. Exact case-insensitive lookup
    name_lower = name.lower()
    for raw, canonical in SPEAKER_MAP.items():
        if raw.lower() == name_lower:
            return canonical

    # 2. Strip titles and retry exact lookup
    stripped = _TITLE_RE.sub('', name).strip()
    if stripped and stripped.lower() != name_lower:
        stripped_lower = stripped.lower()
        for raw, canonical in SPEAKER_MAP.items():
            raw_stripped = _TITLE_RE.sub('', raw).strip().lower()
            if raw_stripped == stripped_lower:
                return canonical

    # 3. Reject garbage: too short, starts with digit, or is a Bible book
    target = stripped or name
    if len(target) < 3:
        return None
    for pattern in _GARBAGE_PATTERNS:
        if re.search(pattern, target, flags=re.IGNORECASE):
            return None

    # 4. Return cleaned, title-cased name (unknown but well-formed)
    return target.title()


# Canonical names — the authoritative set of known speakers.
CANONICAL_SPEAKERS: frozenset[str] = frozenset(SPEAKER_MAP.values()) - {None}


def normalize_speaker_strict(name: str) -> str | None:
    """
    Like normalize_speaker but returns None for names not found in SPEAKER_MAP.
    Use when you need a confirmed canonical match, not a best-guess title-case.
    Also handles OCR-doubled characters (e.g. 'Daaniel Foo' → 'Daniel Foo') by
    trying each doubled-char position individually so unrelated pairs (like 'oo'
    in 'Foo') are not incorrectly collapsed.
    """
    # Direct attempt
    result = normalize_speaker(name)
    if result in CANONICAL_SPEAKERS:
        return result

    # Find positions of consecutive identical alpha chars (OCR artifact candidates)
    candidates = [i for i in range(len(name) - 1) if name[i] == name[i + 1] and name[i].isalpha()]
    if not candidates:
        return None

    # Try collapsing each doubled position in isolation
    for pos in candidates:
        deduped = name[:pos] + name[pos + 1:]
        result = normalize_speaker(deduped)
        if result in CANONICAL_SPEAKERS:
            return result

    # Try collapsing all doubled positions at once (handles 'Loow Kok Guaan' → 'Low Kok Guan')
    deduped = name
    for pos in sorted(candidates, reverse=True):
        deduped = deduped[:pos] + deduped[pos + 1:]
    if deduped != name:
        result = normalize_speaker(deduped)
        if result in CANONICAL_SPEAKERS:
            return result

    return None
