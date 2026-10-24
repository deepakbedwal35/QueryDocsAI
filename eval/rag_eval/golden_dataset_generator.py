# Golden dataset (20 questions) for the QueryDocsAI RAG eval.
#
# Each entry: question + reference answer + the corpus PDF the answer lives in.
# not_found=True entries are OUT of corpus - the correct behaviour is refusal.
#
# Difficulty:
#   easy     : single definitional fact, one chunk is enough
#   hard     : multi-part answer / contrast between theories / specific detail
#   negative : not in the corpus, model must say it cannot find the answer
#
# Seeded from eval/goldens/retriever_golden_dataset.json.

GOLDEN = [
 # ================= EASY =================
 dict(id=1, diff="easy", source="17data", not_found=False,
   q="What is the neuroscientific term for the non-conscious state of advanced meditators characterized by voluntary suspension of consciousness?",
   answer="Extended Cessation (EC)."),

 dict(id=2, diff="easy", source="53data", not_found=False,
   q="What is the primary hub of consciousness in the unified 3D default space model proposed by Jerath et al.?",
   answer="The thalamus."),

 dict(id=3, diff="easy", source="25data", not_found=False,
   q="What anatomical boundary is used to segment the brain into anterior and posterior regions for functional connectivity analysis?",
   answer="The central sulcus."),

 dict(id=4, diff="easy", source="13data", not_found=False,
   q="Global Workspace Theories argue that conscious experience is inextricably linked with which cognitive notion?",
   answer="The cognitive notion of 'access.' Information enters conscious awareness only when it is accessed and made available to various processing subsystems."),

 dict(id=5, diff="easy", source="1data", not_found=False,
   q="What determines whether sensory information will be processed consciously rather than preconsciously or subliminally?",
   answer="The degree of emotional intensity (or emotional appraisal) of the input."),

 dict(id=6, diff="easy", source="53data", not_found=False,
   q="What is the 'winner-take-all' mechanism in Global Workspace Theory?",
   answer="A mechanism in which multiple parallel, independent streams of unconscious processing compete for propagation, and a single stream is selected for binding and propagation."),

 dict(id=7, diff="easy", source="6data", not_found=False,
   q="What is the 'Attention Schema' in the Attention Schema Theory (AST)?",
   answer="A high-level model of the attentional process that interacts with the attentional system to enable flexible adaptation to novel contexts based on current and future behavioral needs."),

 # ================= HARD =================
 dict(id=8, diff="hard", source="0data", not_found=False,
   q="What are the two core processes of the Global Workspace Theory (GWT)?",
   answer="The 'Selection' process (which integrates information from multiple parallel specialized modules) and the 'Broadcast' process (which disseminates the selected information throughout the system)."),

 dict(id=9, diff="hard", source="13data", not_found=False,
   q="According to the GWT neurobiological model, what specific networks realize global informational access?",
   answer="Long-range excitatory neurons forming recurrent corticocortical connections between prefrontal, parietal, and cingulate regions."),

 dict(id=10, diff="hard", source="18data", not_found=False,
   q="What is the primary role of the 'Reality Monitoring' mechanism in Perceptual Reality Monitoring (PRM) theory?",
   answer="To automatically discriminate between meaningful activity in perceptual systems (caused by external stimuli, top-down expectations, memory, etc.) and random noise, identifying representations as conscious when they are deemed reliable."),

 dict(id=11, diff="hard", source="13data", not_found=False,
   q="How does the 'Local Workspace Theory' (LWT) of consciousness differ conceptually from the Global Workspace Theory regarding recurrent processing circuits?",
   answer="LWT posits that conscious experience is realized by local recurrent processing circuits, whereas GWT posits that conscious experience is realized by global recurrent processing circuits."),

 dict(id=12, diff="hard", source="18data", not_found=False,
   q="What does 'ignition' mean in the context of the Global Workspace Theory (GWT)?",
   answer="A step-function process in which sufficiently strong perceptual representations cause workspace neurons (originating in frontoparietal areas) to abruptly fire and code for their content, making the information globally broadcast and conscious."),

 dict(id=13, diff="hard", source="24data", not_found=False,
   q="What is the 'all-or-none law' of neuron firing in the prefrontal and anterior temporal cortex during conscious perception?",
   answer="The principle that neurons fire to a specific concept (and do so only when the corresponding word or image is presented consciously), serving as a neural signature of conscious perception."),

 dict(id=14, diff="hard", source="28data", not_found=False,
   q="What does the Lempel-Ziv (LZ) complexity measure estimate in spiking activity?",
   answer="It measures the approximate quantity of non-redundant information in a spike train by estimating the minimal size of the 'vocabulary' necessary to describe the sequence of data in a lossless manner."),

 dict(id=15, diff="hard", source="43data", not_found=False,
   q="What is the functional role of sleep spindles in the N1 to N2 sleep transition?",
   answer="They act as a threshold gating mechanism that actively suppresses sensory processing and cortical communication, creating a neural gate that abolishes consciousness of the external environment to consolidate Stage 2 sleep."),

 # ================= NEGATIVE (out of corpus -> must refuse) =================
 dict(id=16, diff="negative", source="", not_found=True,
   q="What was the closing price of Apple stock on its IPO day?", answer=""),
 dict(id=17, diff="negative", source="", not_found=True,
   q="How do I bake a sourdough bread at home?", answer=""),
 dict(id=18, diff="negative", source="", not_found=True,
   q="Who won the FIFA World Cup in 2018?", answer=""),
 dict(id=19, diff="negative", source="", not_found=True,
   q="What is the time complexity of quicksort in the worst case?", answer=""),
 dict(id=20, diff="negative", source="", not_found=True,
   q="What is the boiling point of ethanol at sea level?", answer=""),
]

# Key facts per question: regexes (case-insensitive) that a correct answer MUST
# all contain. This is the strict pass/fail check, like the SQL eval's value
# compare - token recall vs `answer` is only a soft score, because RAG answers
# paraphrase. Keep each regex to the core fact, with alternatives via |.
KEYS = {
    1:  [r"\bEC\b|extended cessation"],
    2:  [r"thalamus"],
    3:  [r"central sulcus"],
    4:  [r"\baccess"],
    5:  [r"emotional (intensity|appraisal)|intensity of (the )?emotion"],
    6:  [r"compet", r"\b(single|one|strongest|winner)\b"],
    7:  [r"\bmodel\b", r"attention"],
    8:  [r"select|compet", r"broadcast"],
    9:  [r"long-range|long range", r"prefrontal|frontal", r"pariet"],
    10: [r"discriminat|distinguish", r"noise"],
    11: [r"\blocal\b", r"\bglobal\b"],
    12: [r"step|non-?linear|abrupt|sudden", r"broadcast|globally available"],
    13: [r"concept|specific", r"conscious"],
    14: [r"non-?redundant|information", r"vocabulary|lossless"],
    15: [r"gat(e|ing)|threshold", r"suppress|block|abolish"],
}

if __name__ == "__main__":
    import json
    from collections import Counter
    stems = json.load(open("data/pdf_to_paperid.json"))
    print(f"Validating {len(GOLDEN)} golden entries against data/pdf_to_paperid.json\n")
    ok = 0
    for g in GOLDEN:
        good = g["not_found"] or (g["source"] in stems and g["id"] in KEYS)
        print(f"[{g['diff']:8}] #{g['id']:2}  {'OK  ' if good else 'FAIL'}  source={g['source'] or '-'}")
        ok += good
    print(f"\n{ok}/{len(GOLDEN)} passed")
    print("spread:", dict(Counter(g["diff"] for g in GOLDEN)))
