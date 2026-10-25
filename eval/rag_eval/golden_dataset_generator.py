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

# Fictional resume uploaded into a fixed eval chat by make_golden_dataset,
# to exercise the chat_id (upload) retrieval path.
UPLOAD_CHAT_ID = "eval-upload-chat"
UPLOAD_DOC_ID = "eval-resume"
UPLOAD_FIXTURE = """Asha Verma
B.Tech in Computer Science and Engineering, Ridgeview Institute of Technology (2022-2026), CGPA 8.7

Projects
TerrainSim: a real-time GPU terrain renderer written in Rust and WGPU, with procedural
heightmaps, level-of-detail streaming and a 60 fps target on integrated graphics.
LeafLens: a plant disease classifier trained with PyTorch on 38 leaf classes, served through
a FastAPI inference endpoint and a React frontend.

Skills: Rust, Python, PyTorch, FastAPI, React, PostgreSQL, Docker
"""

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

 # ================= UPLOAD =================
 # Asked inside an eval chat that has UPLOAD_FIXTURE uploaded (chat=True).
 dict(id=21, diff="upload", source=f"upload_{UPLOAD_DOC_ID}", not_found=False, chat=True,
   q="What is the name of the candidate whose resume is given?", answer="Asha Verma."),

 dict(id=22, diff="upload", source=f"upload_{UPLOAD_DOC_ID}", not_found=False, chat=True,
   q="In which college does the person study?", answer="Ridgeview Institute of Technology."),

 dict(id=23, diff="upload", source=f"upload_{UPLOAD_DOC_ID}", not_found=False, chat=True,
   q="Give me a quick summary of the projects in this resume.",
   answer="TerrainSim, a GPU terrain renderer in Rust and WGPU, and LeafLens, a plant disease classifier built with PyTorch and served with FastAPI."),

 dict(id=24, diff="upload", source="", not_found=True, chat=True,
   q="What is the candidate's phone number?", answer=""),

 # ================= COMPLEX =================
 # Summaries, multi-fact lists, numbers, partial context and multi-paper
 # questions: long answers leave room to hallucinate, unlike the one-fact
 # questions above. source "a|b" = the answer spans both papers.
 dict(id=26, diff="complex", source="18data", not_found=False,
   q="Summarize the approach and main conclusion of the report on consciousness in artificial intelligence by Butlin, Long et al.",
   answer="It derives computational 'indicator properties' of consciousness from neuroscientific theories and uses them to assess recent AI systems; it concludes that no current AI systems are conscious but that there are no obvious technical barriers to building systems that satisfy the indicators."),

 dict(id=27, diff="complex", source="18data", not_found=False,
   q="Which scientific theories of consciousness does the Butlin et al. report survey to derive its indicator properties?",
   answer="Recurrent processing theory, global workspace theory, higher-order theories, predictive processing and attention schema theory."),

 dict(id=28, diff="complex", source="6data", not_found=False,
   q="Which three theories of conscious function do Juliani et al. examine, and what do they conclude about current deep learning methods?",
   answer="Global Workspace Theory, Information Generation Theory and Attention Schema Theory; while still far from general intelligence, many state-of-the-art deep learning methods have begun to incorporate key aspects of each of the three theories."),

 dict(id=29, diff="complex", source="43data", not_found=False,
   q="In the sleep EEG study of consciousness transitions, how many transitions and subjects were analysed, and what share of transition types showed gradual versus threshold dynamics?",
   answer="622 sleep stage transitions across 12 subjects; three of four transition types (75%) were gradual and one (25%, N1 to N2) showed threshold behaviour."),

 dict(id=30, diff="complex", source="43data", not_found=False,
   q="According to the sleep EEG study, which sleep transitions are consistent with IIT and which with Global Workspace Theory?",
   answer="Wake to N1, N2 to N3 and REM to Wake follow gradual dynamics consistent with IIT; N1 to N2 shows threshold behaviour consistent with GWT, via thalamic spindle gating."),

 dict(id=31, diff="complex", source="84data", not_found=False,
   q="What findings of the Cogitate adversarial collaboration challenge IIT and GNWT respectively?",
   answer="For IIT, a lack of sustained synchronization within posterior cortex; for GNWT, the general lack of ignition at stimulus offset and limited representation of some conscious dimensions in prefrontal cortex."),

 dict(id=32, diff="complex", source="84data", not_found=False,
   q="How many subjects took part in the Cogitate adversarial collaboration, and which neural recording methods were used?",
   answer="256 subjects, measured with fMRI, magnetoencephalography (MEG) and electrocorticography (ECoG)."),

 # partial context: the handedness of subjects is not in the corpus
 dict(id=33, diff="complex", source="43data", not_found=False,
   q="How many subjects were in the sleep EEG consciousness-transition study, and how many of them were left-handed?",
   answer="12 subjects; the handedness of the subjects is not reported in the provided documents."),

 dict(id=34, diff="complex", source="6data|18data", not_found=False,
   q="Which theories of consciousness are used both by Juliani et al. (conscious function and general intelligence) and by Butlin et al. (consciousness in AI)?",
   answer="Global Workspace Theory and Attention Schema Theory appear in both."),

 dict(id=35, diff="complex", source="43data|84data", not_found=False,
   q="How do the sleep EEG transition study and the Cogitate adversarial collaboration each test IIT against global workspace theories?",
   answer="The sleep study measures whether sleep-stage transitions are gradual (IIT) or threshold-like (GWT) and finds both occur; Cogitate tests preregistered divergent predictions with fMRI, MEG and ECoG and finds evidence challenging both theories."),

 # corpus question inside an upload chat: uploads must not crowd out the corpus
 dict(id=25, diff="upload", source="53data", not_found=False, chat=True,
   q="What is the primary hub of consciousness in the unified 3D default space model proposed by Jerath et al.?",
   answer="The thalamus."),
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
    21: [r"asha verma"],
    22: [r"ridgeview"],
    23: [r"terrainsim", r"leaflens"],
    25: [r"thalamus"],
    26: [r"indicator", r"no current AI|not (currently )?conscious|none of the", r"barrier"],
    27: [r"recurrent processing", r"global workspace", r"higher.order", r"predictive processing", r"attention schema"],
    28: [r"global workspace", r"information generation", r"attention schema", r"incorporat|begun|adopt"],
    29: [r"\b622\b", r"\b12\b", r"75 ?%|three of (the )?four", r"N1\W{0,5}(to\W{0,3})?N2"],
    30: [r"Wake\W{0,5}(to\W{0,3})?N1", r"N2\W{0,5}(to\W{0,3})?N3", r"REM\W{0,5}(to\W{0,3})?Wake", r"N1\W{0,5}(to\W{0,3})?N2"],
    31: [r"synchroni", r"posterior", r"ignition", r"prefrontal"],
    32: [r"\b256\b", r"fMRI|functional magnetic", r"MEG|magnetoencephalograph", r"ECoG|electrocorticograph"],
    33: [r"\b12\b", r"not (reported|mentioned|stated|provided|specified|available)|no information|cannot find|does not (report|mention|state|specify)"],
    34: [r"global workspace", r"attention schema"],
    35: [r"gradual|threshold", r"ignition|synchroni|preregist|adversarial|challeng"],
}

if __name__ == "__main__":
    import json
    from collections import Counter
    stems = json.load(open("data/pdf_to_paperid.json"))
    print(f"Validating {len(GOLDEN)} golden entries against data/pdf_to_paperid.json\n")
    ok = 0
    for g in GOLDEN:
        good = g["not_found"] or (all(s in stems or s.startswith("upload_")
                                      for s in g["source"].split("|")) and g["id"] in KEYS)
        print(f"[{g['diff']:8}] #{g['id']:2}  {'OK  ' if good else 'FAIL'}  source={g['source'] or '-'}")
        ok += good
    print(f"\n{ok}/{len(GOLDEN)} passed")
    print("spread:", dict(Counter(g["diff"] for g in GOLDEN)))
