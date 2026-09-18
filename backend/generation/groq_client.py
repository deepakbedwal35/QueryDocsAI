"""
groq_client.py

Thin wrapper around Groq's chat completion API. Builds the prompt
from the system prompt + formatted chunks + question, sends it, and
returns the raw answer text (citation verification happens
downstream in citation_verifier.py, not here).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from groq import Groq
from backend.config import GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

from .prompt_templates import SYSTEM_PROMPT, build_user_message

DEFAULT_MODEL = "openai/gpt-oss-120b"

_client: Groq | None = None

def _get_client() -> Groq:
    """Lazily construct the Groq client so import-time doesn't require
    GROQ_API_KEY to already be set (useful for tests that mock this
    module out entirely)."""
    global _client
    if _client is None:
        api_key = GROQ_API_KEY
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def generate_answer(
    question: str,
    chunks: list[dict],
    model: str = GROQ_MODEL or DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> str:
    """
    Generate an answer grounded strictly in the retrieved chunks.

    Args:
        question: The user's natural-language question.
        chunks: Retrieved chunks matching the retrieval contract
            (chunk_id, text, paper_title, authors, year, page, score).
        model: Groq model id to use.
        temperature: Kept at 0 by default for faithfulness/determinism
            — this is a grounded Q&A task, not creative writing.
        max_tokens: Cap on generated response length.

    Returns:
        Raw answer text from the LLM, including any [chunk: id]
        citation tags it produced. Not yet verified — that's
        citation_verifier.verify_citations()'s job.
    """
    client = _get_client()
    user_message = build_user_message(question, chunks)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content or ""


chunks=[{'chunk_id': '28data_c11', 'score': 0.03278688524590164, 'text': '6 \nexemplified by two of the most prevailing and influential of these theories: Global Neuronal Workspace (GNW; \n7 \n[7, 8]) and Integrated Information Theory (IIT;[9-11], but see [12-17], for a number of other theories). \n8 \n \n9 \nThe GNW posits that an external stimulus will evoke a conscious experience if the associated neural \n0 \ninformation is widely distributed across distinct brain areas and networks – most prominently in the pre-frontal \n1 \ncortex [8]. Rendering sensory information globally available results in a coherent neural assembly of sustained \n2 \nactivity and is most readily indexed via “neural ignition”, the non-linear process whereby in unconscious states \n3 \nneuronal activation profiles remain encapsulated within their specialized subsystems, whereas in conscious \n4 \nexperiences these activation patterns are widely distributed (see [18] for experimental evidence and [19] for a \n5 \nrecent computational treatise). \n6 \n \n7 \nIn a similar manner, IIT is a systems-level theory of consciousness also postulating that complex and highly \n8 \ninterconnected neural networks support subjective experience [11, 20]. In contrast to GNW, however, the \n9'}, {'chunk_id': '28data_c77', 'score': 0.031754032258064516, 'text': '23\nFigure 8. Neural ignition as a function of conscious state. A) Difference in the observed percentage of trials resulting \n4 \nin neural ignition due to AT stimulation from the percentage of trials that would be predicted based on S1 and vPM \n5 \nactivation alone (y = 0), as a function of conscious state (aware = purple; unaware = gray). B) Neural ignition due to AT \n6 \nstimulation in integrative (purple) and convergent (black) neurons as a function of consciousness state (aware = left; \n7 \nunaware = right). White dots represent means of the distribution.  \n8 \n \n9 \n \n0 \nDiscussion \n1 \n \n2 \nDetailing the neural mechanisms enabling wakefulness (i.e., consciousness-level) and conscious experiences \n3 \n(i.e., consciousness-level) is a central question within systems neuroscience [7]. As such, several theoretical \n4 \nframeworks have been put forth [7, 8, 11-17, 50]. Two of the most influential of these are Information \n5 \nIntegration Theory (IIT; [11, 50]) and Global Neuronal Workspace (GNW; [7, 8]). Unfortunately, these theories \n6 \nhave seldomly been tested neurophysiologically, and never within the same dataset. A series of concrete \n7'}, {'chunk_id': '28data_c66', 'score': 0.031054405392392875, 'text': '20\nBeyond the characterization of single cell properties (i.e., firing rate, Fano Factor, complexity), it is possible to \n9 \nleverage the fact that neurons in both S1 and vPM – a well-studied microcircuit [30] – were concurrently \n0 \nrecorded to test another prominent theory of consciousness; the GNW theory [8]. This theory states that \n1 \nsensory stimuli will elicit a conscious percept when the neural activity associated with the stimuli propagates \n2 \nthrough a broad fronto-parietal network following neural ignition. Adapting this theoretical framework to the \n3 \ncurrent experimental design, GNW predicts a higher likelihood of near concurrent neuronal firing in S1 and \n4 \nvPM (at the single trial level) when animals are in a conscious state (and thus capable of conscious content) \n5 \nthan when they are unconscious (Prediction #3). \n6 \n \n7 \nTo test this prediction we define a response threshold as exceeding spontaneous firing by two standard \n8 \ndeviations (see Methods), and then calculate the percentage of trials that result in significant firing in S1, vPM, \n9 \nor both S1 and vPM, as a function of consciousness and sensory stimulation type. This approach yields \n0'}, {'chunk_id': '22data_c26', 'score': 0.029411764705882353, 'text': 'rience of approximately 400 ms from onset of a sustained stimulus\nat threshold intensity and assumed that this latency reﬂected the\nneed for maintaining the initial frequency and amplitude for an\nextended period of time. This suggests that ampliﬁcation of per-\nceptual processing is required to elicit conscious experience, a\nprocess which in the neuronal global workspace theory is termed\n“ignition.” In addition, evidence also indicates that high signal\nintensity in cortico-thalamic interaction is essential for emergence\nof consciousness. The possible nature of such ampliﬁcation has\nbeen widely discussed in the literature. Several researchers have\nspeculated that extensive recursive activity to bootstrap neuronal\nprocessing is a general mechanism for stimuli to become available\nfor consciousness (Tononi and Edelman, 1998).\nIn spite of this long-standing interest in recurrent neuronal\nactivity, beyond some demonstrations in primary visual cor-\ntices, such activity was only recently demonstrated in higher-\norder, modality non-speciﬁc regions using MEG and the so-called\nGranger causality analysis, a mathematical method initially devel-'}, {'chunk_id': '28data_c83', 'score': 0.02904040404040404, 'text': '5 \nexperience should result in neural ignition – the broadcasting of sensory evidence throughout the brain. \n6 \nConcordant with the basic tenets of GNW, our results suggest that the co-activation of primary sensory areas \n7 \nand higher-order areas on a single trial is more likely under conscious than unconscious conditions. \n8 \nImportantly, the occurrence of this co-activation exceeded the expected values derived from the probability of \n9 \nnoting S1 and vPM activations alone. In addition to these co-activation findings, we analyzed firing rates in a \n0 \ntime-resolved fashion, which allowed us the opportunity to see whether firing rates to sensory stimulation \n1 \nduring consciousness were more sustained than during unconsciousness. As predicted by the GNW and well \n2 \nestablished in electroencephalography and electrocorticography [54, 55, 56], neural activity was more \n3 \n.\nCC-BY-NC-ND 4.0 International license\na \ncertified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under \nThe copyright holder for this preprint (which was not\nthis version posted March 21, 2019. \n;'}, {'chunk_id': '22data_c22', 'score': 0.028370221327967807, 'text': 'chrony in the paralimbic network thus meets one of the main\nrequirements needed to bind conscious experience in the so-called\nneuronal global workspace theory (Dehaene et al., 1998) based on\nearlier theoretical work (Baars, 1989).\nYet, theories of consciousness abound and while the global\nworkspace theory has been highly inﬂuential, other potential can-\ndidate theories exist. An equally important concept developed\nover the last decade is the default mode theory (Gusnard and\nRaichle, 2001; Raichle et al., 2001). According to this theory, sen-\nsory information arriving from the outside world will lead to a\ndecrease in perfusion and energy metabolism of the medial default\nmode network, in analogy with the re-direction of brain activ-\nity with shifting modalities of sensory stimulation (Lam et al.,\n1999).\nAt ﬁrst glance these two theories would seem contradictory,\nwith the global workspace theory referring to a single global brain\nnetwork processing consciousness, while the default mode the-\nory posits one brain network for processing the outer world, and\nanother network for the inner world.\nThe hypothesized paralimbic network offers a solution to this'}]
query ="""What is "neural ignition" in Global Neuronal Workspace (GNW) theory, and did the data support it"""

# print(generate_answer(question=query, chunks=chunks))

