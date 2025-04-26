from sentence_transformers import SentenceTransformer, util

# Load only once
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Define examples of user messages mapped to semantic intent
INTENT_EXAMPLES = {
    "buying_intent": ["أريد شراء", "حابب أطلب", "اشتري", "عايز أطلب", "ممكن أطلب", "احجزلي واحد"],
    "confirm_order": ["تمام", "أوكي", "تم", "أكد الطلب","عايز عدد من قطع", "يلا بينا", "ابدأ"],
    "cancel_order": ["مش عايز", "غيرت رأيي", "الغاء", "ألغى"],
    "track_order": ["فين الطلب", "متأخر ليه", "وصل امتى", "رقم الطلب"],
    "restart_order": ["طلب جديد", "ابدأ من جديد", "عايز اطلب تاني"],
    "change_order": ["أغير", "عدلت", "نسيت", "أعدل"]
}

# Precompute embeddings
# Convert all intent examples into embeddings
intent_embeddings = {}

for intent, phrases in INTENT_EXAMPLES.items():
    embedded_phrases = []
    for phrase in phrases:
        embedded_phrases.append(embedding_model.encode(phrase, convert_to_tensor=True))
    intent_embeddings[intent] = embedded_phrases


def detect_intent(user_message: str, threshold: float = 0.7) -> str:
    """Return the best matching intent or 'unknown'"""
    user_embed = embedding_model.encode(user_message, convert_to_tensor=True)

    best_intent = "unknown"
    best_score = 0

    for intent, examples in intent_embeddings.items():
        for ex_embed in examples:
            score = util.cos_sim(user_embed, ex_embed).item()
            if score > best_score:
                best_score = score
                best_intent = intent

    return best_intent if best_score >= threshold else "unknown"
