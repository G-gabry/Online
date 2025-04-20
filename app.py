# Configuration and Imports
import requests
from flask import Flask, request, jsonify
import uuid
from difflib import SequenceMatcher

app = Flask(__name__)

# Temporary storage (replace with database in production)
PAGE_TOKENS = {
    "641855032337466": "EAAORSK8XIqcBO5DrfZBP9OGWoQWZAZAfrL6ZAp4RiLx7dNRRRoyZB5kZC7k8QZB9C2jfjt8ZCGgaSpjwqA0AbseJgZAKrB3D1SXIeCuobl9ZCxs1FIYuElTV6Y0d7Qpt7G6r0anzxMxZCY4ddiZBjI1ZBDwyaL1AaoS7ZAfcKrXdkrl7ZCk2uRiVJAX4IIzZCDswcEztzCQX5QZDZD",
    "388698940987084": "EAAORSK8XIqcBO5xezeEdYOYgZBxYR09KR1xWtQF2HUCflN0akYHs0u0zTXGEp77VJA4tZBi0fbWkX9bHAGiCStLjZB2h4lecn3yBIDGBjTZAu2qyzCKyyI2jHCoG0UkAfHmFHWwZAcVOi9ZCuIZCsZBmZAyc7ZBjQKdKC7DyHScpM9JAZCHw7ZAjM1Rxbjk5n5rwHHzpFgZDZD"
}

DEEPSEEK_API_KEY = "sk-5f8aaed2e3904f7d853c5f4dba977c1f"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

conversations = {}  # Format: {user_id: {"history": [], "order": None, "page_id": "123"}}

# Product Database
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Sentence Transformer model for Arabic (loaded once at startup)
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Enhanced PRODUCT_DATABASE with vector embeddings
PRODUCT_DATABASE = {
    "388698940987084": {  # Abayas
        "name": "عباية مطرزة",
        "base_price": 249.99,
        "description": "عباية سوداء مطرزة يدوياً بخيوط ذهبية",
        "colors": ["أسود", "بيج", "ذهبي", "أحمر داكن"],
        "sizes": ["48", "50", "52", "54"],
        "predefined_responses": {
            "الأسعار": "سعر العباية: 249.99 جنيه. خصم 10% للطلبات فوق 3 قطع.",
            "المقاسات": "المقاسات المتوفرة: 48 (صغير), 50 (متوسط), 52 (كبير), 54 (كبير جداً).",
            "الألوان": "الألوان المتوفرة: أسود، بيج، ذهبي، أحمر داكن.",
            "الشحن": "مصاريف الشحن: 30 جنيه. القاهرة والجيزة خلال يومين، المحافظات خلال 3 أيام.",
            "الاستبدال": "استبدال خلال 7 أيام برسوم 30 جنيه. لا استبدال بعد الاستخدام."
        },
        "response_embeddings": None  # Will be initialized later
    },
    # ... (other products)
}

# Initialize embeddings for all predefined responses
for page_id, product in PRODUCT_DATABASE.items():
    if 'predefined_responses' in product:
        questions = list(product['predefined_responses'].keys())
        embeddings = embedding_model.encode(questions)
        PRODUCT_DATABASE[page_id]['response_embeddings'] = dict(zip(questions, embeddings))


def semantic_search(page_id, user_message, threshold=0.8):
    """Find most similar predefined response using vector similarity"""
    product = PRODUCT_DATABASE.get(page_id, {})
    if not product.get('response_embeddings'):
        return None

    # Encode user message
    user_embedding = embedding_model.encode([user_message])[0]

    # Calculate similarities
    similarities = []
    for question, embedding in product['response_embeddings'].items():
        sim = cosine_similarity([user_embedding], [embedding])[0][0]
        similarities.append((sim, question))

    # Get best match
    if not similarities:
        return None

    max_sim, best_question = max(similarities, key=lambda x: x[0])
    if max_sim >= threshold:
        return product['predefined_responses'][best_question]
    return None


def get_predefined_response(page_id, user_message):
    """Hybrid approach using both semantic and keyword matching"""
    # First try semantic search
    semantic_response = semantic_search(page_id, user_message)
    if semantic_response:
        return semantic_response

    # Fall back to keyword matching
    product = PRODUCT_DATABASE.get(page_id, {})
    predefined = product.get('predefined_responses', {})

    # Exact match check
    for question, answer in predefined.items():
        if question.lower() in user_message.lower():
            return answer

    # Similarity check with threshold
    best_match = None
    highest_similarity = 0
    threshold = 0.65

    for question in predefined.keys():
        similarity = SequenceMatcher(None, user_message.lower(), question.lower()).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = question

    if highest_similarity >= threshold:
        return predefined[best_match]

    return None


def generate_ai_response(product_info, user_message, conversation_history=None):
    """Enhanced response generation with more natural flow"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    # Check for buying intent
    is_buying_intent = any(word in user_message.lower() for word in ["اشتري", "شراء", "طلب", "حجز"])

    # Build natural context
    context = []
    if conversation_history:
        for msg in conversation_history[-5:]:
            speaker = "العميل" if msg['role'] == 'user' else "أنت"
            context.append(f"{speaker}: {msg['content']}")
        context = "\n".join(context)

    # More natural prompt template
    prompt = f"""
    أنت مساعد مبيعات ذكي يعمل في متجر {product_info['name']}. 
    يجب أن تكون ردودك ودية، طبيعية، ومباشرة.

    معلومات المنتج:
    - الاسم: {product_info['name']}
    - السعر: {product_info['base_price']} جنيه
    - الألوان: {', '.join(product_info.get('colors', []))}
    - المقاسات: {', '.join(product_info.get('sizes', []))}

    محادثة سابقة:
    {context or 'لا يوجد محادثة سابقة'}

    رسالة العميل الجديدة:
    "{user_message}"

    التعليمات:
    1. ارد كما لو كنت شخصاً طبيعياً (ليس روبوت)
    2. استخدم كلمات مثل "أهلاً"، "طبعاً"، "بالنسبة لـ" لجعل الردود أكثر طبيعية
    3. أجب باختصار (سطرين كحد أقصى) إلا إذا طلب العميل تفاصيل أكثر
    4. عند طلب الشراء: قدم تعليمات واضحة ومختصرة
    5. أنه المحادثة بلباقة بعد تأكيد الطلب
    """

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,  # Slightly lower for more consistent responses
        "max_tokens": 200,
        "language": "ar"
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=8)
        response.raise_for_status()

        ai_message = response.json()['choices'][0]['message']['content']


        return format_response(ai_message.strip(), product_info)

    except Exception as e:
        print(f"API Error: {str(e)}")
        return format_fallback_response(product_info)


def format_response(response, product_info):
    """Make responses more natural and consistent"""
    # Add product info if response is too short
    if len(response.split()) < 5:
        return f"{response}\n\n{product_info['name']} - {product_info['base_price']} جنيه"
    return response


def format_fallback_response(product_info):
    """More natural fallback response"""
    return f"""
    أهلاً بك! للأسف لم أفهم سؤالك تماماً.

    فيما يخص {product_info['name']}:
    - السعر: {product_info['base_price']} جنيه
    - للاستفسار عن المقاسات/الألوان راسلنا
    - للطلب اكتب "أريد الشراء"
    """

# Facebook Integration
def generate_order_instructions(product_info):
    """Generate the exact structured order message from your image"""
    return """
    لتسجيل الأورد يرجي ارسال البيانات بالترتيب:
    1. الأسم بالكامل
    2. رقمين للتليفون
    3. العنوان تفصيليا (المحافظة - المدينة - الحي)
    4. العدد المطلوب
    5. الألوان المرغوبة
    6. وزن حضرتك
    """


def handle_order_confirmation(user_id, user_message, product_info):
    """Process order confirmation and close chat"""
    # Generate order ID
    order_id = f"ORD-{str(uuid.uuid4())[:8]}"

    # Get the original order message
    original_order_msg = generate_order_instructions(product_info)

    # Create confirmation message
    confirmation_msg = f"""
    ✅ تم استلام طلبك رقم #{order_id}

    {user_message}

    شكراً لثقتك! 
    """

    # Mark conversation as completed
    if user_id in conversations:
        conversations[user_id]['order_confirmed'] = True
        conversations[user_id]['order_id'] = order_id

    return confirmation_msg


@app.route('/webhook', methods=['GET', 'POST'])
def handle_webhook():
    # Handle verification
    if request.method == 'GET':
        verify_token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')

        # Replace 'YOUR_VERIFY_TOKEN' with your actual token
        if verify_token == 'Gabry':
            return challenge, 200
        return "Verification failed", 403

    # Handle incoming messages
    elif request.method == 'POST':
        data = request.json

        # Make sure this is a page subscription
        if data.get('object') != 'page':
            return jsonify({"status": "not a page event"}), 404

        for entry in data.get('entry', []):
            page_id = entry['id']
            for event in entry.get('messaging', []):
                # Skip non-message events (deliveries, read receipts, etc.)
                if not event.get('message'):
                    continue

                sender_id = event['sender']['id']
                user_message = event.get('message', {}).get('text', '').strip()

                # Initialize conversation if new user
                if sender_id not in conversations:
                    conversations[sender_id] = {
                        "history": [],
                        "order_started": False,
                        "order_confirmed": False,
                        "page_id": page_id
                    }

                # Check if conversation should end
                if conversations[sender_id].get('order_confirmed'):
                    continue  # Skip processing for completed orders

                product_info = PRODUCT_DATABASE.get(page_id, {})

                # Check if user is starting an order
                if not conversations[sender_id]['order_started']:
                    if any(word in user_message.lower() for word in ["اشتري", "شراء", "طلب", "حجز", "أريد شراء"]):
                        response = generate_order_instructions(product_info)
                        conversations[sender_id]['order_started'] = True
                    else:
                        # Normal conversation flow
                        response = get_predefined_response(page_id, user_message)
                        if not response:
                            response = generate_ai_response(product_info, user_message,
                                                            conversations[sender_id]["history"])
                else:
                    # User is in order process - confirm and close
                    response = handle_order_confirmation(sender_id, user_message, product_info)

                # Store and send message
                conversations[sender_id]["history"].append({
                    "role": "user",
                    "content": user_message
                })
                conversations[sender_id]["history"].append({
                    "role": "assistant",
                    "content": response
                })

                send_messenger_message(page_id, sender_id, response)

        return jsonify({"status": "ok"}), 200
def send_messenger_message(page_id, user_id, text):
    page_token = PAGE_TOKENS.get(page_id)
    if not page_token:
        print(f"Error: No token found for page {page_id}")
        return

    payload = {
        "recipient": {"id": user_id},
        "message": {"text": text}
    }

    # Add quick replies for order confirmation


    try:
        requests.post(
            f"https://graph.facebook.com/v19.0/{page_id}/messages",
            params={"access_token": page_token},
            json=payload,
            timeout=5
        )
    except Exception as e:
        print(f"Messenger API Error: {str(e)}")


if __name__ == '__main__':
    app.run(port=5000, debug=True)