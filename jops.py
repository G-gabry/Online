from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from flask import Blueprint, jsonify


# DB connection



import uuid
from difflib import SequenceMatcher
# Initialize Sentence Transformer model for Arabic (loaded once at startup)
DEEPSEEK_API_KEY = "sk-abf137dc5ac7432aa6facad7ebbd4dfd"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

PRODUCT_DATABASE = {
    "388698940987084": {
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
def semantic_search(page_id, user_message, threshold=0.6):
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


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Example intents that imply a customer wants to buy
BUYING_KEYWORDS = ["شراء", "اشتري", "طلب", "أريد شراء"]
def embedding_func(text):
    return embedding_model.encode(text)

def is_buying_semantic(user_message, embedding_func, threshold=0.75):
    user_vec = embedding_func(user_message)
    for keyword in BUYING_KEYWORDS:
        keyword_vec = embedding_func(keyword)
        similarity = cosine_similarity([user_vec], [keyword_vec])[0][0]
        if similarity >= threshold:
            return True
    return False

conversations = {}
def generate_ai_response(product_info, user_message, conversation_history=None):
    """Enhanced response generation with more natural flow"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    # Check for buying intent
    is_buying_intent = is_buying_semantic(user_message, embedding_func)

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

import uuid
import pyodbc
from datetime import datetime




# DB connection setup



import pyodbc

def init_database(db_host: str, db_name: str):
    DRIVER_NAME = 'ODBC Driver 17 for SQL Server'
    SERVER_NAME = db_host
    DATABASE_NAME = db_name

    connection_string = f"DRIVER={{{DRIVER_NAME}}};SERVER={SERVER_NAME};DATABASE={DATABASE_NAME};Trusted_Connection=yes;"
    return pyodbc.connect(connection_string)

# Test connection
conn = init_database('localhost', 'BusinessOrders')
cursor = conn.cursor()
cursor.execute("SELECT 1")
print("✅ Connection successful!")



  # In-memory conversation tracking

def handle_order_confirmation(user_id, user_message, product_info, page_id):
    """Processes an order and inserts it into the SQL Server database."""
    try:
        # Split and extract order fields
        fields = user_message.strip().splitlines()
        if len(fields) < 8:
            return "❗️الرجاء إرسال جميع بيانات الطلب المطلوبة بالترتيب."

        name = fields[0].strip()
        print(name)
        phone = fields[1].strip()
        color = fields[2].strip()
        size = fields[3].strip().upper()
        weight = int(fields[4].strip())
        quantity = int(fields[5].strip())
        governorate = fields[6].strip()
        address = fields[7].strip()
        page_id = str(page_id)
        print(page_id)

        unit_price = product_info.get("base_price", 0)
        total_price = unit_price * quantity

        # Generate order ID for user reference (not DB primary key)
        order_id = f"ORD-{str(uuid.uuid4())[:8]}"

        # Insert order into Orders table
        cursor.execute("""
            INSERT INTO Orders (page_id, customer_name, phone, governorate, address, status, payment_method, shipping_fee, total_price, order_date)
            VALUES (?, ?, ?, ?, ?, 'قيد الانتظار', 'فودافون كاش', ?, ?, ?)
        """, (
            page_id, name, phone, governorate, address, 50, total_price, datetime.now()
        ))
        conn.commit()

        sql_order_id = cursor.execute("SELECT SCOPE_IDENTITY()").fetchone()[0]

        # Insert into OrderItems
        cursor.execute("""
            INSERT INTO OrderItems (order_id, product_name, color, size, quantity, unit_price)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            sql_order_id, product_info['name'], color, size, quantity, unit_price
        ))
        conn.commit()
        print("Done adding")
        # Update conversation state
        conversations[user_id] = {
            "order_confirmed": True,
            "order_id": order_id,
            "history": []
        }

        return f"""
        ✅ تم استلام طلبك رقم #{order_id}

        👤 الاسم: {name}
        📞 الهاتف: {phone}
        🎨 اللون: {color}
        📏 المقاس: {size}
        ⚖️ الوزن: {weight} كجم
        🔢 العدد: {quantity}
        🏙️ العنوان: {governorate} - {address}

        شكراً لثقتك بنا ❤️
        """

    except Exception as e:
        return f"❌ حدث خطأ أثناء حفظ الطلب: {str(e)}"

from intent_detector import detect_intent
def process_message(data):
    """Process messages from the queue"""
    try:
        if data.get('object') != 'page':
            return

        for entry in data.get('entry', []):
            page_id = entry['id']
            for event in entry.get('messaging', []):
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
                    continue

                product_info = PRODUCT_DATABASE.get(page_id, {})

                # Check if user is starting an order
                intent = detect_intent(user_message)

                if not conversations[sender_id]['order_started']:
                    if intent == "start_order":
                        response = generate_order_instructions(product_info)
                        conversations[sender_id]['order_started'] = True
                    elif intent == "greeting":
                        response = "أهلاً بك! كيف يمكنني مساعدتك اليوم؟"
                    elif intent == "track_order":
                        response = "من فضلك زودني برقم الطلب وسأتحقق لك فوراً 🔍"
                    else:
                        response = generate_ai_response(product_info, user_message, conversations[sender_id]["history"])
                else:
                    if intent == "restart_order":
                        conversations[sender_id] = {
                            "history": [],
                            "order_started": False,
                            "order_confirmed": False,
                            "page_id": page_id
                        }
                        response = "تم بدء طلب جديد. من فضلك أخبرني بما ترغب في شرائه 😊"
                    else:
                        response = handle_order_confirmation(sender_id, user_message, product_info, page_id)

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

    except Exception as e:
        print(f"Error processing message: {str(e)}")
        # Implement retry logic here if needed

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
PAGE_TOKENS = {
    "641855032337466": "EAAORSK8XIqcBO5DrfZBP9OGWoQWZAZAfrL6ZAp4RiLx7dNRRRoyZB5kZC7k8QZB9C2jfjt8ZCGgaSpjwqA0AbseJgZAKrB3D1SXIeCuobl9ZCxs1FIYuElTV6Y0d7Qpt7G6r0anzxMxZCY4ddiZBjI1ZBDwyaL1AaoS7ZAfcKrXdkrl7ZCk2uRiVJAX4IIzZCDswcEztzCQX5QZDZD",
    "388698940987084": "EAAORSK8XIqcBO5xezeEdYOYgZBxYR09KR1xWtQF2HUCflN0akYHs0u0zTXGEp77VJA4tZBi0fbWkX9bHAGiCStLjZB2h4lecn3yBIDGBjTZAu2qyzCKyyI2jHCoG0UkAfHmFHWwZAcVOi9ZCuIZCsZBmZAyc7ZBjQKdKC7DyHScpM9JAZCHw7ZAjM1Rxbjk5n5rwHHzpFgZDZD"
}

# In api_routes.py
orders_api = Blueprint('orders_api', __name__)

@orders_api.route('/api/orders/388698940987084')
def get_orders_by_page(page_id):
    cursor.execute("""
        SELECT o.id, o.customer_name, o.phone, o.governorate, o.address,
               o.status, o.order_date, oi.color, oi.size, oi.quantity,
               oi.unit_price, oi.product_name
        FROM Orders o
        JOIN OrderItems oi ON o.id = oi.order_id
        WHERE o.page_id = ?
    """, (page_id,))
    rows = cursor.fetchall()

    orders = {}
    for row in rows:
        order_id = row.id
        if order_id not in orders:
            orders[order_id] = {
                "id": f"#{order_id}",
                "customer": row.customer_name,
                "phone": row.phone,
                "governorate": row.governorate,
                "amount": f"{row.unit_price * row.quantity} ج.م",
                "status": row.status,
                "date": row.order_date.strftime('%Y-%m-%d'),
                "statusClass": "pending" if row.status == "قيد الانتظار" else "completed",
                "details": {
                    "address": row.address,
                    "products": [],
                    "shipping": "50 ج.م",
                    "payment": "فودافون كاش"
                }
            }

        orders[order_id]["details"]["products"].append({
            "name": row.product_name,
            "color": row.color,
            "size": row.size,
            "quantity": row.quantity,
            "price": f"{row.unit_price} ج.م"
        })

    return jsonify(list(orders.values()))
