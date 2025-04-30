from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from flask import Blueprint, jsonify


# DB connection



import uuid
from difflib import SequenceMatcher
# Initialize Sentence Transformer model for Arabic (loaded once at startup)
DEEPSEEK_API_KEY = "sk-6e8a4b83b53343568cbedc3b951c6f7"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

PRODUCT_DATABASE = {
    "388698940987084": {  # Assuming this is the T-shirt product page ID
        "name": "تيشرت أوفر سايز قطن 100%",
        "base_price": 250,
        "description": "تيشرت قطني عالي الجودة بتصميم أوفر سايز",
        "colors": ["أبيض", "أسود", "منت جرين", "رمادي", "بيج"],
        "sizes": {
            "XL": {"weight_range": "65-150 كجم", "dimensions": "XL"},
            "2XL": {"weight_range": "75-165 كجم", "dimensions": "2XL"},
            "3XL": {"weight_range": "90-175 كجم", "dimensions": "3XL"},
            "4XL": {"weight_range": "115-190 كجم", "dimensions": "4XL (خاص)"},
            "5XL": {"weight_range": "130-115 كجم", "dimensions": "5XL (خاص)"}
        },
        "العروض": {
            "2_pieces": 440,
            "3_pieces": 580,
            "4_pieces": 700,
            "5_pieces": 800
        },
        "سياسة الاستبدال": {
            "القاهرة والجيزة": 50,
            "المحافظات": 60,
            "الصعيد": 70
        },
        "predefined_responses": {
            "الأسعار او العروض": "سعر القطعة: 250 جنيه\nالعروض:\n- 2 قطع: 440 جنيه\n- 3 قطع: 580 جنيه\n- 4 قطع: 700 جنيه\n- 5 قطع: 800 جنيه",
            "المقاسات": (
                "المقاسات المتوفرة:\n"
                "XL: 65-150 كجم\n"
                "2XL: 75-165 كجم\n"
                "3XL: 90-175 كجم\n"
                "4XL: 115-190 كجم (مقاسات خاصة)\n"
                "5XL: 130-115 كجم"
            ),
            "الألوان": "الألوان المتوفرة: أبيض، أسود، منت جرين، رضامي، بيع",
            "الشحن": (
                "مصاريف الشحن:\n"
                "- القاهرة والجيزة: 50 جنيه\n"
                "- المحافظات: 60 جنيه\n"
                "- الصعيد: 70 جنيه"
            ),
            "الاستبدال": "يمكن استبدال المقاس خلال 3 أيام من الاستلام برسوم شحن إضافية"
        },
        "response_embeddings": None
    },

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


conversations = {}
def generate_ai_response(product_info, user_message, conversation_history=None):
    """Enhanced response generation with more natural flow"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    # Check for buying intent


    # Build natural context
    context = []
    if conversation_history:
        for msg in conversation_history[-8:]:
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
    4. عند طلب الشراء: قدم تعليمات واضحة ومختصرةو حاول الا تكرر نفس المعلومات 
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
    """Generate structured order message with exact format from the image"""
    return """
📋 *طريقة إرسال الطلب* 📋

_لضمان سرعة معالجة طلبك، يرجى إرسال البيانات بالضبط كما يلي:_

الاسم بالكامل  
رقم التليفون  
اللون  
المقاس  
الوزن  
الكمية  
المحافظة  
الحي/المنطقة  

*مثال:*
عمر على  
01007549327  
أسود  
  54
  XL  
  2  
الجيزة  
الشوبك الغربي  

🔹 ملاحظات:
1. اكتب كل بند في سطر منفصل
2. لا تستخدم علامات الترقيم مثل ( : أو - )
3. تأكد من صحة رقم الهاتف
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
conn = init_database('localhost', 'ordersDB')
cursor = conn.cursor()
cursor.execute("SELECT 1")
print("✅ Connection successful!")



  # In-memory conversation tracking
import json
from typing import Union, Dict


def extract_order_info(user_message: str, product_info: dict) -> Union[Dict, str]:
    """
    Enhanced order info extraction with comprehensive validation
    Args:
        user_message: The user's raw order message
        product_info: Dictionary containing product details (colors, sizes, etc.)
    Returns:
        Dict: If all fields are properly extracted (contains all required keys)
        str: Error message if missing/invalid fields
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    استخرج المعلومات التالية من الرسالة بدقة وعالج أي أخطاء:

    *المعلومات المطلوبة:*
    1. الاسم الكامل (عربي فقط)
    2. رقم الهاتف (11 رقم يبدأ ب 01)
    3. اللون (من الألوان المتاحة: {', '.join(product_info['colors'])})
    4. المقاس (من المقاسات المتاحة: {', '.join(product_info['sizes'].keys())})
    5. عدد القطع (رقم فقط)
    6. المحافظة (من محافظات مصر)
    7. العنوان التفصيلي

    *تعليمات صارمة:*
    - إذا كان أي حقل ناقص أو غير صالح، اذكر جميع الأخطاء مع مثال تصحيح
    - رقم الهاتف يجب أن يكون 11 رقم بدون مسافات
    - المحافظة يجب أن تكون من محافظات مصر المعروفة
    - إذا كانت الرسالة غير مكتملة، أعد نصاً واضحاً يطلب البيانات الناقصة

    *رسالة العميل:*
    "{user_message}"

    *أمثلة للرد عند وجود أخطاء:*
    - "نقص البيانات: يرجى إرسال الاسم الكامل ورقم الهاتف والمحافظة"
    - "المقاس غير صحيح: المقاسات المتاحة هي XL, 2XL, 3XL"
    - "رقم الهاتف يجب أن يكون 11 رقم يبدأ ب 01"

    *تنسيق الإخراج عند اكتمال البيانات:*
    {{
        "الاسم": "...",
        "الهاتف": "...",
        "اللون": "...",
         "الوزن": "...",
        "المقاس": "...",
        "الكمية": "...",
        "المحافظة": "...",
        "الحي": "..."
    }}
    """

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200,
        "language": "ar"
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        ai_message = response.json()['choices'][0]['message']['content'].strip()

        # Try to parse as JSON if complete
        try:
            order_data = json.loads(ai_message)
            required_fields = ['الاسم', 'الهاتف', 'اللون','الوزن', 'المقاس','الكمية', 'المحافظة', 'الحي']
            if all(key in order_data for key in required_fields):
                return order_data
            return ai_message  # Return error message if missing fields
        except json.JSONDecodeError:
            return ai_message  # Return the LLM's error message

    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {str(e)}")
        return "حدث خطأ في الاتصال. يرجى إعادة المحاولة لاحقاً"
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return "حدث خطأ غير متوقع أثناء معالجة طلبك"


def handle_order_confirmation(user_id: str, user_message: str, product_info: dict, page_id: str) -> str:
    """
    Processes an order with comprehensive validation and database insertion

    Args:
        user_id: Unique identifier for the user
        user_message: The user's order message
        product_info: Dictionary containing product details (colors, sizes, etc.)
        page_id: ID of the Facebook page/store

    Returns:
        str: Success message with order details or error message
    """
    try:
        extracted_data = extract_order_info(user_message, product_info)

        # Case 1: Got complete JSON data
        if isinstance(extracted_data, dict):
            # Validate color availability
            if extracted_data['اللون'] not in product_info['colors']:
                return f"اللون غير متاح. الألوان المتوفرة: {', '.join(product_info['colors'])}"

            # Validate size availability
            if extracted_data['المقاس'] not in product_info['sizes']:
                return f"المقاس غير متاح. المقاسات المتوفرة: {', '.join(product_info['sizes'].keys())}"

            # Calculate total price
            quantity = int(extracted_data['الكمية'])
            total_price = product_info['base_price'] * quantity

            # Insert into Orders table
            cursor.execute("""
                INSERT INTO Orders (
                    page_id, 
                    customer_name, 
                    phone, 
                    governorate, 
                    address, 
                    status, 
                    payment_method, 
                    shipping_fee, 
                    total_price, 
                    order_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(page_id),
                extracted_data['الاسم'],
                extracted_data['الهاتف'],
                extracted_data['المحافظة'],
                extracted_data['العنوان'],
                'قيد الانتظار',
                'فودافون كاش',
                50,  # shipping fee
                total_price,
                datetime.now()
            ))
            conn.commit()

            # Get the inserted order ID
            sql_order_id = cursor.execute("SELECT SCOPE_IDENTITY()").fetchone()[0]

            # Insert into OrderItems table
            cursor.execute("""
                INSERT INTO OrderItems (
                    order_id, 
                    product_name, 
                    color, 
                    size, 
                    quantity, 
                    unit_price
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                sql_order_id,
                product_info['name'],
                extracted_data['اللون'],
                extracted_data['المقاس'],
                quantity,
                product_info['base_price']
            ))
            conn.commit()

            # Generate readable order ID
            order_id = f"ORD-{str(uuid.uuid4())[:8]}"

            # Update conversation state
            conversations[user_id] = {
                "order_confirmed": True,
                "order_id": order_id,
                "history": [],
                "order_details": extracted_data
            }

            return f"""
            ✅ تم تأكيد طلبك #{order_id}
            --------------------------
            الاسم: {extracted_data['الاسم']}
            الهاتف: {extracted_data['الهاتف']}
            المنتج: {product_info['name']}
            اللون: {extracted_data['اللون']}
            المقاس: {extracted_data['المقاس']}
            الكمية: {extracted_data['الكمية']}
            العنوان: {extracted_data['المحافظة']} - {extracted_data['العنوان']}
            السعر الإجمالي: {total_price} جنيه (شامل الشحن)
            --------------------------
            شكراً لثقتك بنا! سيتم التواصل معك لتأكيد التفاصيل.
            """

        # Case 2: Got error message from LLM
        else:
            return f"""
            ⚠️ لم نستلم جميع البيانات المطلوبة
            --------------------------
            {extracted_data}
            --------------------------
            📋 يرجى إرسال البيانات كاملة بالشكل التالي:

            الاسم الكامل
            01012345678
            اللون ({', '.join(product_info['colors'])})
            المقاس ({', '.join(product_info['sizes'].keys())})
            الكمية
            المحافظة
            العنوان التفصيلي
            """

    except KeyError as e:
        error_msg = f"Missing field in data: {str(e)}"
        print(f"KeyError: {error_msg}")
        return "حدث خطأ في معالجة البيانات. يرجى إعادة إرسال المعلومات"
    except ValueError as e:
        error_msg = f"Invalid value: {str(e)}"
        print(f"ValueError: {error_msg}")
        return "يبدو أن هناك خطأ في إدخال البيانات. يرجى المراجعة وإعادة المحاولة"
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"Exception: {error_msg}")
        return "عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة لاحقاً"


def cancel_order_by_id(order_id):
    cursor.execute("UPDATE Orders SET status = 'تم الإلغاء' WHERE id = ?", (order_id,))
    conn.commit()
def get_tracking_status(order_id):
    cursor.execute("SELECT status FROM Orders WHERE id = ?", (order_id,))
    result = cursor.fetchone()
    if result:
        return f"🕓 حالة طلبك رقم {order_id}: {result.status}"
    else:
        return "❌ لم يتم العثور على الطلب. تأكد من رقم الطلب."

def is_structured_order_message(message: str) -> bool:
    """Check if message looks like an order with 8 expected fields."""
    lines = message.strip().splitlines()
    return len(lines) >=5

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

                product_info = PRODUCT_DATABASE.get(page_id, {})
                intent = detect_intent(user_message)
                response = None  # 🛡️ Safe default

                # Handle post-confirmation actions
                if conversations[sender_id].get("order_confirmed"):
                    order_id = conversations[sender_id].get("order_id")

                    if intent == "track_order":
                        response = get_tracking_status(order_id)

                    elif intent == "change_order":
                        response = "تم فتح طلبك للتعديل، من فضلك أرسل البيانات الجديدة كاملة."
                        conversations[sender_id]["order_confirmed"] = False
                        conversations[sender_id]["order_started"] = True

                    elif intent == "cancel_order":
                        cancel_order_by_id(order_id)
                        response = f"✅ تم إلغاء الطلب رقم {order_id} بنجاح. شكراً لك!"
                        conversations.pop(sender_id)

                    elif intent == "confirm_order" or is_structured_order_message(user_message):
                        response = handle_order_confirmation(sender_id, user_message, product_info, page_id)

                    else:
                        response = "تم تأكيد طلبك مسبقاً. هل ترغب في تتبع أو تعديل الطلب؟"

                else:
                    if not conversations[sender_id]['order_started']:
                        if intent == "buying_intent":
                            response = generate_order_instructions(product_info)
                            conversations[sender_id]['order_started'] = True

                        elif intent == "track_order":
                            response = "من فضلك زودني برقم الطلب وسأتحقق لك فوراً 🔍"

                        else:
                            predefined_response = get_predefined_response(page_id, user_message)
                            if predefined_response:
                                response = predefined_response
                            else:
                                response = generate_ai_response(
                                    product_info,
                                    user_message,
                                    conversations[sender_id]["history"]
                                )
                    else:
                        # User already started an order but not confirmed — maybe confirming now
                        if is_structured_order_message(user_message) or intent == "confirm_order":
                            response = handle_order_confirmation(sender_id, user_message, product_info, page_id)
                        else:
                            response = generate_ai_response(
                                product_info,
                                user_message,
                                conversations[sender_id]["history"]
                            )

                # 🧠 Save the conversation history
                conversations[sender_id]["history"].append({
                    "role": "user",
                    "content": user_message
                })
                conversations[sender_id]["history"].append({
                    "role": "assistant",
                    "content": response
                })

                # ✅ Send the message
                send_messenger_message(page_id, sender_id, response)

    except Exception as e:
        print(f"❌ Error processing message: {str(e)}")




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

@orders_api.route('/orders/<int:user_id>')
def get_orders_by_user(user_id):
    cursor.execute("""
        select o.id, o.customer_name, o.phone, o.governorate, o.address,
               o.status, o.order_date, oi.color, oi.size, oi.quantity,
               oi.unit_price, oi.product_name
			   from Orders o join OrderItems oi on (o.id=oi.order_id) join Pages p on(o.page_id=p.page_id)
			   where p.user_id=?
    """, (user_id,))
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
