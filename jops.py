from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from intent_detector import detect_intent
import requests
from flask import Blueprint, jsonify
import uuid
from difflib import SequenceMatcher
import numpy as np
import pyodbc
from datetime import datetime, timedelta
import re
import json
from typing import Union, Dict

# Initialize Sentence Transformer model for Arabic (loaded once at startup)
<<<<<<< Updated upstream
DEEPSEEK_API_KEY = "sk-6e8a4b83b53343568cbedc3b951c6f7"
=======

DEEPSEEK_API_KEY = "sk-69ee481627614340aec53974cb6a714e"
>>>>>>> Stashed changes
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
        "temperature": 0.3,  # Slightly lower for more consistent responses
        "max_tokens": 200,
        "language": "ar"
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=25)
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




# DB connection setup




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



def extract_order_info(user_message: str, product_info: dict ,extractedd_history_info=None) -> Union[Dict, str]:
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

    *(اسال العميل للتحقق من صحتها)معلومات مستخرجه مسبقا:*
    "{extractedd_history_info}"

    *رسالة العميل:*
    "{user_message}"

    *أمثلة للرد عند وجود أخطاء:*
    - "نقص البيانات: يرجى إرسال الاسم الكامل ورقم الهاتف والمحافظة"
    - "المقاس غير صحيح: المقاسات المتاحة هي XL, 2XL, 3XL"
    - "رقم الهاتف يجب أن يكون 11 رقم يبدأ ب 01"
    - "تم تغيير رقم الهاتف إلى 01012345678. هل هذا صحيح؟"


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
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=35
                                 )
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


def handle_order_confirmation(user_id: str, user_message: str, product_info: dict, page_id: str, extractedd_history_info) -> str:
    """
    Processes an order with comprehensive validation and database insertion
    using proper transaction management.

    Args:
        user_id: Unique identifier for the user
        user_message: The user's order message
        product_info: Dictionary containing product details (colors, sizes, etc.)
        page_id: ID of the Facebook page/store

    Returns:
        str: Success message with order details or error message
    """
    try:
        # Start transaction
        conn.autocommit = False

        extracted_data = extract_order_info(user_message, product_info, extractedd_history_info)

        # Case 1: Got complete JSON data
        if isinstance(extracted_data, dict):
            # Validate color availability
            if extracted_data['اللون'] not in product_info['colors']:
                conn.rollback()
                return f"اللون غير متاح. الألوان المتوفرة: {', '.join(product_info['colors'])}"

            # Validate size availability
            if extracted_data['المقاس'] not in product_info['sizes']:
                conn.rollback()
                return f"المقاس غير متاح. المقاسات المتوفرة: {', '.join(product_info['sizes'].keys())}"

            # Validate phone number format
            if 'الهاتف' in extracted_data:
                if not re.match(r'^01[0-9]{9}$', extracted_data['الهاتف']):
                    conn.rollback()
                    return "رقم الهاتف غير صحيح. يجب أن يبدأ بـ 01 ويتكون من 11 رقماً"

            try:
                quantity = int(extracted_data['الكمية'])
                if quantity <= 0:
                    conn.rollback()
                    return "❗️الكمية يجب أن تكون رقم موجب"
                total_price = product_info['base_price'] * quantity
            except ValueError:
                conn.rollback()
                return "❗️الرجاء إدخال كمية صحيحة (رقم فقط)"

            try:
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

                # Commit transaction if all operations succeeded
                conn.commit()

                # Generate readable order ID
                order_id = f"ORD-{sql_order_id}"  # Using actual DB ID for better tracking

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

            except Exception as db_error:
                conn.rollback()
                error_msg = f"Database error: {str(db_error)}"
                print(f"Database Error: {error_msg}")
                return "حدث خطأ في قاعدة البيانات. يرجى المحاولة مرة أخرى"

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
        conn.rollback()
        error_msg = f"Missing field in data: {str(e)}"
        print(f"KeyError: {error_msg}")
        return "حدث خطأ في معالجة البيانات. يرجى إعادة إرسال المعلومات"
    except ValueError as e:
        conn.rollback()
        error_msg = f"Invalid value: {str(e)}"
        print(f"ValueError: {error_msg}")
        return "يبدو أن هناك خطأ في إدخال البيانات. يرجى المراجعة وإعادة المحاولة"
    except Exception as e:
        conn.rollback()
        error_msg = f"Unexpected error: {str(e)}"
        print(f"Exception: {error_msg}")
        return "عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة لاحقاً"
    finally:
        # Ensure connection is returned to autocommit mode
        conn.autocommit = True

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
def extract_order_id(text):
    """Extract order ID from text using regex"""
    patterns = [
        r"ORD-\w{8}",  # ORD-1234ABCD
        r"#(\d+)",     # #1234
        r"الطلب (\d+)"  # الطلب 1234
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1) if pattern != r"ORD-\w{8}" else match.group(0)
    return None

def is_within_change_period(order_date):
    """Check if order is within 3-day change period"""
    return datetime.now() - order_date <= timedelta(days=3)

def handle_change_order(user_id, user_message, page_id):
    """Process order change request"""
    try:
        conn.autocommit = False
        # Extract order ID from message
        order_id = extract_order_id(user_message)
        if not order_id:
            return "❗️الرجاء ذكر رقم الطلب الذي تريد تعديله (مثال: ORD-1234ABCD أو #1234)"

        # Check if order exists and is within change period
        cursor.execute("""
            SELECT id, order_date FROM Orders
            WHERE (id = ? OR CONCAT('ORD-', id) = ?)
            AND page_id = ?
        """, (order_id.replace("#", "").replace("ORD-", ""), order_id, page_id))

        order = cursor.fetchone()
        if not order:
            return "❗️لم يتم العثور على الطلب. يرجى التحقق من رقم الطلب والمحاولة مرة أخرى."

        if not is_within_change_period(order.order_date):
            return "❗️لا يمكن تعديل الطلب بعد مرور 3 أيام من تاريخ الطلب."



        return """
📝 جاهز لتعديل طلبك #{order_id}.
الرجاء إرسال التعديلات المطلوبة بالشكل التالي:

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

    except Exception as e:
        conn.rollback()
        return f"❌ حدث خطأ أثناء معالجة طلب التعديل: {str(e)}"

    finally:
        conn.autocommit = True

def handle_cancel_order(user_id, user_message, page_id):
    """Process order cancellation request"""
    try:
        # Extract order ID from message
        order_id = extract_order_id(user_message)
        if not order_id:
            return "❗️الرجاء ذكر رقم الطلب الذي تريد إلغاءه (مثال: ORD-1234ABCD أو #1234)"

        # Check if order exists and is within change period
        cursor.execute("""
            SELECT id, order_date FROM Orders
            WHERE (id = ? OR CONCAT('ORD-', id) = ?)
            AND page_id = ?
            AND status = 'قيد الانتظار'
        """, (order_id.replace("#", "").replace("ORD-", ""), order_id, page_id))

        order = cursor.fetchone()
        if not order:
            return "❗️لم يتم العثور على الطلب أو قد يكون قد تم شحنه بالفعل."

        if not is_within_change_period(order.order_date):
            return "❗️لا يمكن إلغاء الطلب بعد مرور 3 أيام من تاريخ الطلب."

        # Update order status
        cursor.execute("""
            UPDATE Orders SET status = 'ملغي'
            WHERE id = ? AND page_id = ?
        """, (order.id, page_id))
        conn.commit()

        return f"""
        ✅ تم إلغاء الطلب #{order_id} بنجاح.
        إذا كان قد تم الدفع بالفعل، سيتم استرداد المبلغ خلال 3-5 أيام عمل.
        شكراً لاستخدامك خدماتنا.
        """

    except Exception as e:
        return f"❌ حدث خطأ أثناء معالجة طلب الإلغاء: {str(e)}"

def process_changes(user_id, user_message, page_id):
    """Process the actual order changes"""
    try:
        conv = conversations.get(user_id, {})
        if not conv.get("changing_order") or not conv.get("awaiting_changes"):
            return "❗️لم يتم استلام بيانات التعديل بشكل صحيح. يرجى المحاولة مرة أخرى."

        # Get product info
        product_info = PRODUCT_DATABASE.get(page_id, {})

        # Parse changes
        extracted_data = extract_order_info(user_message, product_info)

        # Case 1: Got complete JSON data
        if isinstance(extracted_data, dict):
            # Validate color availability
            if 'اللون' in extracted_data and extracted_data['اللون'] not in product_info['colors']:
                return f"اللون غير متاح. الألوان المتوفرة: {', '.join(product_info['colors'])}"

            # Validate size availability
            if 'المقاس' in extracted_data and extracted_data['المقاس'] not in product_info['sizes']:
                return f"المقاس غير متاح. المقاسات المتوفرة: {', '.join(product_info['sizes'])}"

                if 'الهاتف' in extracted_data:
                  if not re.match(r'^01[0-9]{9}$', extracted_data['الهاتف']):
                      return "رقم الهاتف غير صحيح. يجب أن يبدأ بـ 01 ويتكون من 11 رقماً"

            # Get the order ID from conversation state
            order_id = conv.get("order_id")
            if not order_id:
                return "❗️لم يتم العثور على رقم الطلب. يرجى بدء عملية التعديل من جديد."

            # Calculate total price if quantity is being changed
            total_price = None
            if 'الكمية' in extracted_data:
                try:
                    quantity = int(extracted_data['الكمية'])
                    if quantity <= 0:
                        return "❗️الكمية يجب أن تكون رقم موجب"
                    total_price = product_info['base_price'] * quantity
                except ValueError:
                    return "❗️الرجاء إدخال كمية صحيحة (رقم فقط)"

            # Update Orders table if address or other customer info changed
            update_order_sql = """
                UPDATE Orders SET
                    customer_name = COALESCE(?, customer_name),
                    phone = COALESCE(?, phone),
                    governorate = COALESCE(?, governorate),
                    address = COALESCE(?, address),
                    total_price = COALESCE(?, total_price),
                    modified_date = ?
                WHERE id = ? AND page_id = ?
            """

            cursor.execute(update_order_sql, (
                extracted_data.get('الاسم'),
                extracted_data.get('الهاتف'),
                extracted_data.get('المحافظة'),
                extracted_data.get('العنوان'),
                total_price,
                datetime.now(),
                order_id.replace("ORD-", ""),  # Remove prefix if present
                str(page_id)
            ))

            # Update OrderItems table for product-specific changes
            update_items_sql = """
                UPDATE OrderItems SET
                    color = COALESCE(?, color),
                    size = COALESCE(?, size),
                    quantity = COALESCE(?, quantity)
                WHERE order_id = ?
            """

            cursor.execute(update_items_sql, (
                extracted_data.get('اللون'),
                extracted_data.get('المقاس'),
                extracted_data.get('الكمية'),
                order_id.replace("ORD-", "")  # Remove prefix if present
            ))

            conn.commit()


            return f"""
            ✅ تم تحديث طلبك #{order_id} بنجاح
            --------------------------
            {'الاسم: ' + extracted_data['الاسم'] if 'الاسم' in extracted_data else ''}
            {'الهاتف: ' + extracted_data['الهاتف'] if 'الهاتف' in extracted_data else ''}
            {'المنتج: ' + product_info['name']}
            {'اللون: ' + extracted_data['اللون'] if 'اللون' in extracted_data else ''}
            {'المقاس: ' + extracted_data['المقاس'] if 'المقاس' in extracted_data else ''}
            {'الكمية: ' + str(extracted_data['الكمية']) if 'الكمية' in extracted_data else ''}
            {'العنوان: ' + extracted_data['المحافظة'] + ' - ' + extracted_data['العنوان']
             if 'المحافظة' in extracted_data and 'العنوان' in extracted_data else ''}
            {'السعر الإجمالي: ' + str(total_price) + ' جنيه (شامل الشحن)' if total_price else ''}
            --------------------------
            تم تحديث البيانات بنجاح. سيتم التواصل معك إذا لزم الأمر.
            """

    except Exception as e:
        return f"❌ حدث خطأ أثناء حفظ التعديلات: {str(e)}"

def extract_history_info(user_message,extractedd_history_info=None)->dict:
  headers = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
    }


  prompt =f"""
    مهمتك استخراج وتقييم المعلومات من رسالة العميل مع تحليل السياق.

    السجل الحديث (الأحدث للأقدم):
    {extractedd_history_info}

    التعليمات:
    1. قيم كل حقل بناء على الثقة (0-100%)
    2. استخدم فقط المعلومات الصريحة
    3. تجاهل البيانات غير المؤكدة
    4. حدد إذا كانت الرسالة تحتوي على تحديثات
    5. رقم الهاتف يجب أن يكون 11 رقم بدون مسافات
    6. المحافظة يجب أن تكون من محافظات مصر المعروفة

    الاخراج المطلوب:
    {{
        "الاسم بالكامل": {{"value": "...", "confidence": 0-100}},
        "رقم التليفون": {{"value": "...", "is_updated": bool}},
        "الوزن": {{"value": "...", "is_updated": bool}},
        "المحافظة": {{"value": "...", "is_updated": bool}},
        "الحي/المنطقة": {{"value": "...", "is_updated": bool}},
        "order_data": {{
            "اللون": "...",
            "المقاس": "...",
            "الكمية": "..."
        }},
        "الحقول المفقودة": ["حقل1", ...],
        "يتطلب التأكيد": bool
    }}
    """

  payload = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.2,
    "max_tokens": 200,
    "language": "ar"
}

  try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        ai_message = response.json()['choices'][0]['message']['content']


        return ai_message or{}

  except Exception as e:
        print(f"API Error: {str(e)}")
        return "failed to extract_history_info using LLM"


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

                user_id = event['sender']['id']
                user_message = event.get('message', {}).get('text', '').strip()

                # Initialize conversation if new user
                if user_id not in conversations:
                    conversations[user_id] = {
                        "history": [],
                        "extracted_history_info": {},
                        "order_started": False,
                        "changing_order": False,
                        "canceling_order": False,
                        "awaiting_changes": False,
                        "track_order": False,
                        "page_id": page_id
                    }

                product_info = PRODUCT_DATABASE.get(page_id, {})
                conv = conversations[user_id]
                intent = detect_intent(user_message, conv["history"])

                extracted_history_info = extract_history_info(user_message, conv["extracted_history_info"])
                conv["extracted_history_info"].update(extracted_history_info)

                if conv['order_started']:
                    response = handle_order_confirmation(user_id, user_message, product_info, page_id,
                                                         extracted_history_info)
                    conv['order_started'] = False
                elif conv['changing_order']:
                    if not conv['awaiting_changes']:
                        response = handle_change_order(user_id, user_message, page_id)
                        conv['awaiting_changes'] = True
                    else:
                        response = process_changes(user_id, user_message, page_id)
                        conv['changing_order'] = False
                        conv['awaiting_changes'] = False
                elif conv['canceling_order']:
                    response = handle_cancel_order(user_id, user_message, page_id)
                    conv['canceling_order'] = False
                    extracted_history_info["order_data"] = {}
                elif conv['track_order']:
                    order_id = extract_order_id(user_message)
                    response = get_tracking_status(order_id)
                    conv['track_order'] = False
                elif intent == "شراء":
                    response = extract_order_info(user_message, product_info, extracted_history_info)
                    conv['order_started'] = True
                elif intent == "تعديل_الطلب":
                    conv['changing_order'] = True
                    response = "من فضلك قم بارسال رقم الطلب الذي تريد تعديله (مثال: ORD-1234ABCD)"
                elif intent == "إلغاء":
                    response = "من فضلك قم بارسال رقم الطلب الذي تريد الغائه (مثال: ORD-1234ABCD)"
                    conv['canceling_order'] = True
                    extracted_history_info["order_data"] = {}
                elif intent == "تتبع_الطلب":
                    response = "من فضلك زودني برقم الطلب وسأتحقق لك فوراً 🔍"
                    conv['track_order'] = True
                elif intent == "بدء_جديد":
                    conversations[user_id] = {
                        "history": [],
                        "extracted_history_info": {},
                        "order_started": False,
                        "changing_order": False,
                        "canceling_order": False,
                        "awaiting_changes": False,
                        "track_order": False,
                        "page_id": page_id
                    }
                    response = "تم بدء طلب جديد. من فضلك أخبرني بما ترغب في شرائه 😊"
                else:
                    predefined_response = get_predefined_response(page_id, user_message)
                    if predefined_response:
                        response = predefined_response
                    else:
                        response = generate_ai_response(
                            product_info,
                            user_message,
                            conversations[user_id]["history"]
                        )

                # Save the conversation history
                conv["history"].append({
                    "role": "user",
                    "content": user_message
                })
                conv["history"].append({
                    "role": "assistant",
                    "content": response
                })

                # Send the message
                send_messenger_message(page_id, user_id, response)

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

@orders_api.route('/api/orders')
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