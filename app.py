
# Temporary storage for Page Access Tokens (Replace with your tokens)
PAGE_TOKENS = {
    "641855032337466": "AAORSK8XIqcBO5DrfZBP9OGWoQWZAZAfrL6ZAp4RiLx7dNRRRoyZB5kZC7k8QZB9C2jfjt8ZCGgaSpjwqA0AbseJgZAKrB3D1SXIeCuobl9ZCxs1FIYuElTV6Y0d7Qpt7G6r0anzxMxZCY4ddiZBjI1ZBDwyaL1AaoS7ZAfcKrXdkrl7ZCk2uRiVJAX4IIzZCDswcEztzCQX5QZDZD",
    "388698940987084": "AAORSK8XIqcBO5xezeEdYOYgZBxYR09KR1xWtQF2HUCflN0akYHs0u0zTXGEp77VJA4tZBi0fbWkX9bHAGiCStLjZB2h4lecn3yBIDGBjTZAu2qyzCKyyI2jHCoG0UkAfHmFHWwZAcVOi9ZCuIZCsZBmZAyc7ZBjQKdKC7DyHScpM9JAZCHw7ZAjM1Rxbjk5n5rwHHzpFgZDZD"
}

import requests

DEEPSEEK_API_KEY = "sk-5f8aaed2e3904f7d853c5f4dba977c1"  # Use environment variables in production
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


def generate_ai_response(product_info, user_message, conversation_history=None):
    """
    النسخة المحدثة:
    1. إزالة العروض الترويجية
    2. جمع بيانات الطلب بشكل منظم
    3. إضافة سياسة الاستبدال والشحن
    4. دعم المقاسات والألوان
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    # تحليل نية العميل
    is_buying_intent = any(word in user_message.lower() for word in ["اشتري", "شراء", "طلب", "حجز"])
    is_size_question = any(word in user_message.lower() for word in ["مقاس", "حجم"])
    is_color_question = any(word in user_message.lower() for word in ["لون", "ألوان"])

    # بناء سياق المحادثة
    context = ""
    if conversation_history:
        context = "\n".join([f"{'العميل' if msg['role'] == 'user' else 'البوت'}: {msg['content']}"
                             for msg in conversation_history[-2:]])

    # صياغة البرومبت الدقيق
    prompt = f"""
    أنت مساعد مبيعات لمتجر أزياء. 
    المنتج: {product_info['name']}
    السعر: {product_info['base_price']} جنيه
    الألوان: {', '.join(product_info.get('colors', []))}
    المقاسات: {', '.join(product_info.get('sizes', []))}

    رسالة العميل: "{user_message}"
    السياق: {context or 'لا يوجد سياق سابق'}

    التعليمات:
    1. قدم معلومات دقيقة بطريق ممتعة تجذب الانتباه 
    2. لا تستخدم عروض ترويجية
    3. للاستفسارات عن المقاسات/الألوان: أجب مباشرةبطريقة سلسةو قصيرة 
    ه4. عند طلب الشراء: أرسل تعليمات الطلب الخاصة ب اجابة العميل فقط
    5. استخدم لغة واضحة ومباشرة وتفتح نفس العميل
    6. الردود قصيرة (سطر أو سطرين)
    """

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1,
        "max_tokens": 100,
        "language": "ar"
    }

    try:
        response = requests.post(
            DEEPSEEK_URL,
            headers=headers,
            json=payload,
            timeout=8
        )
        response.raise_for_status()

        ai_message = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')

        if is_buying_intent:
            return generate_order_instructions(product_info)

        if not ai_message:
            return "عذرًا، لم أتمكن من معالجة طلبك."

        return ai_message.strip()

    except Exception as e:
        print(f"خطأ: {str(e)}")
        return generate_fallback_response(product_info)


def generate_order_instructions(product_info):
    """إنشاء تعليمات الطلب المطلوبة"""
    return f"""
    📋 *تعليمات الطلب* 📋

    _لتسجيل الأوردر يرجي ارسال البيانات بالترتيب:_
    1. الأسم بالكامل
    2. رقم للتليفون
    3. العنوان تفصيليا (المحافظة - المدينة - الحي)
    4. العدد المطلوب
    5. اللون والمقاس
    6. الوزن (كجم)

    📦 *سياسة الشحن*:
    - مصاريف الشحن: 30 جنيه مقدمًا
    - القاهرة والجيزة: توصيل خلال يومين
    - المحافظات: توصيل خلال 3 أيام

    🔄 *سياسة الاستبدال*:
    - رسوم استبدال: 30 جنيه
    - يتم دفع الفرق عند طلب مقاس/لون مختلف
    - لا استبدال بعد 7 أيام من الاستلام

    {product_info['name']} | السعر: {product_info['base_price']} جنيه
    """


def generate_fallback_response(product_info):
    """رد بديل عند حدوث أخطاء"""
    return f"""
    {product_info['name']}
    السعر: {product_info['price']} جنيه
    للأستفسار عن المقاسات/الألوان راسلنا
    للطلب اكتب 'اشتري'
    """
from flask import Flask, request, jsonify
import uuid

app = Flask(__name__)

# In-memory storage (replace with database in production)
conversations = {}  # Format: {user_id: {"history": [], "order": None}}

@app.route('/webhook', methods=['POST'])
def handle_messages():
    data = request.json
    for entry in data.get('entry', []):
        page_id = entry['id']
        for event in entry.get('messaging', []):
            sender_id = event['sender']['id']
            user_message = event.get('message', {}).get('text', '').strip()

            # Initialize conversation if new user
            if sender_id not in conversations:
                conversations[sender_id] = {"history": [], "order": None}

            # Get product for this page (from database in production)
            product = get_product_by_page_id(page_id)

            # Generate AI response with context
            ai_response = generate_ai_response(
                product_info=product,
                user_message=user_message,
                conversation_history=conversations[sender_id]["history"]
            )

            # Check if user provided order details
            if is_ready_to_order(user_message, conversations[sender_id]["history"]):
                order_id = create_order(sender_id, product)
                conversations[sender_id]["order"] = order_id
                ai_response += f"\n\n✅ Order #{order_id} received! We'll ship your {product['name']} soon."

            # Store conversation
            conversations[sender_id]["history"].extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": ai_response}
            ])

            # Send response
            send_messenger_message(page_id, sender_id, ai_response)

    return jsonify({"status": "ok"}), 200


def is_ready_to_order(user_message, history):
    """Detects if user provided contact/shipping info"""
    triggers = ["buy", "order", "ship to", "my address is", "name is"]
    return any(trigger in user_message.lower() for trigger in triggers)

def create_order(user_id, product):
    """Creates an order in your system"""
    order_id = str(uuid.uuid4())[:8]  # Generate simple order ID
    # In production: Save to database with user details from conversation history
    print(f"📦 New order #{order_id} for {product['name']} by user {user_id}")
    return order_id


def get_product_by_page_id(page_id):
    products = {
        "388698940987084": {  # Page for Abayas
            "name": "عباية مطرزة",
            "base_price": 249.99,
            "description": "عباية سوداء مطرزة يدوياً بخيوط ذهبية",
            "benefits": [
                "قماش خفيف ومسامي",
                "تطريز يدوي فاخر",
                "تصلح للمناسبات"
            ],
            "category": "ملابس نسائية",
            "colors": ["أسود", "بيج", "ذهبي", "أحمر داكن"],
            "sizes": {
                "48": "مقاس صغير (48-50)",
                "50": "مقاس متوسط (50-52)",
                "52": "مقاس كبير (52-54)",
                "54": "مقاس كبير جداً (54-56)"
            },
            "bulk_pricing": {
                "2_pieces": 450,
                "3_pieces": 650,
                "5_pieces": 1000
            },
            "shipping_policy": {
                "base_fee": 30,
                "cairo_giza": 20,
                "other_governorates": 40,
                "upper_egypt": 50
            },
            "exchange_policy": "استبدال خلال 7 أيام بشرط عدم الاستخدام مع دفع 30 جنيه مصاريف شحن"
        },
        "641855032337466": {  # Page for T-Shirts
            "name": "تيشرت أوفر سايز قطن 100%",
            "base_price": 199.99,
            "description": "تيشرت قطني عالي الجودة بتصميم أوفر سايز",
            "benefits": [
                "قطن مصري 100%",
                "تصميم مريح",
                "متانة عالية ضد الغسيل"
            ],
            "category": "ملابس رجالية",
            "colors": ["أسود", "أبيض", "رمادي", "أزرق نيلي"],
            "sizes": {
                "XL": "50-65 كجم",
                "2XL": "65-75 كجم",
                "3XL": "75-90 كجم",
                "4XL": "90-115 كجم",
                "5XL": "115-130 كجم"
            },
            "bulk_pricing": {
                "2_pieces": 350,
                "3_pieces": 450,
                "4_pieces": 560,
                "5_pieces": 650,
                "special_2_pieces": 440,
                "special_3_pieces": 580,
                "special_4_pieces": 700,
                "special_5_pieces": 800
            },
            "shipping_policy": {
                "base_fee": 30,
                "cairo_giza": 20,
                "other_governorates": 40,
                "upper_egypt": 50
            },
            "exchange_policy": "استبدال خلال 5 أيام مع دفع الفروق إن وجدت"
        }

    }

    return products.get(page_id, {
        "name": "منتج غير معروف",
        "base_price": 0,
        "description": "",
        "benefits": [],
        "category": "عام"
    })



def send_messenger_message(page_id, user_id, text):
    """Sends message with optional quick replies"""
    page_token = PAGE_TOKENS[page_id]  # Your existing token lookup

    # Add quick reply buttons when appropriate
    if "order" in text.lower() or "buy" in text.lower():
        payload = {
            "recipient": {"id": user_id},
            "message": {
                "text": text,
                "quick_replies": [
                    {
                        "content_type": "text",
                        "title": "Yes, order now!",
                        "payload": "USER_CONFIRM_ORDER"
                    },
                    {
                        "content_type": "text",
                        "title": "More details",
                        "payload": "MORE_INFO"
                    }
                ]
            }
        }
    else:
        payload = {
            "recipient": {"id": user_id},
            "message": {"text": text}
        }

    requests.post(
        f"https://graph.facebook.com/v19.0/{page_id}/messages",
        params={"access_token": page_token},
        json=payload
    )

if __name__ == '__main__':
    app.run(port=5000, debug=True)
