


import requests
def detect_intent(user_message, conversation_history=None):
    DEEPSEEK_API_KEY = "sk-72aaab047f7d4eacb9124aea8dd997ec"
    DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    context = []
    if conversation_history:
        for msg in conversation_history[-4:]:
            speaker = "العميل" if msg['role'] == 'user' else "أنت"
            context.append(f"{speaker}: {msg['content']}")
        context = "\n".join(context)

    prompt = f"""
صنّف الرسالة التالية بناءً على المحادثات السابقة والنوايا المحددة. يجب أن يكون الرد كلمة واحدة فقط من القائمة التالية:

شراء
إلغاء
تتبع_الطلب
تعديل_الطلب
بدء_جديد
استفسار
غير_معروف

النوايا المتاحة:
1. "شراء" - عندما يعبر العميل عن رغبته في الشراء (مثال: "عايز أطلب"، "حابب أشتري")
2. "إلغاء" - رغبة في إلغاء الطلب (مثال: "عايز ألغي"، "غيرت رأيي")
3. "تتبع_الطلب" - استفسار عن حالة الشحن (مثال: "الطلب هيوصل امتى؟"، "عايز أعرف حالة الطلب")
4. "تعديل_الطلب" - طلب تغيير في الطلب الحالي (مثال: "عايز أغير اللون"، "نفسي في مقاس أصغر")
5. "بدء_جديد" - رغبة في بدء طلب جديد (مثال: "مسح اللي فات"، "ابتدى من جديد")
6. "استفسار" - أسئلة عامة عن المنتج (مثال: "فيها خصم؟"، "المقاسات موجودة؟")
7. "غير_معروف" - عندما لا تنتمي الرسالة لأي نية محددة

محادثات سابقة:
{context or 'لا يوجد محادثة سابقة'}

الرسالة الجديدة: "{user_message}"
التصنيف:"""

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Lower temperature for more deterministic responses
        "max_tokens": 10,    # Limit to prevent verbose responses
        "stop": ["\n"],     # Stop at newlines to prevent explanations
        "language": "ar"
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=35)
        response.raise_for_status()
        ai_message = response.json()['choices'][0]['message']['content'].strip()

        # Validate the response is one of our expected intents
        valid_intents = ["شراء", "إلغاء", "تتبع_الطلب", "تعديل_الطلب",
                        "بدء_جديد", "استفسار", "غير_معروف"]

        if ai_message in valid_intents:
          return ai_message
        else:
          return "غير_معروف"

    except Exception as e:
        print(f"API Error: {str(e)}")
        return "غير_معروف"

print(detect_intent("عايز قطعتين"))



