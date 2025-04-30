# Configuration and Imports
import requests
from flask import Flask, request, jsonify
import uuid
from difflib import SequenceMatcher
import requests
from flask import Flask, request, jsonify
import uuid
from difflib import SequenceMatcher
import redis
from rq import Queue
from rq.job import Job
import threading
import time
import multiprocessing
from rq.worker import Worker
from jops import orders_api

from jops import process_message
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import check_password_hash
import pyodbc

# Then insert that into the `password_hash` column

app = Flask(__name__)
# redis_conn = redis.Redis(host='localhost', port=6379, db=0)
# task_queue = Queue('facebook_messages', connection=redis_conn)
# Temporary storage (replace with database in production)
app.register_blueprint(orders_api)




  # Format: {user_id: {"history": [], "order": None, "page_id": "123"}}

# Product Database
import numpy as np


# Enhanced PRODUCT_DATABASE with vector embeddings
@app.route('/webhook', methods=['GET', 'POST'])
def handle_webhook():
    # Handle verification
    if request.method == 'GET':
        verify_token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')

        if verify_token == 'Gabry':
            return challenge, 200
        return "Verification failed", 403

    # Handle incoming messages
    elif request.method == 'POST':
        data = request.json

        # Validate basic message structure
        if not data or data.get('object') != 'page':
            return jsonify({"status": "invalid request"}), 400

        # Add message to processing queue
        # task_queue.enqueue(process_message, data)
        process_message(data)
        return jsonify({"status": "queued"}), 200


# def start_worker():
#     """Run a single worker process"""
#     worker = Worker(queues=[task_queue], connection=redis_conn)
#     worker.work()





app = Flask(__name__)
app.secret_key = 'your_secret_key'  # change this to a secure random string

# Database connection setup (adjust as needed)
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=DESKTOP-MINNO8Q;"
    "DATABASE=ordersDB;"
    "Trusted_Connection=yes;"  # or use UID/PWD if needed
)
conn = pyodbc.connect(conn_str)
@app.route('/')
def home():
    user_id = session.get('user_id')
    if user_id:
        return redirect(url_for('orders_api.get_orders_by_user',user_id=user_id))

    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor = conn.cursor()
        cursor.execute("SELECT id, password_hash FROM Users WHERE email = ?", (email,))
        user = cursor.fetchone()

        if user and password == user[1]:
            session['user_id'] = user[0]
            print(f"Login success for user #{user[0]}")
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password", "danger")

    return render_template('Login.html')
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    # Start 4 worker processes
    # for _ in range(4):
    #     multiprocessing.Process(target=start_worker).start()

    app.run(debug=True, port=5000)