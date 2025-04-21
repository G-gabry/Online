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





if __name__ == '__main__':
    # Start 4 worker processes
    # for _ in range(4):
    #     multiprocessing.Process(target=start_worker).start()

    app.run(port=5000)