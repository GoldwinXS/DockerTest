import pika
import json

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='celery_request')


def encode_data(data):
    return json.dumps(data)


def make_request():
    body = {'divide': (8, 2)}
    data = encode_data(body)
    channel.basic_publish(exchange='', routing_key='celery_request', body=data)

make_request()
