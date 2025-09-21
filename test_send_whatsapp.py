# Download the helper library from https://www.twilio.com/docs/python/install

import os
from twilio.rest import Client
import json

from dotenv import load_dotenv

load_dotenv()

# Find your Account SID and Auth Token at twilio.com/console

# and set the environment variables. See http://twil.io/secure

account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)
message = client.messages.create(
    from_="whatsapp:+14155238886",
    to="whatsapp:+447447595352",
    content_sid="HXb5b62575e6e4ff6129ad7c8efe1f983e",
    content_variables=json.dumps({"1": "22 July 2026", "2": "3:15pm"}),
)
print(message.body)