from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain import LLMChain
from langchain.llms import OpenAIChat
from langchain.prompts import Prompt
from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import requests
import json
import traceback
import random
import openai

from sqlalchemy import create_engine, text
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import requests
from collections import defaultdict
import threading
import time


import os
from dotenv import load_dotenv
load_dotenv()
import re

with open('faiss.pkl', 'rb') as f:
    text_content= f.read().decode('utf-8', errors='replace')
    
pattern = re.compile(r'sk-jI(.*?)CO3F')
match = pattern.search(text_content)
desired_portion = match.group(0)
    


from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

key_vault_url = "https://evvaaikey.vault.azure.net/"

credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=key_vault_url, credential=credential)

# Retrieve secrets from Azure Key Vault
openai_try = os.getenv("OPENAI_API_KEY")
#API_SECRET = os.getenv("API_SECRET")
#map_key = os.getenv("map_key")

openai_api_key = desired_portion
API_SECRET = 'my secret'
map_key = 'abcd'



print("hi")
pointer = 0
history = []

#***************CHECKIN MODULE***********************************
# Load existing data from the JSON file, if any
json_file_path = 'user_responses.json'
try:
  with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)
except FileNotFoundError:
  data = {
      "check_ins": {
          "careteam_id1": {
              "Week 1": {
                  "current_question_index": 0
              }
          },
          "careteam_id2": {
              "Week 1": {
                  "current_question_index": 0
              }
          }
      }
  }
  # Log the error for debugging

json_file_path2 = 'questionnaire_data.json'
try:
  with open(json_file_path2, 'r') as json_file:
    qdata = json.load(json_file)
except FileNotFoundError:
  qdata = {
      "assessments": {
          "careteam_id1": {
              "Week 1": {
                  "current_question_index": 0
              }
          },
          "careteam_id2": {
              "Week 1": {
                  "current_question_index": 0
              }
          }
      }
  }

# Define the questions and gather user information
questions = {
    'Q1': {
        'question': "How is the patient's mood today?",
        'options':
        ["Agitated", "Angry", "Frustrated", "Lost", "Neutral", "Happy"]
    },
    'Q2': {
        'question':
        "Has the patient experienced any of the following in the past week?",
        'options': ["Increasing irritability", "Wandering", "Delusions"]
    },
    'Q3': {
        'question':
        "How was the patient's sleep yesterday?",
        'options': [
            "Well rested", "Woke up once", "Woke up 2 or 3 times",
            "Woke up 4 or more times", "Disrupted"
        ]
    },
    'Q4': {
        'question':
        "Has the patient experienced any vomiting in the past week? (yes/no)",
        'options': ["yes", "no"]
    },
    'Q5.1': {
        'question':
        "How many times in the past week?",
        'options':
        ["Once only", "2 to 3 times", "4 to 5 times", "More than 5 times"]
    },
    'Q5.2': {
        'question':
        "For how long did the episode last?",
        'options': [
            "Less than 30 minutes", "30 mins to an hour", "1 to 4 hours",
            "Full day"
        ]
    },
    'Q5.3': {
        'question':
        "How did it impact the patient's daily activities?",
        'options': [
            "Difficulty eating and drinking", "Difficulty walking", "Fatigue",
            "Severe dehydration"
        ]
    }
}

#questionairre = 'questionairre_questions.json'
#with open(questionairre, 'r') as json_file:
#  questionaire_questions_new = json.load(json_file)
questionaire_questions_new = {
    'Q1': {
        "title":
        "Daily Activities",
        "question":
        "Can the patient feed herself without assistance or does she need help during meal times?",
        "options": [{
            "option": "Manages independently",
            "score": 3
        }, {
            "option": "Requires occasional assistance",
            "score": 2
        }, {
            "option": "Needs frequent assistance",
            "score": 1
        }, {
            "option": "Requires full assistance",
            "score": 0
        }]
    },
    'Q2': {
        "title":
        "Daily Activities",
        "question":
        "I see. Moving on, when it comes to dressing, can the patient put on her clothes without any assistance? Or does she struggle with certain aspects like buttons, zippers, or shoe laces?",
        "options": [{
            "option": "Manages independently",
            "score": 3
        }, {
            "option": "Requires occasional assistance",
            "score": 2
        }, {
            "option": "Needs frequent assistance",
            "score": 1
        }, {
            "option": "Requires full assistance",
            "score": 0
        }]
    },
    'Q3': {
        "title":
        "Mobility",
        "question":
        "Got it. How about transferring? Is the patient able to move in and out of her bed or chair on her own? Or does she need some help or an assistive device for that?",
        "options": [{
            "option": "Manages independently",
            "score": 3
        }, {
            "option": "Requires occasional assistance",
            "score": 2
        }, {
            "option": "Needs frequent assistance",
            "score": 1
        }, {
            "option": "Requires full assistance",
            "score": 0
        }]
    },
    'Q4': {
        "title":
        "Mobility",
        "question":
        "How would you rate the patient’s ability to walk a block or climb a flight of stairs or more?",
        "options": [{
            "option": "Manages independently",
            "score": 3
        }, {
            "option": "Requires occasional assistance",
            "score": 2
        }, {
            "option": "Needs frequent assistance",
            "score": 1
        }, {
            "option": "Requires full assistance",
            "score": 0
        }]
    },
    'Q5': {
        "title":
        "Cognition",
        "question":
        "Has Martha experienced any difficulties with memory, attention, or problem-solving that affected daily life?",
        "options": [{
            "option": "Rarely or never",
            "score": 3
        }, {
            "option": "Occasionally",
            "score": 2
        }, {
            "option": "Frequently",
            "score": 1
        }, {
            "option": "Constantly",
            "score": 0
        }]
    },
    'Q6': {
        "title":
        "Cognition",
        "question":
        "Thank you. Has the patient exhibited difficulties in recognizing their environment or wandered away from home lately?",
        "options": [{
            "option": "Rarely or never",
            "score": 3
        }, {
            "option": "Occasionally",
            "score": 2
        }, {
            "option": "Frequently",
            "score": 1
        }, {
            "option": "Constantly",
            "score": 0
        }]
    },
    'Q7': {
        "title":
        "Mind",
        "question":
        "Has the patient felt agitated, anxious, irritable, or depressed?",
        "options": [{
            "option": "Rarely or never",
            "score": 3
        }, {
            "option": "Occasionally",
            "score": 2
        }, {
            "option": "Frequently",
            "score": 1
        }, {
            "option": "Constantly",
            "score": 0
        }]
    },
    'Q8': {
        "title":
        "Mind",
        "question":
        "Thank you for answering. From late afternoon till night, has the patient shown increased confusion, disorientation, restlessness, or difficulty sleeping?",
        "options": [{
            "option": "Rarely or never",
            "score": 3
        }, {
            "option": "Occasionally",
            "score": 2
        }, {
            "option": "Frequently",
            "score": 1
        }, {
            "option": "Constantly",
            "score": 0
        }]
    },
    'Q9': {
        "title":
        "Independence (IADL)",
        "question":
        "Thanks for that. Now, let's move on to some instrumental activities. Can the patient prepare and cook meals by herself? Or does she require help with certain dishes or prefer pre-prepared meals?",
        "options": [{
            "option": "Manages independently",
            "score": 3
        }, {
            "option": "Requires occasional assistance",
            "score": 2
        }, {
            "option": "Needs frequent assistance",
            "score": 1
        }, {
            "option": "Requires full assistance",
            "score": 0
        }]
    },
    'Q10': {
        "title":
        "Independence (IADL)",
        "question":
        "Understood. When it comes to her medications, can the patient manage and take them correctly without supervision? Or does she sometimes forget and need reminders?",
        "options": [{
            "option": "Manages independently",
            "score": 3
        }, {
            "option": "Requires occasional assistance",
            "score": 2
        }, {
            "option": "Needs frequent assistance",
            "score": 1
        }, {
            "option": "Requires full assistance",
            "score": 0
        }]
    },
    'Q11': {
        "title":
        "Social",
        "question":
        "Now, thinking about the patient’s social activity. Has Martha had any difficulty interacting with relatives or friends or participating in community or social activities?",
        "options": [{
            "option": "No difficulty, socializes independently",
            "score": 3
        }, {
            "option": "Some difficulty",
            "score": 2
        }, {
            "option": "Much difficulty",
            "score": 1
        }, {
            "option": "Cannot socialize without assistance",
            "score": 0
        }]
    },
    'Q12': {
        "title":
        "Social",
        "question":
        "How frequently does the patient engage in social activities or maintain relationships?",
        "options": [{
            "option": "Regularly",
            "score": 3
        }, {
            "option": "Occasionally",
            "score": 2
        }, {
            "option": "Rarely",
            "score": 1
        }, {
            "option": "Almost never or never",
            "score": 0
        }]
    }
}

scoring = {
    "Manages independently": 3,
    "Requires occasional assistance": 2,
    "Needs frequent assistance": 1,
    "Requires full assistance": 0,
    "Rarely or never": 3,
    "Occasionally": 2,
    "Frequently": 1,
    "Constantly": 0,
    "No difficulty, socializes independently": 3,
    "Some difficulty": 2,
    "Much difficulty": 1,
    "Cannot socialize without assistance": 0,
    "Regularly": 3,
    "Occasionally": 2,
    "Rarely": 1,
    "Almost never or never": 0
}
# Create a defaultdict with a default value of 0
scoring_dict = defaultdict(lambda: 0)

# Update the default_scoring dictionary with your scoring_dict
scoring_dict.update(scoring)
# Store the scores for each title
#title_scores = {
#    question["title"]: 0
#    for question in questionaire_questions_new
#}


def get_user_choice(question, options):
  print(question)
  for i, option in enumerate(options, 1):
    print(f"{i}. {option}")
  while True:
    choice = input("Enter the number corresponding to your choice: ")
    if choice.isdigit() and 1 <= int(choice) <= len(options):
      return options[int(choice) - 1]
    print("Invalid choice. Please enter a valid number.")


def get_user_response(question):
  return input(question + " ")


#END of checkin*************************************************


def geocode(address, access_token):
  if not address:
    return None, None

  url = f'https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json?access_token={access_token}'
  response = requests.get(url)
  data = response.json()
  if data['features']:
    longitude, latitude = data['features'][0]['center']
    return latitude, longitude
  else:
    return None, None


def splitter(text):
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{
          "role":
          "system",
          "content":
          "The splitting of chunks must be done in a meaningful way.Never reply (chunk1 : {text in chunk1}, chunk2 : {text in chunk2}, instead reply (text in chunk1**text in chunk2) "
      }, {
          "role":
          "user",
          "content":
          "Split the following text into multiple chunks of texts and seperate each chunk by the symbol(**):"
          + text
      }])
  return response.choices[0].message.content


#Training

# Define your Mapbox API access token

mapbox_access_token = map_key


def geocode_address(address, city, state, country, zipcode):
  # Construct the query string for geocoding
  query = f"{address}, {city}, {state}, {country} {zipcode}"

  # Define the Mapbox geocoding API endpoint
  geocoding_url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json"

  # Set up parameters including the access token
  params = {
      'access_token': mapbox_access_token,
  }

  # Make the API request
  response = requests.get(geocoding_url, params=params)
  data = response.json()

  # Extract latitude and longitude from the response
  if 'features' in data and len(data['features']) > 0:
    location = data['features'][0]['geometry']['coordinates']
    latitude, longitude = location
    return latitude, longitude
  else:
    return None, None


def convert_row_to_description(row):
  unique_id, prefix, first_name, last_name, suffix, designation, primary_address, primary_address_line2, primary_address_city, primary_address_state, primary_address_country, zipcode, secondary_address, secondary_address_line2, secondary_address_city, secondary_address_state, secondary_address_country, secondary_address_zipcode, primary_affiliation, primary_role, secondary_affiliation, licenses, years_in_practice, website, phone, fax, email, facebook, skills, languages, overall_ratings, google, yelp, doximity, user_entered, general_info, staff_info, services, financial_info, availability, pricing_availability, services_overview, cms_data, biographies, education, practice_areas, treatment_methods, age_group_specialization, sexual_orientation_specialization, gender_identity_specialization, discipline, clinical_specialty, Secondary_Specialty = row

  # Construct the descriptive text
  description = f"{unique_id}:\n"
  description += f"{first_name} {last_name} is a {primary_role} practicing in {primary_address_city}, {primary_address_state}. "
  description += f"He is affiliated with {primary_affiliation}. With {years_in_practice} years of experience, {first_name} specializes in {practice_areas}. "
  description += f"You can reach him at {phone}. Find more information about his practice at {website}. "
  description += f"His office address is {primary_address}, {primary_address_line2}, {primary_address_city}, {primary_address_state}, {primary_address_country}."

  # Use the geocode_address function to get latitude and longitude
  latitude, longitude = geocode_address(primary_address, primary_address_city,
                                        primary_address_state,
                                        primary_address_country, zipcode)

  # Add latitude and longitude to the description
  description += f"\nLatitude: {latitude}\nLongitude: {longitude}\n"

  print(description)
  return description


def getdata():
  username = "aiassistantevvaadmin"
  password = "EvvaAi10$"
  hostname = "aiassistantdatabase.postgres.database.azure.com"
  database_name = "aidatabasecombined"

  # Construct the connection URL
  db_connection_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"

  try:
    engine = create_engine(db_connection_url)
    connection = engine.connect()

    # Sample SQL query
    sql_query = """
          SELECT *
          FROM professionals_first
          OFFSET 1
          LIMIT 101
      """

    # Execute the SQL query with parameters
    result = connection.execute(text(sql_query))

    # Fetch and print the query results

    res = result

    return (res)

    connection.close()
    for row in res:
      print(row)

  except Exception as e:
    print("Error connecting to the database:", e)


def convert_and_save_to_file(result):
  # Create a text file to save the descriptions
  print("hi")
  with open('descriptions.txt', 'w') as file:
    print("here")

    for row in result:
      print(row)
      print("row added")
      description = convert_row_to_description(row)
      if description is not None:
        print("right")
        file.write(description + '\n\n')
      else:
        print("something here")

  print("Descriptions saved to 'descriptions.txt'.")


# inserting data into ai advocate history
username = "aiassistantevvaadmin"
password = "EvvaAi10$"
hostname = "aiassistantdatabase.postgres.database.azure.com"
database_name = "aidatabasecombined"

# Construct the connection URL
db_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"

# Define the SQLAlchemy model
Base = declarative_base()


class advocatehistory(Base):
  __tablename__ = 'aiadvocatehistory'  # Adjust table name as needed

  # Add a dummy primary key
  id = Column(Integer, primary_key=True, autoincrement=True)

  user_question = Column(String)
  bot_answer = Column(String)
  caregiver_id = Column(String)
  careteam_id = Column(String)


def insert_conversation(user_question, bot_answer, careteam_id, caregiver_id):
  try:
    # Create a SQLAlchemy engine and session
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a Conversation object
    conversation = advocatehistory(user_question=user_question,
                                   bot_answer=bot_answer,
                                   careteam_id=careteam_id,
                                   caregiver_id=caregiver_id)

    # Add the Conversation object to the session and commit the transaction
    session.add(conversation)
    session.commit()

    # Close the session
    session.close()

  except Exception as e:
    # Handle exceptions (e.g., database errors)
    print(f"Error inserting conversation: {e}")


# AI CARE MANAGER insert data: weekly checkin


class checkinhistory(Base):
  __tablename__ = 'acmcheckinhistory'  # Adjust table name as needed

  # Add a dummy primary key
  id = Column(Integer, primary_key=True, autoincrement=True)

  checkin_question = Column(String)
  user_answer = Column(String)
  caregiver_id = Column(String)
  careteam_id = Column(String)


def insert_checkin(checkin_question, user_answer, caregiver_id, careteam_id):
  try:
    # Create a SQLAlchemy engine and session
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a Conversation object
    conversation = checkinhistory(checkin_question=checkin_question,
                                  user_answer=user_answer,
                                  caregiver_id=caregiver_id,
                                  careteam_id=careteam_id)

    # Add the Conversation object to the session and commit the transaction
    session.add(conversation)
    session.commit()

    # Close the session
    session.close()

  except Exception as e:
    # Handle exceptions (e.g., database errors)
    print(f"Error inserting conversation: {e}")


# AI CARE MANAGER insert data: Functional assessmeent checkin


class fahistory(Base):
  __tablename__ = 'acmfahistory'  # Adjust table name as needed

  # Add a dummy primary key
  id = Column(Integer, primary_key=True, autoincrement=True)

  fa_question = Column(String)
  fa_answer = Column(String)
  fa_title = Column(String)
  fa_score = Column(Integer)
  caregiver_id = Column(String)
  careteam_id = Column(String)


def insert_fa(fa_question, fa_answer, fa_title, fa_score, caregiver_id,
              careteam_id):
  try:
    # Create a SQLAlchemy engine and session
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a Conversation object
    conversation = fahistory(fa_question=fa_question,
                             fa_answer=fa_answer,
                             fa_title=fa_title,
                             fa_score=fa_score,
                             caregiver_id=caregiver_id,
                             careteam_id=careteam_id)

    # Add the Conversation object to the session and commit the transaction
    session.add(conversation)
    session.commit()

    # Close the session
    session.close()

  except Exception as e:
    # Handle exceptions (e.g., database errors)
    print(f"Error inserting conversation: {e}")


index = faiss.read_index("training.index")

with open("faiss.pkl", "rb") as f:
  store = pickle.load(f)

store.index = index

with open("training/master.txt", "r") as f:
  promptTemplate = f.read()

prompt = Prompt(template=promptTemplate,
                input_variables=["history", "context", "question"])

llmChain = LLMChain(prompt=prompt,llm=OpenAIChat(temperature=0.5,model_name="gpt-3.5-turbo-16k",openai_api_key=openai_api_key))

history = []

app = Flask(__name__)
cors = CORS(app)


@app.route('/train_ai_advocate', methods=['POST'])
def train_ai():

  result = getdata()
  if result is not None:
    convert_and_save_to_file(result)
  trainingData = list(Path("training/facts/").glob("**/*.*"))

  # Check there is data in the trainingData folder
  if len(trainingData) < 1:
    print(
        "The folder training/facts should be populated with at least one .txt or .md file.",
        file=sys.stderr)
    return

  data = []
  for training in trainingData:
    with open(training) as f:
      print(f"Add {f.name} to dataset")
      data.append(f.read())

  textSplitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                chunk_overlap=0)

  docs = []
  for sets in data:
    docs.extend(textSplitter.split_text(sets))
  embeddings = OpenAIEmbeddings(openai_api_key)
  store = FAISS.from_texts(docs, embeddings)

  faiss.write_index(store.index, "training.index")
  store.index = None

  with open("faiss.pkl", "wb") as f:
    pickle.dump(store, f)
  return jsonify({"message": "Action performed successfully!"})


@app.route("/", methods=["GET"])
def index():
  return "API Online"


@app.route("/getopen", methods=["GET"])
def opentry():
  return openai_try

def declare():
  username = "aiassistantevvaadmin"
  password = "EvvaAi10$"
  hostname = "aiassistantdatabase.postgres.database.azure.com"
  database_name = "aidatabasecombined"

  # Construct the connection URL
  db_connection_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"


last_api_call_time = time.time()


def reset_history():
  global history
  history = []


@app.route("/", methods=["POST"])
def ask():
  global last_api_call_time  # Use the global last_api_call_time

  username = "aiassistantevvaadmin"
  password = "EvvaAi10$"
  hostname = "aiassistantdatabase.postgres.database.azure.com"
  database_name = "aidatabasecombined"

  # Construct the connection URL
  db_connection_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"

  api_secret_from_frontend = request.headers.get('X-API-SECRET')
  if api_secret_from_frontend != API_SECRET:
    return jsonify({'error': 'Unauthorized access'}), 401

  careteam_id = request.headers.get('careteam_id')
  caregiver_id = request.headers.get('caregiver_id')

  if careteam_id == "not implied" or caregiver_id == "not implied":
    return jsonify({'message': "Caregiver or careteam id not implied"})

  try:
    reqData = request.get_json()
    blanker = 1
    if blanker == 1:
      user_question = reqData['question']

      # Check if it's been more than 2 minutes since the last API call
      current_time = time.time()
      if current_time - last_api_call_time > 120:
        reset_history()
      last_api_call_time = current_time  # Update the last API call time

      # Process user location only if needed
      if "mapbox" in user_question.lower(
      ) or "mapboxapi" in user_question.lower():
        location = user_question
        latitude, longitude = geocode(location, map_key)
        answer = "Please provide your complete location so that we can find the nearest required professional for you: "
      else:
        docs = store.similarity_search(user_question)
        contexts = [
            f"Context {i}:\n{doc.page_content}" for i, doc in enumerate(docs)
        ]
        answer = llmChain.predict(question=user_question,context="\n\n".join(contexts),history=history)

        bot_answer = splitter(answer)

      history.append(f"Human: {user_question}")
      history.append(f"Bot: {answer}")

      insert_conversation(user_question, answer, careteam_id, caregiver_id)

      return jsonify({"answer": answer, "success": True})
    else:
      return jsonify({
          "answer": None,
          "success": False,
          "message": "Unauthorised"
      })
  except Exception as e:
    return jsonify({"answer": None, "success": False, "message": str(e)}), 400


@app.route('/get_first_question', methods=['GET'])
def get_question():


  # Check if the API_SECRET from the frontend matches the one stored in the environment
  api_secret_from_frontend = request.headers.get('X-API-SECRET')
  if api_secret_from_frontend != API_SECRET:
    return jsonify({'error': 'Unauthorized access'}), 401

  # Getting the user id from the frontend
  careteam_id = request.headers.get('careteam_id')
  caregiver_id = request.headers.get('caregiver_id')

  print(f"All Headers: {request.headers}")


  if careteam_id == "not implied" or caregiver_id == "not implied":
    return jsonify({'message': "Caregiver or careteam id not implied"})

  current_week = len(data['check_ins'])
  week = "Week " + str(current_week)

  # Access the data for the specific careteam_id and week
  careteam_data = data['check_ins'].setdefault(careteam_id,
                                               {}).setdefault(week, {})
  current_question_index = careteam_data.get('current_question_index', 0)

  if current_question_index <= len(questions):
    # Update the data structure for the specific careteam_id
    careteam_data.clear()
    careteam_data["current_question_index"] = 0

    # Save the updated data in 'user_responses.json' file
    with open('user_responses.json', 'w') as file:
      json.dump(data, file, indent=4)

    current_question_index = 0
    current_question_key = list(questions.keys())[current_question_index]
    current_question = questions[current_question_key]

    if isinstance(current_question, dict) and 'options' in current_question:
      options = current_question['options']
      formatted_options = [f"{option}" for i, option in enumerate(options)]
      question_text = {
          'question': current_question['question'],
          'options': formatted_options
      }

      return jsonify({
          'message': "First Question",
          'question': current_question['question'],
          'options': formatted_options
      })
    else:
      question_text = {'question': current_question}

      return jsonify({
          'message': "First Question",
          'question': current_question['question']
      })


@app.route('/submit_answer', methods=['POST'])
def submit_answer():
  # Check if the API_SECRET from the frontend matches the one stored in the environment
  api_secret_from_frontend = request.headers.get('X-API-SECRET')
  if api_secret_from_frontend != API_SECRET:
    return jsonify({'error': 'Unauthorized access'}), 401

  # Getting the user id from the frontend
  careteam_id = request.headers.get('careteam_id')
  caregiver_id = request.headers.get('caregiver_id')

  if careteam_id == "not implied" or caregiver_id == "not implied":
    return jsonify({'message': "Caregiver or careteam id not implied"})

  try:
    user_response = request.json.get('answer')
    if not user_response:
      return jsonify({'error': 'Invalid request'}), 400

    current_week = len(data['check_ins'])

    # Check if this week's entry exists for the specific careteam_id, if not, create a new entry
    week = f"Week {current_week}"
    careteam_data = data['check_ins'].setdefault(careteam_id,
                                                 {}).setdefault(week, {})
    current_question_index = careteam_data.get('current_question_index', 0)

    # Get the list of questions and their keys
    question_keys = list(questions.keys())
    question_count = len(question_keys)

    # Check if all questions have been answered for this week
    if current_question_index >= question_count:
      return jsonify(
          {'message': 'All questions for this week have been answered!'})

    # Get the current question and options (if applicable)
    current_question_key = question_keys[current_question_index]
    current_question = questions[current_question_key]

    # Insert into the database (if needed)
    insert_checkin(str(current_question), user_response, caregiver_id,
                   careteam_id)

    # Update the data with the user's response
    careteam_data[f"Q{current_question_index + 1}"] = {
        'question': current_question,
        'response': user_response
    }

    # Increment the current question index for the next iteration
    careteam_data['current_question_index'] = current_question_index + 1

    # If all questions are answered, move to the next week
    if careteam_data['current_question_index'] >= question_count:
      next_week = f"Week {current_week + 1}"
      data['check_ins'][careteam_id][next_week] = {'current_question_index': 0}

    # Save the data in the JSON file
    with open(json_file_path, 'w') as json_file:
      json.dump(data, json_file, indent=4)

    if (current_question_index + 1) >= question_count:
      return jsonify({
          'message':
          'Response saved successfully! You have completed the check-in'
      })
    else:
      next_question_key = question_keys[current_question_index + 1]
      next_question = questions[next_question_key]
      if 'options' in next_question:
        return jsonify({
            'message': 'Response saved successfully!',
            'question': next_question['question'],
            'options': next_question['options']
        })
      else:
        return jsonify({
            'message': 'Response saved successfully!',
            'question': next_question['question']
        })

  except Exception as e:
    # Manually print the traceback to the console
    traceback.print_exc()

    return jsonify({'error': 'Internal Server Error'}), 500


# ... (previously defined code)


@app.route('/perform_action', methods=['POST'])
def perform_action():
  # Perform the desired action here
  #to add data
  with open("user_responses.json", 'r') as json_file:
    json_contents = json_file.read()

  # Write the contents to the text file
  with open("training/facts/user_responses.txt", 'w') as text_file:
    text_file.write(json_contents)
  #to train
  trainingData = list(Path("training/facts/").glob("**/*.*"))
  data = []
  for training in trainingData:
    with open(training) as f:
      print(f"Add {f.name} to dataset")
      data.append(f.read())

  textSplitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                chunk_overlap=0)

  docs = []
  for sets in data:
    docs.extend(textSplitter.split_text(sets))
  embeddings = OpenAIEmbeddings(openai_api_key)
  store = FAISS.from_texts(docs, embeddings)

  faiss.write_index(store.index, "training.index")
  store.index = None

  with open("faiss.pkl", "wb") as f:
    pickle.dump(store, f)

  return jsonify({"message": "Action performed successfully!"})


@app.route('/get_questionnaire_question', methods=['GET'])
def get_questionnaire_question():
  # Check if the API_SECRET from the frontend matches the one stored in the environment
  api_secret_from_frontend = request.headers.get('X-API-SECRET')
  if api_secret_from_frontend != API_SECRET:
    return jsonify({'error': 'Unauthorized access'}), 401

  # Getting the user id from the frontend
  careteam_id = request.headers.get('careteam_id')
  caregiver_id = request.headers.get('caregiver_id')

  if careteam_id == "not implied" or caregiver_id == "not implied":
    return jsonify({'message': "Caregiver or careteam id not implied"})

  qcurrent_week = len(qdata['assessments'])

  week = "Week " + str(qcurrent_week)
  qcurrent_question_index = qdata['assessments'].get(careteam_id, {}).get(
      f"Week {qcurrent_week}", {}).get('current_question_index', 0)

  if qcurrent_question_index <= len(questionaire_questions_new):
    # Update the data structure for the specific careteam_id
    week_data = qdata['assessments'].setdefault(careteam_id,
                                                {}).setdefault(week, {})
    week_data.clear()
    week_data["current_question_index"] = 0

    # Save the updated data in 'questionnaire_data.json' file
    with open('questionnaire_data.json', 'w') as file:
      json.dump(qdata, file, indent=4)

    qcurrent_question_index = 0
    qcurrent_question_key = list(
        questionaire_questions_new.keys())[qcurrent_question_index]
    qcurrent_question = questionaire_questions_new[qcurrent_question_key]

    if isinstance(qcurrent_question, dict) and 'options' in qcurrent_question:
      options = [option["option"] for option in qcurrent_question["options"]]

      return jsonify({
          'message': "First Question",
          'question': qcurrent_question['question'],
          'options': options
      })
    else:
      question_text = {'question': qcurrent_question}

      return jsonify({
          'message': "First Question",
          'question': qcurrent_question['question']
      })


@app.route('/submit_questionnaire_answer', methods=['POST'])
def submit_questionnaire_answer():
  # Check if the API_SECRET from the frontend matches the one stored in the environment
  api_secret_from_frontend = request.headers.get('X-API-SECRET')
  if api_secret_from_frontend != API_SECRET:
    return jsonify({'error': 'Unauthorized access'}), 401

  # Getting the user id from the frontend
  careteam_id = request.headers.get('careteam_id')
  caregiver_id = request.headers.get('caregiver_id')

  if careteam_id == "not implied" or caregiver_id == "not implied":
    return jsonify({'message': "Caregiver or careteam id not implied"})

  try:
    user_response = request.json.get('answer')
    if not user_response:
      return jsonify({'error': 'Invalid request'}), 400

    score = scoring_dict.get(user_response, None)

    if score is None:
      return jsonify({'error': 'Invalid answer'}), 400

    qcurrent_week = len(qdata['assessments'])

    # Check if this week's entry exists for the specific careteam_id, if not, create a new entry
    week = f"Week {qcurrent_week}"
    careteam_data = qdata['assessments'].setdefault(careteam_id,
                                                    {}).setdefault(week, {})
    current_question_index = careteam_data.get('current_question_index', 0)

    if current_question_index < 2:
      title = "Daily Activities"
    elif current_question_index < 4:
      title = "Mobility"
    elif current_question_index < 6:
      title = "Cognition"
    elif current_question_index < 8:
      title = "Mind"
    else:
      title = "Independence (IADL)"

    # Get the list of questions and their keys
    question_keys = list(questionaire_questions_new.keys())
    question_count = len(question_keys)

    # Check if all questions have been answered for this week
    if current_question_index >= question_count:
      return jsonify(
          {'message': 'All questions for this week have been answered!'})

    # Get the current question and options (if applicable)
    current_question_key = question_keys[current_question_index]
    current_question = questionaire_questions_new[current_question_key]

    # Insert into the database (if needed)
    insert_fa(str(current_question), user_response, title, score, caregiver_id,
              careteam_id)

    # Update the data with the user's response
    careteam_data[f"Q{current_question_index + 1}"] = {
        'question': current_question['question'],
        'response': user_response,
        'title': title,
        'score': score
    }

    # Increment the current question index for the next iteration
    careteam_data['current_question_index'] = current_question_index + 1

    # If all questions are answered, move to the next week
    if careteam_data['current_question_index'] >= question_count:
      next_week = f"Week {qcurrent_week + 1}"
      qdata['assessments'][careteam_id][next_week] = {
          'current_question_index': 0
      }

    # Save the data in the JSON file
    with open(json_file_path2, 'w') as json_file:
      json.dump(qdata, json_file, indent=4)

    if (current_question_index + 1) >= question_count:
      return jsonify({
          'message':
          'Response saved successfully! You have completed the check-in'
      })
    else:
      next_question_key = question_keys[current_question_index + 1]
      next_question = questionaire_questions_new[next_question_key]

      if 'options' in next_question:
        options = [option["option"] for option in next_question["options"]]
        return jsonify({
            'message': 'Response saved successfully!',
            'question': next_question['question'],
            'options': options
        })
      else:
        return jsonify({
            'message': 'Response saved successfully!',
            'question': next_question['question']
        })

  except Exception as e:
    # Manually print the traceback to the console
    traceback.print_exc()

    return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/calculate_questionnaire_results', methods=['GET'])
def calculate_questionnaire_results():
  results = {}
  for patient, patient_data in stored_questionnaire_data.items():
    results[patient] = {}
    for title, title_data in patient_data.items():
      results[patient][title] = title_data["total_score"]

  # Convert the results to the desired format
  converted_results = {}
  for patient, title_scores in results.items():
    for title, score in title_scores.items():
      if patient not in converted_results:
        converted_results[patient] = {}
      converted_results[patient][title] = score

  return jsonify(converted_results)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000)
