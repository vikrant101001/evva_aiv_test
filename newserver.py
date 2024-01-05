


from flask import Flask, request, jsonify
from geopy.geocoders import MapBox
from geopy.distance import geodesic
from pathlib import Path
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os
import json
from langchain import LLMChain
from langchain.llms import OpenAIChat
from langchain.prompts import Prompt
import time
from flask_cors import CORS

import re


import random


from sqlalchemy import create_engine, text
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


from collections import defaultdict
import threading




app = Flask(__name__)
cors = CORS(app)

with open('junk.pkl', 'rb') as f:
    text_content= f.read().decode('utf-8', errors='replace')
    
pattern = re.compile(r'sk-DX(.*?)T41S')
match = pattern.search(text_content)
desired_portion = match.group(0)
    

os.environ["OPENAI_API_KEY"] = desired_portion
openai_api_key = os.environ["OPENAI_API_KEY"]
mapbox_api_key = 'pk.eyJ1IjoiZXZ2YWhlYWx0aCIsImEiOiJjbGp5anJjY2IwNGlnM2RwYmtzNGR0aGduIn0.Nx4jv-saalq2sdw9qKuvbQ'
geocoder = MapBox(api_key=mapbox_api_key)

API_SECRET = 'my secret'

last_api_call_time = 0
history = []
llmChain = None


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
        "Can the patient feed themself without assistance or do they need help during meal times?",
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
        "I see. Moving on, when it comes to dressing, can the patient put on their clothes without any assistance? Or do they struggle with certain aspects like buttons, zippers, or shoe laces?",
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
        "Got it. How about transferring? Is the patient able to move in and out of their bed or chair on her own? Or do they need some help or an assistive device for that?",
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
        "Has the patient experienced any difficulties with memory, attention, or problem-solving that affected daily life?",
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
        "Thanks for that. Now, let's move on to some instrumental activities. Can the patient prepare and cook meals by themself? Or do they require help with certain dishes or prefer pre-prepared meals?",
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
        "Understood. When it comes to their medications, can the patient manage and take them correctly without supervision? Or do they sometimes forget and need reminders?",
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
        "Now, thinking about the patient’s social activity. Has the patient had any difficulty interacting with relatives or friends or participating in community or social activities?",
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



def reset_history():
    global history
    history = []


def get_coordinates(address):
    mapbox_api_key = "pk.eyJ1IjoiZXZ2YWhlYWx0aCIsImEiOiJjbGp5anJjY2IwNGlnM2RwYmtzNGR0aGduIn0.Nx4jv-saalq2sdw9qKuvbQ"
    geocoder = MapBox(api_key=mapbox_api_key)

    # Convert the address to the central zipcode
    location = geocoder.geocode(address)
    
    if location:
        latitude, longitude = location.latitude, location.longitude
        print(f"Confirmed:\nAddress: {address}\nCoordinates: {latitude}, {longitude}")
        return latitude, longitude
    else:
        raise ValueError("Could not retrieve location information from the address.")


def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).miles

def train(user_location):
    count = 0
    print(count)
    try:
        os.remove("faiss.pkl")
        os.remove("training.index")
        print("Removed existing faiss.pkl and training.index")
    except FileNotFoundError:
        pass 


    # Check there is data fetched from the database
    training_data_folders = list(Path("training/facts/").glob("**/latitude*,longitude*"))

    # Check there is data in the trainingData folder
    if len(training_data_folders) < 1:
        print("The folder training/facts should be populated with at least one subfolder.")
        return

    
    latitude, longitude = user_location
    user_coordinates = (latitude, longitude)

    data = []
    for folder in training_data_folders:
        folder_coordinates = folder.name.replace('latitude', '').replace('longitude', '').split(',')
        folder_latitude, folder_longitude = map(float, folder_coordinates)

        folder_coords = (folder_latitude, folder_longitude)
        distance = calculate_distance(user_coordinates, folder_coords)
        print(f" the distance between {user_coordinates} and {folder.name} is {distance}.")

        if distance < 100:
            count = count +1
            print(f"Added {folder.name}'s contents to training data.")
            for json_file in folder.glob("*.json"):
                with open(json_file) as f:
                    data.extend(json.load(f))
                    print(f"  Added {json_file.name} to training data.")
 
    if count == 0:
       print("No relevant data found within 50 miles.")
       print(count)
       return count
        

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

    docs = []
    for entry in data:
    	address = entry.get('address', '')
    	if address is not None:
    		print(f"Address to split: {address}")
    		docs.extend(text_splitter.split_text(address))

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    store = FAISS.from_texts(docs, embeddings)

    faiss.write_index(store.index, "training.index")
    store.index = None
    print(count)
    with open("faiss.pkl", "wb") as f:
        pickle.dump(store, f)
    
    return count

searched = 0
# ...
previous_response = ""
# ...

@app.route("/", methods=["GET"])
def index():
  return "API Online"

#last_api_call_time = time.time()

@app.route("/", methods=["POST"])
def ask():
    global previous_response
    global last_api_call_time
    global llmChain
    global searched
    global count1

    username = "aiassistantevvaadmin"
    password = "EvvaAi10$"
    hostname = "aiassistantdatabase.postgres.database.azure.com"
    database_name = "aidatabasecombined"
    db_connection_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"

    api_secret_from_frontend = request.headers.get('X-API-SECRET')
    if api_secret_from_frontend != API_SECRET:
        return jsonify({'error': 'Unauthorized access'}), 401

    careteam_id = request.headers.get('careteamid')
    caregiver_id = request.headers.get('userid')

    if careteam_id == "not implied" or caregiver_id == "not implied":
        return jsonify({'message': "Caregiver or careteam id not implied"})

    try:
        reqData = request.get_json()
        user_question = reqData['question']
        user_address = request.headers.get('userprimaddress')
        print(f"All Headers: {request.headers}")

        current_time = time.time()
        if current_time - last_api_call_time > 600:
            
            reset_history()

            # Only confirm address if the question is related to a search
            user_location = get_coordinates(user_address)
            count1 = train(user_location)  # Train based on user location for the first call of a session
            print(count1)
            last_api_call_time = current_time

        if not llmChain:
            # Initialize llmChain if it's not initialized yet
            with open("training/master.txt", "r") as f:
                promptTemplate = f.read()

            prompt = Prompt(template=promptTemplate, input_variables=["history", "context", "question"])
            llmChain = LLMChain(prompt=prompt, llm=OpenAIChat(temperature=0.5,
                                                             model_name="gpt-4-1106-preview",
                                                             openai_api_key=openai_api_key))
        # Only confirm the user's address for search-related questions
        search_keywords = ["need", "looking for", "search for", "find", "locate", "where is","want information about", "details about", "data on", "facts on", "tell me about","need assistance with", "help with", "support for", "information on","seeking", "inquiring about", "inquiring on", "searching for", "looking up"]
        
        if any(keyword in user_question.lower() for keyword in search_keywords):
                if searched == 0:
                     searched = searched + 1
                     print(searched)
                     confirm_message = f"Do you want me to search near\n{user_address}\n\nReply with 'yes' or 'no'."
                     previous_response = confirm_message
                     response = confirm_message
                     searched = searched + 1
                     print(searched)
                else:
                     response = llmChain.predict(question=user_question, context="\n\n".join(history), history=history)
        elif previous_response.startswith("Do you want me to search near") and "yes" in user_question.lower():
            # Continue with the user's provided address
            if count1 < 1:
                 response = "I am sorry! 🙁 I couldn’t find any suitable results within 100 miles. Evva is only available in limited geographies. Please contact Team Evva at info@evva360.com to learn more about when your region may be next. Would you like me to search near a different location?.. \n Please Reply with 'yes' or 'no'"
                 previous_response = "I am sorry! 🙁 I couldn’t find any suitable results within 100 miles. Evva is only available in limited geographies. Please contact Team Evva at info@evva360.com to learn more about when your region may be next."
            else:
                 response = llmChain.predict(question=user_question, context="\n\n".join(history), history=history)
        elif previous_response.startswith("Do you want me to search near") and "no" in user_question.lower():
            # Ask for a new location
            previous_response = "Please enter the new location where you want to search"
            response = previous_response
        elif previous_response.startswith("Do you want me to search near") and "yes" not in user_question.lower() and "no" not in user_question.lower():
            response = "Please include yes or no in your answer"

        elif previous_response.startswith("Please enter the new location where you want to search"):
            user_address = user_question
            user_location = get_coordinates(user_address)
            count2 = train(user_location)  # Train based on the new user location
            if count2 < 1:
                 response = "I am sorry! 🙁 I couldn’t find any suitable results within 100 miles. Evva is only available in limited geographies. Please contact Team Evva at info@evva360.com to learn more about when your region may be next. Would you like me to search near a different location?.. \n Please Reply with 'yes' or 'no'"
                 previous_response = "I am sorry! 🙁 I couldn’t find any suitable results within 100 miles. Evva is only available in limited geographies. Please contact Team Evva at info@evva360.com to learn more about when your region may be next."
            else:
                 previous_response = ""
                 response = llmChain.predict(question=user_question, context="\n\n".join(history), history=history)
        elif previous_response.startswith("I am sorry! 🙁 I couldn’t find ") and "no" in user_question.lower():
            response = llmChain.predict(question=user_question, context="\n\n".join(history), history=history)
            previous_response = ""
        elif previous_response.startswith("I am sorry! 🙁 I couldn’t find ") and "yes" in user_question.lower():
            previous_response = "Please enter the new location where you want to search"
            response = previous_response
        elif previous_response.startswith("I am sorry! 🙁 I couldn’t find ") and "yes" not in user_question.lower() and "no" not in user_question.lower():
            response = "Please include yes or no in your answer"
        else:
            # Continue with the user's question for non-search queries
            response = llmChain.predict(question=user_question, context="\n\n".join(history), history=history)

        history.append(f"Bot: {response}")
        history.append(f"Human: {user_question}")
        insert_conversation(user_question, response, careteam_id, caregiver_id)

        return jsonify({"answer": response, "success": True})
    except Exception as e:
        return jsonify({"answer": None, "success": False, "message": str(e)}), 400

@app.route('/get_first_question', methods=['GET'])
def get_question():


  # Check if the API_SECRET from the frontend matches the one stored in the environment
  api_secret_from_frontend = request.headers.get('X-API-SECRET')
  if api_secret_from_frontend != API_SECRET:
    return jsonify({'error': 'Unauthorized access'}), 401

  # Getting the user id from the frontend
  careteam_id = request.headers.get('careteam')
  caregiver_id = request.headers.get('caregiver')
  patient_name = request.headers.get('patient')

  print(f"All Headers: {request.headers}")
  print(patient_name)


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
      if "the patient" in current_question['question']:
      	modified_question = current_question['question'].replace("the patient", patient_name)
      else:
      	modified_question = current_question['question']
      question_text = {
          'question': modified_question ,
          'options': formatted_options
      }

      return jsonify({
          'message': "First Question",
          'question': modified_question,
          'options': formatted_options
      })
    else:
      if "the patient" in question['question']:
      	modified_question = current_question['question'].replace("the patient", patient_name)
      else:
      	modified_question = current_question['question']
      question_text = {'question': current_question}

      return jsonify({
          'message': "First Question",
          'question': modified_question
      })



def end_checkin(careteam_id):
    # Logic to handle the end of check-in for the given careteam_id
    current_week = len(data['check_ins'])
    week = f"Week {current_week}"
    data['check_ins'][careteam_id][week] = {'current_question_index': 0}
    
    # Save the data in the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)



@app.route('/submit_answer', methods=['POST'])
def submit_answer():
  # Check if the API_SECRET from the frontend matches the one stored in the environment
  api_secret_from_frontend = request.headers.get('X-API-SECRET')
  if api_secret_from_frontend != API_SECRET:
    return jsonify({'error': 'Unauthorized access'}), 401

  # Getting the user id from the frontend
  careteam_id = request.headers.get('careteam')
  caregiver_id = request.headers.get('caregiver')
  patient_name = request.headers.get('patient')

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

    # Check if the user responded "no" to question 4
    if current_question_key == 'Q4' and user_response.lower() == 'no':
        end_checkin(careteam_id)
        return jsonify({'message': 'Response saved successfully! You have completed the check-in'})

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
      if "the patient" in next_question['question']:
      	modified_question = next_question['question'].replace("the patient", patient_name)
      else:
      	modified_question = next_question['question']

      if 'options' in next_question:
        return jsonify({
            'message': 'Response saved successfully!',
            'question': modified_question,
            'options': next_question['options']
        })
      else:
        if "the patient" in next_question['question']:
                modified_question = next_question['question'].replace("the patient", patient_name)
        else:
                modified_question = next_question['question']
        return jsonify({
            'message': 'Response saved successfully!',
            'question': modified_question
        })

  except Exception as e:
    # Manually print the traceback to the console
    traceback.print_exc()

    return jsonify({'error': 'Internal Server Error'}), 500




@app.route('/get_questionnaire_question', methods=['GET'])
def get_questionnaire_question():
  # Check if the API_SECRET from the frontend matches the one stored in the environment
  api_secret_from_frontend = request.headers.get('X-API-SECRET')
  if api_secret_from_frontend != API_SECRET:
    return jsonify({'error': 'Unauthorized access'}), 401

  # Getting the user id from the frontend
  careteam_id = request.headers.get('careteam')
  caregiver_id = request.headers.get('caregiver')
  patient_name = request.headers.get('patient')

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
      if "the patient" in qcurrent_question['question']:
                modified_question = qcurrent_question['question'].replace("the patient", patient_name)
      else:
                modified_question = qcurrent_question['question']

      return jsonify({
          'message': "First Question",
          'question': modified_question,
          'options': options
      })
    else:
      if "the patient" in next_question['question']:
                modified_question = qcurrent_question['question'].replace("the patient", patient_name)
      else:
                modified_question = qcurrent_question['question']

      question_text = {'question': qcurrent_question}

      return jsonify({
          'message': "First Question",
          'question': modified_question
      })


@app.route('/submit_questionnaire_answer', methods=['POST'])
def submit_questionnaire_answer():
  # Check if the API_SECRET from the frontend matches the one stored in the environment
  api_secret_from_frontend = request.headers.get('X-API-SECRET')
  if api_secret_from_frontend != API_SECRET:
    return jsonify({'error': 'Unauthorized access'}), 401

  # Getting the user id from the frontend
  careteam_id = request.headers.get('careteam')
  caregiver_id = request.headers.get('caregiver')
  patient_name = request.headers.get('patient')
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
      if "the patient" in next_question['question']:
                modified_question = next_question['question'].replace("the patient", patient_name)
      else:
                modified_question = next_question['question']

      if 'options' in next_question:
        options = [option["option"] for option in next_question["options"]]
        return jsonify({
            'message': 'Response saved successfully!',
            'question': modified_question,
            'options': options
        })
      else:
        return jsonify({
            'message': 'Response saved successfully!',
            'question': modified_question
        })

  except Exception as e:
    # Manually print the traceback to the console
    traceback.print_exc()

    return jsonify({'error': 'Internal Server Error'}), 500



if __name__ == '__main__':
  app.run(host='0.0.0.0', port=3000)


