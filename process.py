from pathlib import Path
import sys
import faiss
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain import LLMChain
from langchain.llms import OpenAIChat
from langchain.prompts import Prompt
import requests
import os
import openai

from sqlalchemy import create_engine, text

import requests

from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()


openai.api_key = os.environ["OPENAI_API_KEY"]




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


def train():

  # Check there is data fetched from the database

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
  embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
  store = FAISS.from_texts(docs, embeddings)

  faiss.write_index(store.index, "training.index")
  store.index = None

  with open("faiss.pkl", "wb") as f:
    pickle.dump(store, f)


def splitter(text):
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{
          "role":
          "system",
          "content":
          'Reply the chunks seperated by symbol "*" and no spaces'
      }, {
          "role":
          "user",
          "content":
          'Split this text into meaningful chunks seperated by "*" symbol. A chunk maybe a single or multiple lines: '
          + text
      }])
  return response.choices[0].message.content


def runPrompt():
  index = faiss.read_index("training.index")

  with open("faiss.pkl", "rb") as f:
    store = pickle.load(f)

  store.index = index

  with open("training/master.txt", "r") as f:
    promptTemplate = f.read()

  prompt = Prompt(template=promptTemplate,
                  input_variables=["history", "context", "question"])

  llmChain = LLMChain(prompt=prompt,
                      llm=OpenAIChat(
                          temperature=0.5,
                          model_name='gpt-3.5-turbo',
                          openai_api_key=os.environ["OPENAI_API_KEY"]))

  def onMessage(question, history):
    # Check if the question is related to the user's location
    if "need" in question.lower():
      location = input(
          "Please provide your complete location so that we can find the nearest required professional for you: "
      )
      latitude, longitude = geocode(location, os.environ["MAP_KEY"])
      # Store the latitude and longitude in your database
      # Perform actions related to address-based functionality
      # Sort professionals based on proximity using latitude and longitude

    docs = store.similarity_search(question)
    contexts = []
    for i, doc in enumerate(docs):
      contexts.append(f"Context {i}:\n{doc.page_content}")

    answer = llmChain.predict(question=question,
                              context="\n\n".join(contexts),
                              history=history)

    return answer

  history = []
  while True:
    question = input("Ask a question > ")
    answer = onMessage(question, history)
    bot_answer = splitter(answer)
    print(f"Bot: {bot_answer}")
    history.append(f"Human: {question}")
    history.append(f"Bot: {bot_answer}")


# Define your Mapbox API access token
mapbox_access_token = os.environ["MAP_KEY"]


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
  latitude, longitude = geocode_address(primary_address, primary_address_city, primary_address_state, primary_address_country, zipcode)

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
          INSERT INTO aiadvocatehistory (user_question, bot_answer)
            VALUES ('Hey','he')
          
         
      """

    # Execute the SQL query with parameters
    result = connection.execute(text(sql_query))

    # Fetch and print the query results
    for row in result:
      print(row)

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


def work():
  result = getdata()
  if result is not None:
    convert_and_save_to_file(result)




username = "aiassistantevvaadmin"
password = "EvvaAi10$"
hostname = "aiassistantdatabase.postgres.database.azure.com"
database_name = "aidatabasecombined"

  # Construct the connection URL
db_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"


# Define the SQLAlchemy model
Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'aiadvocatehistory'  # Adjust table name as needed

    # Add a dummy primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    user_question = Column(String)
    bot_answer = Column(String)

def insert_conversation(user_question, bot_answer):
    try:
        # Create a SQLAlchemy engine and session
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Create a Conversation object
        conversation = Conversation(user_question=user_question, bot_answer=bot_answer)

        # Add the Conversation object to the session and commit the transaction
        session.add(conversation)
        session.commit()

        # Close the session
        session.close()

    except Exception as e:
        # Handle exceptions (e.g., database errors)
        print(f"Error inserting conversation: {e}")

      
# Example usage:
def inserter():
    user_question1 = "What's the weather today?"
    bot_answer1 = "The weather is sunny and warm."

    user_question2 = "Tell me a joke."
    bot_answer2 = "Why did the chicken cross the road? To get to the other side!"

    insert_conversation(user_question1, bot_answer1)
    insert_conversation(user_question2, bot_answer2)



# Example usage
train()
#runPrompt()

#getdata()
#inserter()
