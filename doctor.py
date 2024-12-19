import streamlit as st


# from langchain.schema import HumanMessage, AIMessage
import uuid
from dotenv import load_dotenv
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from transformers import pipeline
import os
import chromadb
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from database import (
    register_user, login_user, insert_personal_information,
    fetch_user_chat_sessions, fetch_chat_history, save_chat_history,
    fetch_available_slots, book_appointment, collect_feedback
)
from audio_processing import record_audio, recognize_speech, text_to_speech, generate_audio_download_link

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize OpenAI
# openai_api_key = os.getenv("OPENAI_API_KEY")
def create_openai_client(api_key):
    openai.api_key = api_key
def audio():
    tts = gTTS(text=ai_response, lang='en')
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    st.audio(audio_file.getvalue(), format="audio/mp3")

# Create ChromaDB client
def create_chromadb_client():
    try:
        client = chromadb.Client()
        print("ChromaDB client created successfully")
        return client
    except Exception as e:
        print(f"Error creating ChromaDB client: {e}")
   
        return None

client = create_chromadb_client()
collection_name = 'faq_collection'
result=""
# Read FAQ data
def read_faq_file(file_path):
    faq_data = {}
    current_question = None
    current_answer = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Q:"):
                if current_question is not None:
                    faq_data[current_question] = " ".join(current_answer)
                current_question = line
                current_answer = []
            elif line.startswith("A:"):
                current_answer.append(line[2:].strip())  # Remove 'A:' and strip whitespace
            else:
                current_answer.append(line)
        if current_question is not None:
            faq_data[current_question] = " ".join(current_answer)
    return faq_data

file_path = './faq.txt'
faq_data = read_faq_file(file_path)

# Initialize ChromaDB collection
def initialize_collection(client, collection_name, faq_data):
    try:
        try:
            collection = client.get_collection(collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except ValueError:
            print(f"Collection '{collection_name}' not found. Creating a new one.")
            collection = client.create_collection(collection_name)
        
        documents = []
        for question, answer in faq_data.items():
            # Placeholder embedding logic
            embedding = [0.0] * 128
            document = {
                'id': question,
                'text': answer,
                'embedding': embedding
            }
            documents.append(document)

        for document in documents:
            collection.add({
                'id': document['id'],
                'embedding': document['embedding'],
                'text': document['text']
            })
       
        return collection

    except Exception as e:
       
        return None

collection = initialize_collection(client, collection_name, faq_data)

# Initialize toxicity detection model
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")

def detect_toxicity(text):
    results = toxicity_model(text)
    return any(result['label'] == 'toxic' and result['score'] > 0.5 for result in results)
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.info("Listening... Speak now.")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Sorry, the speech recognition service is unavailable."

def text_to_speech(text):
    tts = gTTS(text)
    return tts

# Initialize Streamlit session state
if "username" not in st.session_state:
    st.session_state["username"] = None
if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = None
if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "personal_info_collected" not in st.session_state:
    st.session_state["personal_info_collected"] = False
if "feedback_collected" not in st.session_state:
    st.session_state["feedback_collected"] = False
if "logout" not in st.session_state:
    st.session_state["logout"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to the doctor appointment service! How can I assist you today?"}]
if 'appointment_booked' not in st.session_state:
    st.session_state['appointment_booked'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = 'chat'
# Define functions to switch pages
def go_to_chat():
    st.session_state['page'] = 'chat'

def go_to_booking():
    st.session_state['page'] = 'booking'

def extract_date_time(user_input):
    # Simple regex for extracting date and time from the input
    date_time_pattern = r"(\d{4}-\d{2}-\d{2}) at (\d{2}:\d{2})"
    match = re.search(date_time_pattern, user_input)
    if match:
        date_str, time_str = match.groups()
        return date_str, time_str
    return None, None
# Generate AI response
def generate_response(messages):
    chat = ChatGroq(
        model="Gemma2-9b-It",
        groq_api_key=groq_api_key,
        temperature=0.5
    )

    ai_response = chat.invoke(messages)
    return ai_response.content

def get_most_relevant_faq(user_input, collection):
    if collection is None:
        return None
    
    # Encode the user input
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(user_input).tolist()

    try:
        # Perform the query (Adjust according to your ChromaDB API)
        results = collection.query(
            embeddings=[query_embedding],  # Use the correct argument as per ChromaDB documentation
            k=5  # Assuming you want to retrieve the top 5 results
        )
        
        if results and len(results['documents']) > 0:
            most_relevant = results['documents'][0]
            similarity = results['distances'][0]
            faq_question = most_relevant['id']
            faq_answer = most_relevant['text']
            
            return faq_question, faq_answer, similarity
        return None

    except Exception as e:
        
        return None
# Build message list for AI interaction
def build_message_list():
    messages = []
    
    for user_msg in st.session_state['past']:
        messages.append(HumanMessage(content=user_msg))
    
    for ai_msg in st.session_state['generated']:
        messages.append(AIMessage(content=ai_msg))
    return messages
# Logout functionality
def logout_user():
    st.session_state["username"] = None
    st.session_state["current_session_id"] = None
    st.session_state["past"] = []
    st.session_state["generated"] = []
    st.session_state["messages"] = []
    st.session_state["personal_info_collected"] = False
    st.session_state["feedback_collected"] = False
    st.session_state["logout"] = False
    st.success("You have successfully logged out.")


# Navbar HTML and CSS
navbar_html = """
<nav class="navbar">
  <div class="navbar-container">
    <img  class="navbar-logo">
    <h1 class="navbar-title">Doctor Appointment Chatbot</h1>
  </div>
</nav>
"""

navbar_css = """
<style>
.navbar {
  background-color: black;
  padding: 5px;
}
.navbar-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.navbar-logo {
  height: 100px;
}
.navbar-title {
  color: white;
  text-align: center;
  flex-grow: 1;
  font-size: 30px;
  margin: 0;
}
</style>
"""
st.markdown("""
    <style>
    .stButton button {
        background-color: black;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("""
    <style>
    .sidebar .stButton button {
        background-color: white;
        color: black;
        border: none;
        text-align: left;
        padding: 10px;
        width: 100%;
    }
    .sidebar .stButton button:hover {
        background-color: #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True)

def build_message_list1():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        *st.session_state['messages']
    ]

# Display Navbar
st.markdown(navbar_html, unsafe_allow_html=True)
st.markdown(navbar_css, unsafe_allow_html=True)

if not st.session_state.get('username'):
    # Registration and Login Forms with Radio Button
    mode = st.radio("Select Mode", ("Login", "Register"))
    print("HELLO")
    if mode == "Register":
        st.write("## Register")
        username = st.text_input("Enter Username", key="register_username")
        password = st.text_input("Enter Password", type='password', key="register_password")
        confirm_password = st.text_input("Confirm Password", type='password', key="register_confirm_password")

        if st.button("Register"):
            if password == confirm_password:
                register_user(username, password)
            else:
                st.warning("Passwords do not match")

    elif mode == "Login":
        st.write("## Login")
        username = st.text_input("Enter Username", key="login_username")
        password = st.text_input("Enter Password", type='password', key="login_password")

        if st.button("Login"):
            if login_user(username, password):
                st.success(f"Logged in as {username}")
                st.session_state['username'] = username
                st.session_state['current_session_id'] = str(uuid.uuid4())
                st.session_state['past'] = []
                st.session_state['generated'] = []
                st.session_state['messages'] = []
                
                # Greet the user upon successful login
                st.session_state['greeting'] = f"Hello {username}, welcome! How can I assist you?"
                # st.rerun()
            else:
                st.warning("Invalid Username/Password")

if st.session_state.get('username'):
    # Sidebar for chat sessions
    st.sidebar.write("## Your Chat Sessions:")
    user_sessions = fetch_user_chat_sessions(st.session_state['username'])
    
    if not st.session_state["personal_info_collected"]:
        # Personal Information Collection
        st.write("## Please Provide Your Personal Information")
        name = st.text_input("Name")
        birth_date = st.date_input("Birth Date")
        reason_for_appointment = st.text_area("Reason for Appointment")

        if st.button("Submit Personal Information"):
            insert_personal_information(st.session_state["username"], name, birth_date, reason_for_appointment)
            st.session_state["personal_info_collected"] = True
            st.success("Personal information saved successfully.")
            st.session_state['greeting'] = "You can now start chatting with the doctor."
            st.experimental_rerun() 

    for session_id in user_sessions:
        if st.sidebar.button(session_id):
            st.session_state['current_session_id'] = session_id
            st.session_state['past'] = []
            st.session_state['generated'] = []

            history = fetch_chat_history(st.session_state['username'], session_id)
            for user_msg, ai_msg in history:
                st.session_state['past'].append(user_msg)
                st.session_state['generated'].append(ai_msg)

    # Main chat interface
    chat_container = st.container()
    with chat_container:
        # Display greeting message
        if 'greeting' in st.session_state:
            st.write(st.session_state['greeting'])
            del st.session_state['greeting']
        
        # Display entire chat history for the current session
        for user_msg, ai_msg in zip(st.session_state['past'], st.session_state['generated']):
            st.write(f"**You:** {user_msg}")
            st.write(f"**Doctor:** {ai_msg}")

    if st.session_state['page'] == 'chat':
        with st.form(key='chat_form', clear_on_submit=True):
            col1, col2 = st.columns([9, 1])
            with col1:
                user_input = st.text_input("You:", key="input_text")


            with col2:
                submit_button = st.form_submit_button(label="↑")

            if submit_button and user_input:
                if st.session_state['current_session_id'] is None:
                    st.error("No active session selected.")
                elif detect_toxicity(user_input):
                    st.warning("Your input contains inappropriate language. Please modify your message.")
                    ai_response = "Sorry, I cannot respond to this request."
                    st.session_state.generated.append(ai_response)
                    audio()
                elif "book appointment" in user_input.lower():
                    st.session_state['page'] = 'booking'  # Switch to booking page
                    st.experimental_rerun()  # This is the correct method to refresh the app.
   # Refresh to show booking page

                else:
                    result = get_most_relevant_faq(user_input, collection=None)
                    if result is not None and len(result) == 3:
                        faq_question, faq_answer, similarity = result
                        similarity_threshold = 0.7
                        ai_response = faq_answer if similarity > similarity_threshold else "I'm not sure how to help with that."
                        audio()
                    else:
                        st.session_state['messages'].append({"role": "user", "content": user_input})
                        with st.spinner("Generating response..."):
                            ai_response = generate_response(build_message_list1())
                        audio()

                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(ai_response)
                    save_chat_history(st.session_state['username'], st.session_state['current_session_id'], user_input, ai_response)

                if not "book appointment" in user_input.lower():
                    with chat_container:
                        st.write(f"**You:** {user_input}")
                        st.write(f"**Doctor:** {ai_response}")

    if st.session_state['page'] == 'booking':
        st.write("Please select a date for the appointment:")
        appointment_date = st.date_input("Appointment Date", key='booking_date')

        if appointment_date:
            available_slots = fetch_available_slots(appointment_date)
            if available_slots:
                st.write("Available slots:")
                slot_time_str = st.selectbox("Select a time slot", [slot.strftime("%H:%M") for slot in available_slots])
                
                if st.button("Confirm Booking"):
                    try:
                        result = book_appointment(st.session_state['username'], appointment_date, slot_time_str)
                        cleaned_result = result.strip().lower().rstrip('.')
                        st.write(f"Booking result: '{cleaned_result}'")  # Debug line to check cleaned result
                        print(cleaned_result)
                        if cleaned_result == "slot already booked":
                            st.warning("This slot is already booked. Please choose another available slot.")
                        elif cleaned_result == "appointment booked successfully":
                            st.session_state['generated'].append(result)
                            st.session_state['appointment_booked'] = True
                            st.success(result)
                            # Switch back to chat page
                            st.session_state['page'] = 'chat'
                              # Refresh to show chat page
                        else:
                            st.error(f"Unexpected result: {result}. Please try again.")
                    
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}. Please try again.")
                    
            else:
                st.write("No available slots for the selected date.")

                    
    if st.session_state.get('username'):
        if st.button("Logout"):
            st.session_state["logout"] = True

        if st.session_state["logout"]:
            collect_feedback(st.session_state["username"])
            st.session_state["feedback_collected"] = True

        if st.session_state["feedback_collected"]:
            logout_user()
            st.session_state["logout"] = False
            st.session_state["feedback_collected"] = False
           









# import streamlit as st
# from langchain_openai import ChatGroq
# from langchain.schema import HumanMessage, AIMessage
# import uuid
# from dotenv import load_dotenv
# from gtts import gTTS
# import speech_recognition as sr
# from io import BytesIO
# import os
# import chromadb
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
# import re
# from transformers import pipeline

# from database import (
#     register_user, login_user, insert_personal_information,
#     fetch_user_chat_sessions, fetch_chat_history, save_chat_history,
#     fetch_available_slots, book_appointment, collect_feedback
# )
# from audio_processing import record_audio, recognize_speech, text_to_speech, generate_audio_download_link

# load_dotenv()

# # Initialize Groq API
# groq_api_key = os.getenv("GROQ_API_KEY")

# def audio():
#     tts = gTTS(text=ai_response, lang='en')
#     audio_file = BytesIO()
#     tts.write_to_fp(audio_file)
#     st.audio(audio_file.getvalue(), format="audio/mp3")

# # Create ChromaDB client
# def create_chromadb_client():
#     try:
#         client = chromadb.Client()
#         print("ChromaDB client created successfully")
#         return client
#     except Exception as e:
#         print(f"Error creating ChromaDB client: {e}")
#         return None

# client = create_chromadb_client()
# collection_name = 'faq_collection'

# # Read FAQ data
# def read_faq_file(file_path):
#     faq_data = {}
#     current_question = None
#     current_answer = []

#     with open(file_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if line.startswith("Q:"):
#                 if current_question is not None:
#                     faq_data[current_question] = " ".join(current_answer)
#                 current_question = line
#                 current_answer = []
#             elif line.startswith("A:"):
#                 current_answer.append(line[2:].strip())
#             else:
#                 current_answer.append(line)
#         if current_question is not None:
#             faq_data[current_question] = " ".join(current_answer)
#     return faq_data

# file_path = './faq.txt'
# faq_data = read_faq_file(file_path)

# # Initialize ChromaDB collection
# def initialize_collection(client, collection_name, faq_data):
#     try:
#         try:
#             collection = client.get_collection(collection_name)
#             print(f"Collection '{collection_name}' already exists.")
#         except ValueError:
#             print(f"Collection '{collection_name}' not found. Creating a new one.")
#             collection = client.create_collection(collection_name)

#         documents = []
#         for question, answer in faq_data.items():
#             embedding = [0.0] * 128
#             document = {
#                 'id': question,
#                 'text': answer,
#                 'embedding': embedding
#             }
#             documents.append(document)

#         for document in documents:
#             collection.add({
#                 'id': document['id'],
#                 'embedding': document['embedding'],
#                 'text': document['text']
#             })

#         return collection

#     except Exception as e:
#         return None

# collection = initialize_collection(client, collection_name, faq_data)

# # Initialize toxicity detection model
# toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")

# def detect_toxicity(text):
#     results = toxicity_model(text)
#     return any(result['label'] == 'toxic' and result['score'] > 0.5 for result in results)

# def recognize_speech():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         recognizer.adjust_for_ambient_noise(source)
#         st.info("Listening... Speak now.")
#         audio = recognizer.listen(source)
#     try:
#         return recognizer.recognize_google(audio)
#     except sr.UnknownValueError:
#         return "Sorry, I could not understand the audio."
#     except sr.RequestError:
#         return "Sorry, the speech recognition service is unavailable."

# def text_to_speech(text):
#     tts = gTTS(text)
#     return tts

# # Streamlit session state initialization
# if "username" not in st.session_state:
#     st.session_state["username"] = None
# if "current_session_id" not in st.session_state:
#     st.session_state["current_session_id"] = None
# if "past" not in st.session_state:
#     st.session_state["past"] = []
# if "generated" not in st.session_state:
#     st.session_state["generated"] = []
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []
# if "personal_info_collected" not in st.session_state:
#     st.session_state["personal_info_collected"] = False
# if "feedback_collected" not in st.session_state:
#     st.session_state["feedback_collected"] = False
# if "logout" not in st.session_state:
#     st.session_state["logout"] = False
# if 'appointment_booked' not in st.session_state:
#     st.session_state['appointment_booked'] = False
# if 'page' not in st.session_state:
#     st.session_state['page'] = 'chat'

# def extract_date_time(user_input):
#     date_time_pattern = r"(\d{4}-\d{2}-\d{2}) at (\d{2}:\d{2})"
#     match = re.search(date_time_pattern, user_input)
#     if match:
#         date_str, time_str = match.groups()
#         return date_str, time_str
#     return None, None

# def generate_response(messages):
#     chat = ChatGroq(
#         model="Gemma2-9b-It",
#         groq_api_key=groq_api_key,
#         temperature=0.5
#     )

#     ai_response = chat.invoke(messages)
#     return ai_response.content

# def get_most_relevant_faq(user_input, collection):
#     if collection is None:
#         return None

#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     query_embedding = model.encode(user_input).tolist()

#     try:
#         results = collection.query(
#             embeddings=[query_embedding],
#             k=5
#         )

#         if results and len(results['documents']) > 0:
#             most_relevant = results['documents'][0]
#             similarity = results['distances'][0]
#             faq_question = most_relevant['id']
#             faq_answer = most_relevant['text']

#             return faq_question, faq_answer, similarity
#         return None

#     except Exception as e:
#         return None

# def build_message_list():
#     messages = []
#     for user_msg in st.session_state['past']:
#         messages.append(HumanMessage(content=user_msg))
#     for ai_msg in st.session_state['generated']:
#         messages.append(AIMessage(content=ai_msg))
#     return messages

# def logout_user():
#     st.session_state.clear()
#     st.success("You have successfully logged out.")

# # Navbar HTML and CSS
# navbar_html = """
# <nav class="navbar">
#   <div class="navbar-container">
#     <h1 class="navbar-title">Doctor Appointment Chatbot</h1>
#   </div>
# </nav>
# """

# navbar_css = """
# <style>
# .navbar { background-color: black; padding: 5px; }
# .navbar-title { color: white; text-align: center; font-size: 30px; margin: 0; }
# </style>
# """

# st.markdown(navbar_html, unsafe_allow_html=True)
# st.markdown(navbar_css, unsafe_allow_html=True)

# if not st.session_state.get('username'):
#     mode = st.radio("Select Mode", ("Login", "Register"))

#     if mode == "Register":
#         st.write("## Register")
#         username = st.text_input("Enter Username", key="register_username")
#         password = st.text_input("Enter Password", type='password', key="register_password")
#         confirm_password = st.text_input("Confirm Password", type='password", key="register_confirm_password")

#         if st.button("Register"):
#             if password == confirm_password:
#                 register_user(username, password)
#             else:
#                 st.warning("Passwords do not match")

#     elif mode == "Login":
#         st.write("## Login")
#         username = st.text_input("Enter Username", key="login_username")
#         password = st.text_input("Enter Password", type='password', key="login_password")

#         if st.button("Login"):
#             if login_user(username, password):
#                 st.success(f"Logged in as {username}")
