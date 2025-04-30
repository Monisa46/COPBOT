import os
import json
import sqlite3
import ssl
import pandas as pd
import datetime
import random
import streamlit as st
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import wavio
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load copbot_intents.json Correctly
INTENTS_FILE = os.path.join(BASE_DIR, "copbot_intents.json")

try:
    with open(INTENTS_FILE, "r", encoding="utf-8") as file:
        copbot_intents = json.load(file)  # Store in a variable
except FileNotFoundError:
    raise FileNotFoundError(f"❌ ERROR: 'copbot_intents.json' not found in {BASE_DIR}")
except json.JSONDecodeError:
    raise ValueError("❌ ERROR: 'copbot_intents.json' is not a valid JSON file. Please check formatting.")

# ✅ Verify JSON structure before using it
if "intents" not in copbot_intents or not isinstance(copbot_intents["intents"], list):
    raise ValueError("❌ ERROR: 'copbot_intents.json' is missing the 'intents' key or is not a list. Please check formatting.")

# Initialize empty lists for training
tags, patterns = [], []

# Extract Patterns and Tags from JSON
for intent in copbot_intents["intents"]:
    if "tag" in intent and "patterns" in intent and "responses" in intent:
        for pattern in intent["patterns"]:
            tags.append(intent["tag"])  # Add intent tag
            patterns.append(pattern)  # Add pattern
    else:
        raise ValueError(f"❌ ERROR: Intent {intent} is missing 'tag', 'patterns', or 'responses'. Check JSON formatting.")

print("✅ Successfully loaded intents and prepared training data!")
def chatbot(input_text):
    # First, check if the query matches an intent
    input_text_vectorized = vectorizer.transform([input_text])
    tag = clf.predict(input_text_vectorized)[0]

    # Use the corrected intents JSON
    for intent in copbot_intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    # If no intent match, check the database for relevant information
    return fetch_all_data(input_text)


# Setup SQLite Database
DB_FILE = os.path.join(BASE_DIR, "copbot.db")
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create tables if they do not exist
cursor.execute('''CREATE TABLE IF NOT EXISTS legal_info (law TEXT, description TEXT, punishment TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS traffic_fines (violation TEXT, fine_amount TEXT, points TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS fir_data (fir_number TEXT, crime_type TEXT, status TEXT, police_station TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS missing_persons (name TEXT, age TEXT, last_seen TEXT, contact TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS stolen_vehicles (vehicle_no TEXT, model TEXT, stolen_from TEXT, reported_date TEXT)''')

conn.commit()

# Train NLP Model for Intents
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags, patterns = [], []
for intent in copbot_intents["intents"]:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)


# 📌 Function to Fetch FIR Status
def get_fir_status(fir_number):
    conn = sqlite3.connect("copbot.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM fir_data WHERE fir_number = ?", (fir_number,))
    result = cursor.fetchone()
    
    conn.close()

    if result:
        return f"🚔 **FIR Number:** {result[0]}\n🔍 **Crime Type:** {result[1]}\n📌 **Status:** {result[2]}\n🏢 **Police Station:** {result[3]}"
    else:
        return "❌ FIR not found. Please enter a valid FIR number."

# 📌 Function to Fetch Stolen Vehicle Status
def get_stolen_vehicle(vehicle_no):
    conn = sqlite3.connect("copbot.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM stolen_vehicles WHERE vehicle_no = ?", (vehicle_no,))
    result = cursor.fetchone()
    
    conn.close()

    if result:
        return f"🚗 **Vehicle No.:** {result[0]}\n🚘 **Model:** {result[1]}\n📍 **Stolen From:** {result[2]}\n📅 **Reported Date:** {result[3]}"
    else:
        return "❌ No record of a stolen vehicle with this number."

# 📌 Function to Fetch Legal Information
def get_legal_info(law):
    conn = sqlite3.connect("copbot.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM legal_info WHERE law LIKE ?", ('%' + law + '%',))
    result = cursor.fetchone()
    
    conn.close()

    if result:
        return f"📜 **Law:** {result[0]}\n🔍 **Description:** {result[1]}\n⚖ **Punishment:** {result[2]}"
    else:
        return "❌ No legal information found for this law."

# 📌 Function to Fetch Missing Person Details
def get_missing_person(name):
    conn = sqlite3.connect("copbot.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM missing_persons WHERE name LIKE ?", ('%' + name + '%',))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return f"🔎 **Missing Person:** {result[0]}\n🎂 **Age:** {result[1]}\n📍 **Last Seen:** {result[2]}\n📞 **Contact:** {result[3]}"
    else:
        return "❌ No missing person found with this name."

# 📌 Function to Fetch Traffic Fine Details
def get_traffic_fine(violation):
    conn = sqlite3.connect("copbot.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM traffic_fines WHERE violation LIKE ?", ('%' + violation + '%',))
    result = cursor.fetchone()
    
    conn.close()

    if result:
        return f"🚦 **Violation:** {result[0]}\n💰 **Fine Amount:** {result[1]}\n⚠ **Points Deducted:** {result[2]}"
    else:
        return "❌ No fine information found for this violation."



def upload_datasets(fir_file, legal_file, traffic_file, missing_file, stolen_file):
    conn = sqlite3.connect("copbot.db")
    cursor = conn.cursor()

    # Upload FIR Data
    if fir_file is not None:
        try:
            df_fir = pd.read_csv(fir_file)
            df_fir.to_sql("fir_data", conn, if_exists="replace", index=False)
            st.success("✅ FIR Data Uploaded Successfully!")
        except Exception as e:
            st.error(f"❌ Error uploading FIR data: {str(e)}")

    # Upload Legal Info
    if legal_file is not None:
        try:
            df_legal = pd.read_csv(legal_file)
            df_legal.to_sql("legal_info", conn, if_exists="replace", index=False)
            st.success("✅ Legal Data Uploaded Successfully!")
        except Exception as e:
            st.error(f"❌ Error uploading Legal data: {str(e)}")

    # Upload Traffic Fines
    if traffic_file is not None:
        try:
            df_traffic = pd.read_csv(traffic_file)
            df_traffic.to_sql("traffic_fines", conn, if_exists="replace", index=False)
            st.success("✅ Traffic Fines Data Uploaded Successfully!")
        except Exception as e:
            st.error(f"❌ Error uploading Traffic data: {str(e)}")

    # Upload Missing Persons Data
    if missing_file is not None:
        try:
            df_missing = pd.read_csv(missing_file)
            df_missing.to_sql("missing_persons", conn, if_exists="replace", index=False)
            st.success("✅ Missing Persons Data Uploaded Successfully!")
        except Exception as e:
            st.error(f"❌ Error uploading Missing Persons data: {str(e)}")

    # Upload Stolen Vehicles Data
    if stolen_file is not None:
        try:
            df_stolen = pd.read_csv(stolen_file)
            df_stolen.to_sql("stolen_vehicles", conn, if_exists="replace", index=False)
            st.success("✅ Stolen Vehicles Data Uploaded Successfully!")
        except Exception as e:
            st.error(f"❌ Error uploading Stolen Vehicles data: {str(e)}")

    conn.close()
# Function to Fetch Any Relevant Data from Database
def fetch_all_data(query):
    conn = sqlite3.connect("copbot.db")
    cursor = conn.cursor()

    # Check FIR Data
    cursor.execute("SELECT * FROM fir_data WHERE fir_number = ?", (query,))
    result = cursor.fetchone()
    if result:
        return f"🚔 **FIR Number:** {result[0]}\n🔍 **Crime Type:** {result[1]}\n📌 **Status:** {result[2]}\n🏢 **Police Station:** {result[3]}"

    # Check Missing Persons
    cursor.execute("SELECT * FROM missing_persons WHERE name LIKE ?", ('%' + query + '%',))
    result = cursor.fetchone()
    if result:
        return f"🔎 **Missing Person:** {result[0]}\n🎂 **Age:** {result[1]}\n📍 **Last Seen:** {result[2]}\n📞 **Contact:** {result[3]}"

    # Check Stolen Vehicles
    cursor.execute("SELECT * FROM stolen_vehicles WHERE vehicle_no = ?", (query,))
    result = cursor.fetchone()
    if result:
        return f"🚗 **Vehicle No.:** {result[0]}\n🚘 **Model:** {result[1]}\n📍 **Stolen From:** {result[2]}\n📅 **Reported Date:** {result[3]}"

    # Check Traffic Fines
    cursor.execute("SELECT * FROM traffic_fines WHERE violation LIKE ?", ('%' + query + '%',))
    result = cursor.fetchone()
    if result:
        return f"🚦 **Violation:** {result[0]}\n💰 **Fine Amount:** {result[1]}\n⚠ **Points Deducted:** {result[2]}"

    # Check Legal Information
    cursor.execute("SELECT * FROM legal_info WHERE law LIKE ?", ('%' + query + '%',))
    result = cursor.fetchone()
    if result:
        return f"📜 **Law:** {result[0]}\n🔍 **Description:** {result[1]}\n⚖ **Punishment:** {result[2]}"

    conn.close()
    return "🤖 No matching records found."
def create_chat_history_table():
    conn = sqlite3.connect("copbot.db")
    cursor = conn.cursor()

    # Create chat history table if it does not exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        user_input TEXT, 
                        bot_response TEXT, 
                        timestamp TEXT)''')

    conn.commit()
    conn.close()

# Call the function to ensure the table is created before using it
create_chat_history_table()


# 📌 Function to Save Chat History
def save_chat_history(user_input, bot_response):
    conn = sqlite3.connect("copbot.db")
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, user_input TEXT, bot_response TEXT, timestamp TEXT)''')

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO chat_history (user_input, bot_response, timestamp) VALUES (?, ?, ?)", 
                   (user_input, bot_response, timestamp))

    conn.commit()
    conn.close()

# 📌 Function to Display Chat History
def display_chat_history():
    conn = sqlite3.connect("copbot.db")
    cursor = conn.cursor()

    cursor.execute("SELECT user_input, bot_response, timestamp FROM chat_history ORDER BY id DESC LIMIT 20")
    chat_records = cursor.fetchall()
    conn.close()

    if not chat_records:
        st.write("❌ No chat history found.")
    else:
        for record in chat_records:
            st.markdown(f"👤 **User:** {record[0]}")
            st.markdown(f"🤖 **CopBot:** {record[1]}")
            st.markdown(f"🕒 **Timestamp:** {record[2]}")
            st.markdown("---")  # Adds a separator



# Function to Capture Voice Input
def voice_input():
    duration = 5  # Recording duration in seconds
    samplerate = 44100  # Standard audio sample rate
    st.write("🎤 Recording... Speak now!")

    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        wavio.write("recorded_audio.wav", recording, samplerate, sampwidth=2)

        recognizer = sr.Recognizer()
        with sr.AudioFile("recorded_audio.wav") as source:
            audio = recognizer.record(source)

        return recognizer.recognize_google(audio)
    except Exception as e:
        return f"❌ Could not process the audio: {str(e)}"

# Function to Translate Text
def translate_text(text, target_language):
    try:
        translated_text = GoogleTranslator(source="auto", target=target_language).translate(text)
        return f"🌍 **Translated Text:** {translated_text}"
    except Exception as e:
        return f"❌ Translation Error: {str(e)}"

# Streamlit UI
def main():
    st.title("🚔 CopBot - AI-Powered Police Assistant")

    menu = ["Chat", "Check FIR Status", "Find Missing Person", "Check Stolen Vehicle", "Traffic Fine", "Legal Information", "Translate", "Upload Dataset", "Chat History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Chat":
        user_input = st.text_input("You:")
        if st.button("🎙️ Voice Input"):
            user_input = voice_input()
        if user_input:
            response = chatbot(user_input)
            st.write("🤖 " + response)

            # Save chat history
            save_chat_history(user_input, response)

    elif choice == "Check FIR Status":
        st.header("🔍 Check FIR Status")
        fir_number = st.text_input("Enter FIR Number:")
        if st.button("Check"):
            st.write(get_fir_status(fir_number))

    elif choice == "Find Missing Person":
        st.header("🔎 Find Missing Person")
        name = st.text_input("Enter Name:")
        if st.button("Search"):
            st.write(get_missing_person(name))

    elif choice == "Check Stolen Vehicle":
        st.header("🚗 Check Stolen Vehicle")
        vehicle_no = st.text_input("Enter Vehicle Number:")
        if st.button("Search"):
            st.write(get_stolen_vehicle(vehicle_no))

    elif choice == "Traffic Fine":
        st.header("🚦 Check Traffic Fines")
        violation = st.text_input("Enter Traffic Violation:")
        if st.button("Check Fine"):
            st.write(get_traffic_fine(violation))

    elif choice == "Legal Information":
        st.header("📜 Search Legal Information")
        law_name = st.text_input("Enter Law Name (e.g., IPC 420):")
        if st.button("Search"):
            st.write(get_legal_info(law_name))

    elif choice == "Translate":
        st.header("🌍 Translate Queries")
        user_query = st.text_input("Enter your text:")
        target_lang = st.selectbox("Select Language:", {"ta": "Tamil", "hi": "Hindi", "bn": "Bengali", "te": "Telugu", "kn": "Kannada"})
        if st.button("Translate"):
            st.write(translate_text(user_query, target_lang))

    elif choice == "Upload Dataset":
        st.header("📂 Upload Multiple CSV Files to Update Database")

        # Upload all datasets at once
        fir_file = st.file_uploader("Upload FIR Data CSV", type=["csv"])
        legal_file = st.file_uploader("Upload Legal Info CSV", type=["csv"])
        traffic_file = st.file_uploader("Upload Traffic Fines CSV", type=["csv"])
        missing_file = st.file_uploader("Upload Missing Persons CSV", type=["csv"])
        stolen_file = st.file_uploader("Upload Stolen Vehicles CSV", type=["csv"])

        if st.button("Upload All"):
            upload_datasets(fir_file, legal_file, traffic_file, missing_file, stolen_file)

    elif choice == "Chat History":
        st.header("📜 Chat History")
        display_chat_history()

    elif choice == "About":
        st.header("ℹ️ About CopBot")
        st.write("CopBot assists with police-related queries, FIR tracking, missing persons, traffic fines, stolen vehicles, and legal information.")

if __name__ == '__main__':
    main()
