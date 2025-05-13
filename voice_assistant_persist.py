import argparse
import sys_msgs
import ast
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
from datetime import datetime, timedelta, timezone
from queue import Queue
from time import sleep
from sys import platform
import ollama  # Added for Ollama integration
from speak import speak  # Import the speak function
from colorama import init, Fore, Style
import psycopg
import chromadb
from psycopg.rows import dict_row

init(autoreset=True)

# --- Configuration ---
WAKE_WORD = "COMPUTER"
MODEL_OLLAMA = 'gemma3:4b'
QUERY_GENERATOR = 'gemma3:4b'
CLASSIFIER = 'gemma3:4b'
EXIT_PHRASE = "Exit"

SYS_MSG_OLLAMA = (
    '''System_instructions:
  role: "AI Voice Assistant"
  behavior:
    - "Respond as an expert problem solver specialized in voice interactions."
    - "Provide concise, logical, and helpful answers—avoid verbosity."
    - "Act strictly as a voice assistant, not as a general-purpose chatbot."
    - "Use clear reasoning and deductive logic to address user questions."
    - "If information is outside training scope, inform the user honestly."
  constraints:
    - "No improvisation or fabrication when knowledge is lacking."
    - "Responses must be optimized for spoken clarity and brevity."
'''
)
DB_PARAMS = {
    'dbname': os.getenv('DB_NAME', 'memory_agent'),
    'user': os.getenv('DB_USER', 'admin'),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
}

# --- Global Variables ---
data_queue = Queue()
phrase_bytes = bytes()
triggered = False
last_phrase_time = None
full_transcript = []
chat_history = [{'role': 'system', 'content': SYS_MSG_OLLAMA}]
vector_db_client = chromadb.Client()  # Initialize Chroma client globally

# --- Helper Functions ---
def connect_db():
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg.connect(**DB_PARAMS)
        return conn
    except psycopg.OperationalError as e:
        print(f"Error connecting to the database: {e}")
        return None  # Important: Return None on failure

def fetch_conversations():
    """Fetch all conversations from the database."""
    conn = connect_db()
    if conn is None:
        return []  # Return an empty list if the connection failed.
    try:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute('SELECT * FROM conversations')
            conversations = cursor.fetchall()
        conn.close()
        return conversations
    except Exception as e:
        print(f"Error fetching conversations: {e}")
        conn.close()
        return []

def store_conversations(prompt, response):
    """Store a conversation (prompt and response) in the database."""
    conn = connect_db()
    if conn is None:
        return  # Exit if the database connection failed
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                'INSERT INTO conversations (timestamp, prompt, response) VALUES(CURRENT_TIMESTAMP, %s, %s)',
                (prompt, response),
            )
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error storing conversation: {e}")
        conn.close()

def remove_last_conversation():
    """Remove the last conversation from the database."""
    conn = connect_db()
    if conn is None:
        return  # Exit if the database connection failed
    try:
        with conn.cursor() as cursor:
            cursor.execute('DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)')
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error removing last conversation: {e}")
        conn.close()

def generate_ollama_response(convo_messages):
    """
    Sends a conversation to the Ollama model and streams the response.
    Args:
        convo_messages (list): A list of message dictionaries for the Ollama API.
    Returns:
        str: The full text of the assistant's response.
    """
    try:
        stream = ollama.chat(model=MODEL_OLLAMA, messages=convo_messages, stream=True)
        full_response_text = ''
        print(f'{Fore.LIGHTMAGENTA_EX}{WAKE_WORD}:')
        for chunk in stream:
            content = chunk['message']['content']
            print(f'{Fore.LIGHTCYAN_EX}{content}{Style.RESET_ALL}', end='', flush=True)
            full_response_text += content
        print('\n\n')
        return full_response_text
    except Exception as e:
        print(f"\n[Ollama Error] Failed to get response: {e}")
        return "Sorry, I encountered an error trying to respond."

def create_vectordb(conversations):
    """Create or update the ChromaDB vector database with conversation data."""
    vectordb_name = 'conversations'
    try:
        vector_db_client.delete_collection(name=vectordb_name)
    except ValueError:
        pass  # Collection doesn't exist, which is fine
    vectordb = vector_db_client.create_collection(name=vectordb_name)

    if not conversations:
        return  # if there are no conversations, return.

    for c in conversations:
        serialized_convo = f"prompt: {c['prompt']} response: {c['response']}"
        try:
            embedding = ollama.embeddings(model='nomic-embed-text:latest', prompt=serialized_convo)
            vectordb.add(
                embeddings=[embedding['embedding']],
                documents=[serialized_convo],
                ids=[str(c['id'])] #use the id from the database.
            )
        except Exception as e:
            print(f"Error generating or adding embedding: {e}")

def retrieve_embeddings(query):
    """Retrieve relevant embeddings from the vector database based on a query."""
    vectordb_name = 'conversations'
    vectordb = vector_db_client.get_collection(name=vectordb_name)
    try:
        # Should use create_queries here.
        results = vectordb.query(
            query_texts=[query],
            n_results=5,  # Or another suitable number
        )
        return results['documents']  # Return the documents
    except Exception as e:
        print(f"Error retrieving embeddings: {e}")
        return []

def create_queries(prompt):
    """
    Generates multiple search queries from a given prompt using the Ollama model.
    """
    convo_messages = [
        {'role': 'system', 'content': sys_msgs.query_generation_prompt},  # Make sure this prompt exists.
        {'role': 'user', 'content': prompt},
    ]
    try:
        response = ollama.chat(model=QUERY_GENERATOR, messages=convo_messages)
        # Parse the response.  The response should be a python list of strings.
        queries_str = response['message']['content']
        try:
            queries = ast.literal_eval(queries_str)
            if not isinstance(queries, list):
                print(f"Error: create_queries - Expected a list, got {type(queries)}, returning original prompt")
                return [prompt]
            if not all(isinstance(q, str) for q in queries):
                print(f"Error: create_queries - Expected strings in list, returning original prompt")
                return [prompt]
            return queries

        except (SyntaxError, ValueError) as e:
            print(f"Error parsing Ollama query response: {e}, returning original prompt")
            return [prompt]

    except Exception as e:
        print(f"Error generating queries: {e}, returning original prompt")
        return [prompt]  # Return the original prompt on error

def classify_embedding(query, retrieved_embedding):
    """
    Classifies the relevance of a retrieved embedding to the given query using the Ollama model.
    """
    convo_messages = [
        {'role': 'system', 'content': sys_msgs.classification_prompt}, #  Make sure this prompt exists.
        {'role': 'user', 'content': f"Query: {query}  Context: {retrieved_embedding}"},
    ]
    try:
        response = ollama.chat(model=CLASSIFIER, messages=convo_messages)
        classification = response['message']['content'].strip().lower()
        if "relevant" in classification:
            return True
        elif "irrelevant" in classification:
            return False
        else:
            print(f"Warning: classify_embedding -  Unexpected classification: {classification}, assuming irrelevant")
            return False

    except Exception as e:
        print(f"Error classifying embedding: {e}, assuming irrelevant")
        return False  # Default to irrelevant on error

def recall(prompt):
    """
    Perform semantic recall of relevant past conversations and add them to the current context.
    """
    global chat_history
    retrieved_memories = []
    queries = create_queries(prompt)  # Generate multiple queries
    for query in queries:
        retrieved_embeddings = retrieve_embeddings(query)
        for embedding in retrieved_embeddings:
            if classify_embedding(query, embedding):
                retrieved_memories.append(embedding)

    if retrieved_memories:
        memory_content = "\n".join(retrieved_memories)
        chat_history.append({'role': 'user', 'content': f'MEMORIES: {memory_content} \n\nUSER PROMPT: {prompt}'})
    else:
        chat_history.append({'role': 'user', 'content': prompt}) # add the prompt even if no memories are found.

def main():
    """Main function to run the voice assistant."""
    global phrase_bytes, triggered, last_phrase_time, full_transcript, chat_history

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="medium",
        help="Whisper model to use",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    parser.add_argument(
        "--non_english",
        action='store_true',
        help="Use non-English model (if available for chosen size)",
    )
    parser.add_argument(
        "--energy_threshold",
        default=1000,
        help="Energy level for mic to detect sound",
        type=int,
    )
    parser.add_argument(
        "--record_timeout",
        default=3,  # User's original default
        help="How long to record audio for a single phrase (seconds) by SpeechRecognition",
        type=float,
    )
    parser.add_argument(
        "--phrase_timeout",
        default=7,  # User's original default
        help="How much silence before a voice command is considered over (seconds)",
        type=float,
    )
    args = parser.parse_args()

    # --- Initialization ---
    # Initialize chat history.  This should be done only once at the start.
    chat_history = [{'role': 'system', 'content': SYS_MSG_OLLAMA}]
    # SpeechRecognition setup
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Load Whisper model
    whisper_model_name = args.model
    if not args.non_english and whisper_model_name in ["large", "medium", "small"]:
        whisper_model_name += ".en"
    audio_model = whisper.load_model(whisper_model_name,)##

    # --- Main Loop ---
    try:
        print(f"✅  Voice assistant is ready! Say '{WAKE_WORD}' to begin.")
        # Create a dedicated microphone instance
        mic_source = sr.Microphone(sample_rate=16000)
        
        # Calibrate microphone once before starting
        with mic_source as source:
            recorder.adjust_for_ambient_noise(source)

        def record_callback(recognizer, audio_data):
            """Callback function to process incoming audio data."""
            global phrase_bytes, last_phrase_time
            now = datetime.now(timezone.utc)
            phrase_bytes += audio_data.get_wav_data()
            last_phrase_time = now
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"✅ Ready. Say '{WAKE_WORD}' to begin. To exit, say '{EXIT_PHRASE}' after the wake word.")
            # Clear queue of any audio that might have arrived during Ollama processing

        # Start background listening with dedicated mic source
        stop_listening = recorder.listen_in_background(
            mic_source,
            record_callback,
            phrase_time_limit=args.record_timeout
        )

        while True:
            now = datetime.now(timezone.utc)

            # 1. Detect wake word
            if not triggered and last_phrase_time and (
                now - last_phrase_time
            ) > timedelta(seconds=2):  # Short delay after audio
                audio_data = phrase_bytes
                if audio_data:
                    try:
                        # Use a tiny whisper model for wake word detection to save resources.
                        tiny_model = whisper.load_model("tiny", device="cpu")
                        audio_np = (
                            np.frombuffer(audio_data, dtype=np.int16)
                            .astype(np.float32)
                            / 32768.0
                        )
                        result = tiny_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                        text = result["text"].strip().lower()

                        if WAKE_WORD.lower() in text:
                            triggered = True
                            print(f"{Fore.RED}✅  Wake word detected! Listening for command...")
                            speak("I am listening. What is your command?")
                            phrase_bytes = bytes()  # Clear buffer after wake word
                            full_transcript = []
                            last_phrase_time = now  # Reset timer
                        else:
                            phrase_bytes = b''
                            last_phrase_time = None

                    except Exception as e:
                        print(f"{Fore.RED}Error processing audio for wake word: {e}")
                        phrase_bytes = b''
                        last_phrase_time = None

            # 2. Process command after wake word
            elif triggered and last_phrase_time and (
                now - last_phrase_time
            ) > timedelta(seconds=args.phrase_timeout):
                audio_data = phrase_bytes
                if audio_data:
                    try:
                        audio_np = (
                            np.frombuffer(audio_data, dtype=np.int16)
                            .astype(np.float32)
                            / 32768.0
                        )
                        result = audio_model.transcribe(audio_np, language="en")
                        text = result["text"].strip().lower()
                        full_transcript.append(text)
                        print(f"{Fore.YELLOW}Transcribed command: {text}")

                        if EXIT_PHRASE.lower() in text:
                            print(f"{Fore.BLUE}Exiting on user command.")
                            speak("Goodbye!")
                            break  # Exit the main loop
                        else:
                            # Join the full transcript and process
                            full_text = " ".join(full_transcript)
                            print(f"{Fore.LIGHTCYAN_EX}Full transcript: {full_text}")

                            if "/recall" in full_text:
                                prompt = full_text.replace("/recall", "").strip()
                                recall(prompt)
                                response = generate_ollama_response(chat_history)
                                speak(response)
                                store_conversations(prompt, response)
                                chat_history.append({'role': 'assistant', 'content': response})

                            elif "/forget" in full_text:
                                remove_last_conversation()
                                if len(chat_history) > 1:
                                    chat_history.pop() # Remove the last user message
                                speak("I have forgotten the last interaction.")

                            elif "/memorize" in full_text:
                                prompt = full_text.replace("/memorize", "").strip()
                                store_conversations(prompt, "Memory stored.")
                                speak("I have stored that in my memory.")

                            else:
                                # normal chat
                                chat_history.append({'role': 'user', 'content': full_text}) # Add this line
                                response = generate_ollama_response(chat_history)
                                speak(response)
                                store_conversations(full_text, response)
                                chat_history.append({'role': 'assistant', 'content': response})

                            # Reset state
                            triggered = False
                            phrase_bytes = bytes()
                            full_transcript = []
                            last_phrase_time = None

                    except Exception as e:
                        print(f"{Fore.RED}Error processing audio: {e}")
                        speak("Sorry, I encountered an error processing your request.")
                        triggered = False
                        phrase_bytes = bytes()
                        full_transcript = []
                        last_phrase_time = None

            # 3. Check for general timeout (if not triggered, to clear long silent audio accumulation)
            elif not triggered and last_phrase_time and (
                now - last_phrase_time > timedelta(seconds=args.phrase_timeout * 2)
            ):
                if len(phrase_bytes) > 0:
                    phrase_bytes = bytes()
                    full_transcript = []
                last_phrase_time = None

            sleep(0.15)

    except KeyboardInterrupt:
        print("\n🛑 Exiting voice assistant via KeyboardInterrupt.")
        speak("Exiting. Goodbye!")
    except Exception as e:
        print(f"\n⚠️  An unexpected error occurred: {e}")
        speak("An unexpected error occurred.")
        import traceback

        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if 'stop_listening' in locals() and stop_listening:
            stop_listening(wait_for_stop=False)
        print("Goodbye!")


if __name__ == "__main__":
    main()
