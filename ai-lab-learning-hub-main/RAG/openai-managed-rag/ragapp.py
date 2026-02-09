import os
import glob
import time
import json
import streamlit as st
import logging
import cProfile
import pstats
import io
import traceback
from openai import OpenAI, AuthenticationError
from html_templates import css, user_template, bot_template

# Global variable for logging level
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'WARNING').upper()

# Function to set logging level
def set_logging_level(level):
    logging_level = getattr(logging, level, logging.WARNING)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging_level)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

# Set logging level based on global variable
set_logging_level(LOGGING_LEVEL)

logger = logging.getLogger(__name__)

client = None

# Function to log errors with stack trace
def log_error(error_message):
    """Logs the error message along with the full stack trace."""
    logger.error(f"{error_message}\n{traceback.format_exc()}")

def is_running_in_docker():
    try:
        # Check for /.dockerenv file
        if os.path.exists('/.dockerenv'):
            return True
        # Check for docker in /proc/1/cgroup
        with open('/proc/1/cgroup', 'rt') as f:
            if any('docker' in line for line in f):
                return True
    except FileNotFoundError:
        pass
    return False

def load_config(config_dir="app/config", config_file="config.json"):
    config_path = os.path.join(config_dir, config_file)
    logger.info("Loading configuration file")

    if not os.path.exists(config_path):
        logger.warning("Configuration file not found, creating default configuration")
        config = {"assistant_id": "",
                "vector_store_id": "",
                "api_key": ""}
        save_config(config, config_dir, config_file)
    else:
        logger.info("Loading configuration file")
    
    with open(config_path, "r") as file:
        return json.load(file)

def save_config(config, config_dir="app/config", config_file="config.json"):
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, config_file), "w") as file:
        json.dump(config, file)
        logger.info("Configuration saved")

# Function to initialize OpenAI client
def initialize_openai_client(api_key):
    global client
    if client is None:
        client = OpenAI(api_key=api_key)
        try:
            response = client.chat.completions.create(
                messages=[{"role":"user", "content":"Hello world"}],
                model="gpt-4o-mini"
            )
        except AuthenticationError as e:
            client = None
            raise e

def check_assistant_exists(assistant_id):
    if not assistant_id:
        logger.warning("Vector store ID is empty")
        return False
    
    try:
        response = client.beta.assistants.retrieve(assistant_id)
        exists = True if response else False
        if exists:
            logger.debug(f"Assistant {assistant_id} exists")
        else:
            logger.warning(f"Assistant {assistant_id} does not exist")
        return exists
    except Exception as e:
        logger.error("An error occurred while checking the assistant", exc_info=True)
        return False
    
def check_vector_store_exists(vector_store_id):
    if not vector_store_id:
        logger.warning("Vector store ID is empty")
        return False
    
    try:
        response = client.vector_stores.retrieve(vector_store_id)
        exists = True if response else False
        if exists:
            logger.debug(f"Vector store {vector_store_id} exists")
        else:
            logger.warning(f"Vector store {vector_store_id} does not exist")
        return exists
    except Exception as e:
        logger.error("An error occurred while checking the vector store", exc_info=True)
        return False

def profile_function(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    logger.info(s.getvalue())
    
    return result

def startAssistantThread(prompt):
    logger.info(f"Starting assistant thread for query: {prompt}")
    messages = [{"role": "user", "content": prompt}]
    try:
        thread = client.beta.threads.create(messages=messages)
        logger.info(f"Thread {thread.id} created successfully")
        logger.info(f"Request Id {thread._request_id}")
        return thread.id
    except Exception as e:
        logger.error("Failed to create assistant thread", exc_info=True)
        raise e

def retrieveThread(thread_id):
    logger.info(f"Retrieving thread {thread_id}")
    try:
        thread_messages = client.beta.threads.messages.list(thread_id)
        logger.debug(f"Retrieved {len(thread_messages.data)} messages from thread {thread_id}")
        list_messages = thread_messages.data
        thread_messages = []
        for message in list_messages:
            obj = {}
            obj['content'] = process_response_with_annotations(thread_id, message.id)
            obj['role'] = message.role
            thread_messages.append(obj)
        logger.debug("Thread messages: \n", thread_messages[::-1])
        return thread_messages[::-1]
    except Exception as e:
        logger.error("Failed to retrieve thread", exc_info=True)
        raise e

def addMessageToThread(thread_id, prompt):
    client.beta.threads.messages.create(thread_id, role="user", content=prompt)

def runAssistant(thread_id, assistant_id):
    logger.info(f"Running assistant thread {thread_id} for assistant {assistant_id}")
    try:
        run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
        logger.info(f"Run {run.id} started successfully")
        return run.id
    except Exception as e:
        logger.error("Failed to run assistant thread", exc_info=True)
        raise e

def checkRunStatus(thread_id, run_id):
    logger.debug(f"Checking status for run {run_id} in thread {thread_id}")
    try:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        logger.debug(f"Status for run {run_id}: {run.status}")
        return run.status
    except Exception as e:
        logger.error("Failed to check run status", exc_info=True)
        raise e
    
def process_response_with_annotations(thread_id, message_id):
    # Retrieve the message object
    message = client.beta.threads.messages.retrieve(
        thread_id=thread_id,
        message_id=message_id
    )
    
    # Extract the message content
    message_content = message.content[0].text
    annotations = message_content.annotations
    citations = []

    # Iterate over the annotations and add footnotes
    for index, annotation in enumerate(annotations):
        # Replace the text with a footnote
        message_content.value = message_content.value.replace(annotation.text, f' [{index+1}]')
        # Gather citations based on annotation attributes
        if (file_citation := getattr(annotation, 'file_citation', None)):
            logger.debug("File Citation Obj:\n", file_citation)
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f'<div class="citation">[{index+1}] from {cited_file.filename}</div>')
        elif (file_path := getattr(annotation, 'file_path', None)):
            cited_file = client.files.retrieve(file_path.file_id)
            citations.append(f'<div>[{index+1}] Click <here> to download {cited_file.filename}</div>')
        
    # Add footnotes to the end of the message before displaying to user
    message_content.value += '\n' + '\n'.join(citations)
    
    return message_content.value


# initialise a boolean attr in session state

if "configbutton" not in st.session_state:
    st.session_state.configbutton = False

# write a function for toggle functionality
def toggle():
    if st.session_state.configbutton:
        st.session_state.configbutton = False
    else:
        st.session_state.configbutton = True



def main():

    st.set_page_config(page_title="What the AI (WAI)", layout="wide", initial_sidebar_state="expanded")
    

    # Apply CSS
    st.write(css, unsafe_allow_html=True)
    st.title("Bio Fin Analyst")
    st.subheader("I answer questions using my 20 annual-report PDFs loaded in my knowldegebase. Ask me anything about revenue, margins, or risk factors.")
    # # create the button
    # st.button("Config", on_click=toggle)

    # Decide whether to expand based on missing configuration
    all_configs_present = all([
        st.session_state.get('api_key'),
        st.session_state.get('assistant_id'),
        st.session_state.get('vector_store_id')
    ])

    # Enable querying only if all configurations are valid
    if not all_configs_present:
        st.session_state.configbutton = True

    # with st.expander("Click to expand"):
    #     st.write("This is inside the expander")
    #     st.button("Button inside expander")
    #     ecol1, ecol2 = st.columns(2)
    #     with ecol1:
    #         st.write("This is column 1")
    #     with ecol2:
    #         st.write("This is column 2")


    sidebar = st.sidebar

    with sidebar:
        st.header("Configuration")
        st.write("Collpase when not needed")
        if is_running_in_docker():
            st.info("The script is running inside a Docker container.")
            config_dir = "/app/config"
        else:
            st.info("The script is not running inside a Docker container.")
            config_dir = "app/config"

   
        config = load_config(config_dir)

        # Initialize variables
        api_key = st.session_state.get('api_key', config.get("api_key"))
        assistant_id = st.session_state.get('assistant_id', config.get("assistant_id"))
        vector_store_id = st.session_state.get('vector_store_id', config.get("vector_store_id"))

        # Check and prompt for API Key
        if not api_key:
            api_key = st.text_input("Please enter your OpenAI API key:", type="password", key="api_key_input")
            if st.button("Save API Key", key="save_api_key_button"):
                try:
                    initialize_openai_client(api_key)
                    config["api_key"] = api_key
                    save_config(config, config_dir)
                    st.session_state['api_key'] = api_key
                    st.success("API key saved.")
                # except AuthenticationError:
                #     st.error("Failed to initialize OpenAI client. Please try again.")
                except AuthenticationError as e:
                    error_msg = "Invalid API key. Please check your key and try again."
                    log_error(f"{error_msg}: {str(e)}")
                    st.error(error_msg)

                except ConnectionError as e:
                    error_msg = "Network issue detected. Please check your internet connection and try again."
                    log_error(f"{error_msg}: {str(e)}")
                    st.error(error_msg)

                except TimeoutError as e:
                    error_msg = "Request timed out. OpenAI servers may be experiencing high traffic. Try again later."
                    log_error(f"{error_msg}: {str(e)}")
                    st.error(error_msg)

                except Exception as e:
                    error_msg = "An unexpected error occurred"
                    log_error(f"{error_msg}: {str(e)}")
                    st.error(f"{error_msg}. Please try again or contact support.")

        
        else:
            st.info("Using saved API key.")
            try:
                initialize_openai_client(api_key)
            except AuthenticationError:
                config["api_key"] = ""
                save_config(config, config_dir)
                st.session_state['api_key'] = None
                st.warning("The saved API key is invalid. Please enter a new API key.")

        # Check and prompt for Assistant ID
        if client and not assistant_id:
            assistant_id = st.text_input("Please enter your OpenAI Assistant ID:", key="assistant_id_input")
            if st.button("Save Assistant ID", key="save_assistant_id_button"):
                if check_assistant_exists(assistant_id):
                    config["assistant_id"] = assistant_id
                    save_config(config, config_dir)
                    st.session_state['assistant_id'] = assistant_id
                    st.success("Assistant ID saved.")
                else:
                    st.warning("Failed to validate the entered OpenAI Assistant ID. Please try again.")
        elif client:
            st.info("Using saved OpenAI Assistant ID.")

        # Check and prompt for Vector Store ID
        if client and assistant_id and not vector_store_id:
            vector_store_id = st.text_input("Please enter your OpenAI Vector Store ID:", key="vector_store_id_input")
            if st.button("Save Vector Store ID", key="save_vector_store_id_button"):
                if check_vector_store_exists(vector_store_id):
                    config["vector_store_id"] = vector_store_id
                    save_config(config, config_dir)
                    st.session_state['vector_store_id'] = vector_store_id
                    st.success("Vector Store ID saved.")
                else:
                    st.warning("Failed to validate the entered OpenAI Vector Store ID. Please try again.")
        elif client and assistant_id:
            st.info("Using saved OpenAI Vector Store ID.")

    thread_id = st.session_state.get("thread_id", None)
    
    # Enable querying only if all configurations are valid
    if client and check_assistant_exists(assistant_id) and check_vector_store_exists(vector_store_id):
        query = st.text_input("Ask a question about the Revenue, Margins, or Risk Factors", key="query_input")

        if query:
            with st.spinner('Generating answer...'):
                try:
                    if thread_id == None:
                        #Thread creation
                        thread_id = profile_function(startAssistantThread, query)
                        st.session_state.thread_id = thread_id
                    else:
                        addMessageToThread(thread_id, query)

                    # Run Assistant
                    run_id = profile_function(runAssistant, thread_id, assistant_id)
                    st.session_state.run_id = run_id

                    status = profile_function(checkRunStatus, thread_id, run_id)
                    st.session_state.status = status

                    # Optimized run retry to reduce unnecessary waiting
                    while st.session_state.status != 'completed':
                        with st.spinner('Waiting for process to complete...'):
                                backoff_time = 1 # Start with 1 sec delay
                                max_backoff_time = 30 # Max delay
                                max_retries = 10

                                retries = 0

                                while st.session_state.status != "completed" and retries < max_retries:
                                    with st.spinner("Generating answer..."):
                                        time.sleep(backoff_time)
                                        st.session_state.status = profile_function(checkRunStatus, st.session_state.thread_id, st.session_state.run_id)

                                        # If status is still not completed, increase delay with backoff
                                        if st.session_state.status != "completed":
                                            backoff_time = min(backoff_time * 2, max_backoff_time) # Exponential backoff
                                            retries += 1

                                if st.session_state.status != "completed":
                                    st.error("Request timed out. Please try again.")
                                st.session_state.status = profile_function(checkRunStatus, st.session_state.thread_id, st.session_state.run_id) 

                    # Store conversation
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    chat_history = profile_function(retrieveThread, st.session_state.thread_id)
                    last_2_messages = chat_history[-2:]
                    # st.session_state.chat_history.extend(last_2_messages)
                    for message in last_2_messages:
                        if message['role'] == 'user':
                            st.session_state.chat_history.append(f"USER: {message['content']}")
                        else:
                            st.session_state.chat_history.append(f"AI: {message['content']}")
  
                    # Display conversation in reverse order
                    for i, message in enumerate(reversed(st.session_state.chat_history)):
                        if i % 2 == 0: st.markdown(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)
                        else: st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
                except Exception as e:
                    logger.error("An error occurred", exc_info=True)
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
