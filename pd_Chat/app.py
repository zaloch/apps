#Project: pdfChat
__author__ = "Gonzalo Zeballos"
__license__ = "GNU GPLv3"
__version__ = "1.0"
# Description: A simple app that curates a scientific publicationa and reads it to you.
# Scroll down to the bottom to see more license info.

#Web and File Handling
import streamlit as st
import PyPDF2
from math import *

#Chatbot
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI

#Text to Speech
import text_to_speech as tts
from explainer import *

#Music Player
import base64

#files
main_logo = f"https://github.com/zaloch/apps/blob/main/pd_Chat/img/pdfchat2.jpg?raw=true"
musique_file = f"https://github.com/zaloch/apps/blob/main/pd_Chat/musique/Boys%20(Summertime%20Love)%20-%20Sabrina%20(Salerno)%20-%20backingtrackx.com.mp3"
charming_boy = f"https://raw.githubusercontent.com/zaloch/apps/main/pd_Chat/video/charming_boy.mp4"

#CSS
st.set_page_config(page_title='pdfChat!', layout='wide')
streamlit_style = """
                    <style>
                    @import url('https://fonts.googleapis.com/css2?family=Cabin');

                    html, body, [class*="css"]  {
                    font-family: 'Cabin', sans-serif;
                    }

                    .media-container {
                        position: fixed;
                        top: 2%;
                        right: 0;
                        width: 24%;
                        height: 100%;
                        padding: 2rem;
                        overflow-y: auto;
                    }

                    .music-container {
                        position: fixed;
                        top: 21%;
                        right: 0;
                        width: 24%;
                        height: 100%;
                        padding: 1rem;
                        overflow-y: auto;
                    }

                    .footer {
                    position: fixed;
                    bottom: 0;
                    left: 0;
                    width: 100%;
                    background-color: #3b403d;
                    text-align: center;
                    padding: 0.25rem 0;
                    box-sizing: border-box;
                    z-index: 1000;
                    }

                    .tip-jar {
                        display: inline-block;
                        margin-bottom: 0;
                    } 

                    .tip-jar a {
                        color: blue;
                        text-decoration: yellow;
                    }

                    .tip-jar a:hover {
                        text-decoration: underline;
                    }

                    </style>
                    """
st.markdown(streamlit_style, unsafe_allow_html=True)

#Python Code
def display_header() -> None:
    st.image(main_logo, width = 700, use_column_width = True)
    #st.markdown('<div class="center-image"><img src="img/pdfchat.jpg" width="500"></div>', unsafe_allow_html=True)

def display_footer() -> None:
    paypal_link = "https://www.paypal.me/zaloworks"  # Replace with your PayPal link
    cashapp_link = "https://cash.app/$neurogz"  # Replace with your Cash App link

    footer_html = f"""
        <style>
            ...
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.0/css/all.min.css">
        </style>
        <div class="footer">
            <p class="tip-jar">
                <i class="fas fa-donate"></i>
                Tip Jar :
                <a href={paypal_link} target="_blank"><i class="fab fa-paypal"></i> PayPal </a>
                 or  
                <a href={cashapp_link} target="_blank"><i class="fab fa-square"></i> Cash App </a>
            </p>
        </div>
        """
    st.markdown(footer_html, unsafe_allow_html=True)

def display_widgets() -> tuple[UploadedFile, str]:
    #text_area = st.text_area("Copy and paste a section of the paper here, young padawan.", key = "area_for_text_unique")
    file = st.file_uploader("Or Upload your paper here, you lazy grad student.", type=["pdf"])
    
    #if not (file, text_area):
    if not (file):
        st.error("So where's the pdf? Upload it or copy and paste it here.")
    #return file, text_area
    return file

def choose_voice():
    voices = tts.list_available_names()
    return st.sidebar.selectbox(
        "Choose a voice to narrate your paper",
        voices,
    )


def create_music_player(audio_file_path: str, initial_volume: float = 0.15) -> None:
    
    with open(audio_file_path, "rb") as f:
        audio_data = f.read()

    b64_audio = base64.b64encode(audio_data).decode()

    audio_html = f'''
    <div class = "music-container">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.0/css/all.min.css">
        <audio id="music-player" src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3" controls loop></audio>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const audio = document.getElementById('music-player');
            audio.volume = {initial_volume};
        }});
    </script>
    '''

    st.markdown(audio_html, unsafe_allow_html=True)
    #st.write("Yaw, control them tunes here.")

def create_autoplay_video_player(video_file_path: str) -> None:
    with open(video_file_path, "rb") as f:
        video_data = f.read()

    b64_video = base64.b64encode(video_data).decode()

    video_html = f"""
    <div class="media-container">
        <video width="300" height="240" controls autoplay loop>
            <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
        </video>
    </div>
    """

    st.markdown(video_html, unsafe_allow_html=True)

def display_video_and_music_player(video_path: str, audio_file_path: str) -> None:
    st.markdown('<div class="media-container">', unsafe_allow_html=True)
    create_autoplay_video_player(video_path)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="music-container">', unsafe_allow_html=True)
    create_music_player(audio_file_path, initial_volume = 0.15)
    st.markdown('</div>', unsafe_allow_html=True)


def extract_pdf() -> str:
    #if "uploaded_pdf" not in st.session_state or "pasted_text" not in st.session_state:
    #    st.session_state.uploaded_pdf, st.session_state.pasted_text = display_widgets()
    
    #uploaded_pdf, pasted_text = st.session_state.uploaded_pdf, st.session_state.pasted_text

    uploaded_pdf = display_widgets()
    #, pasted_text = display_widgets()

    if uploaded_pdf:
        pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
        extracted_text = ""

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            extracted_text += text

        # Display PDF in a smaller window
        st.markdown('<div class="media-container">', unsafe_allow_html=True)
        st.markdown('<embed src="data:application/pdf;base64,{}" width="100%" height="600px" />'.format(
            base64.b64encode(uploaded_pdf.read()).decode("utf-8")), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        return extracted_text
    #return pasted_text or ""
    return ""

def display_extracted_text(text: str, height: str = 500) -> None:
    custom_css = f"""
    <style>
        .scrollable-text {{
            height: {height}px;  /* Adjust height to control the number of visible lines */
            overflow-y: auto;
            background-color: #3b403d;  /* Set the background color to dark gray */
            padding: 10px;  /* Add padding for spacing */
        }}
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown('<div class="scrollable-text">' + text.replace("\n", "<br>") + '</div>', unsafe_allow_html=True)


#Cache variables to avoid loss of workflow
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()


def display_api_input() -> str:   
    api_key = st.sidebar.text_input("API Key:", type="password", key="api_key_input")
    return api_key


def main() -> None:
    # Display the header
    display_header()
    # Display API input box
    st.sidebar.write(" Voice Narrator")
    selected_voice = choose_voice()

    # Display the footer with the tip jar
    display_footer()
    
    st.markdown(streamlit_style, unsafe_allow_html=True)
    # Display API input box
    st.sidebar.write("🤖 API Input")
    with st.sidebar.expander(" 🛠️ Settings ", expanded=True):
    # Option to preview memory store
        if st.checkbox("Preview memory store"):
            st.write(st.session_state.entity_memory.store)
        # Option to preview memory buffer
        if st.checkbox("Preview memory buffer"):
            st.write(st.session_state.entity_memory.buffer)
        MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo-0301','gpt-4','text-davinci-003','text-davinci-002'])
        K = st.number_input(' (#)Summary of prompts to consider',min_value=0,max_value=1000)
    st.sidebar.write(" ")
    API_O = display_api_input()
    st.sidebar.write(" ")

    if not API_O:
        # Display error message if no API key is provided
        st.warning("Please enter your API key in the sidebar before uploading a file!", icon = "⚠️" ) 
        st.markdown(''' 
            ```
            ⬅️ On the sidebar enter your openAI API Key & Hit enter 🔐 

            📁 Upload a PDF file or copy and paste text into the text box.

            🗣️ Select a voice from the sidebar.

            🎧 Hit play to listen to the audio.

            📝 Hit the button to save the summaries to a file.

            🔄 Hit the button to start a new chat.

            ```
            
            ''')
        st.sidebar.warning('API key required to try this app. The API key is not stored in any form.')
        st.sidebar.info("Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.")
           
    if API_O:
        # Display the upload button interface
        st.warning("Warning: uploaded files have precedence on copied and pasted text.")

        #Conversation = create_bot(API_O = API_O, MODEL = MODEL, K = K, ENTITY_MEMORY_CONVERSATION_TEMPLATE = ENTITY_MEMORY_CONVERSATION_TEMPLATE)

        # Create an OpenAI instance
        if MODEL == 'gpt-3.5-turbo' or 'gpt-4' or 'gpt-3.5-turbo-0301':
            llm = ChatOpenAI(temperature=0,
                             openai_api_key=API_O,
                             model_name=MODEL,
                             verbose=False,
                             max_tokens = 500)
        else:
            llm = OpenAI(temperature=0,
                        openai_api_key=API_O, 
                        model_name=MODEL, 
                        verbose=False) 
        

        # Create a ConversationEntityMemory object if not already created
        if 'entity_memory' not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K)
            

        # Create the ConversationChain object with the specified configuration
        Conversation = ConversationChain(
                llm=llm, 
                prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                memory=st.session_state.entity_memory
            )  

        if pdf_content := extract_pdf():

            st.header("Extracted Text")
            display_extracted_text(pdf_content)
            st.header("The blurb")

            with st.spinner(text="Yeah science, b*%tch! - Jesse Pinkman ... probably"):
                # Send the text in chunks of 4096 words to the API
                chunks = ceil(len(pdf_content.split(" "))/100)

                Conversation.run("You are a Scientist And you provide answers for the layperson in 100 words or less. Stricly limit yourself to this. Also remain engaging!")

                title = Conversation.run(input = retrieve_better_title(text = pdf_content[0*len(pdf_content)//chunks:1*len(pdf_content)//chunks]))
                st.session_state.past.append(title)
                st.session_state.generated.append(title)

                paper_content = Conversation.run(input = retrieve_paper_text(text = pdf_content[1*len(pdf_content)//chunks:2*len(pdf_content)//chunks]))
                st.session_state.past.append(paper_content)
                st.session_state.generated.append(paper_content)

                references = Conversation.run(input = retrieve_key_references(text = pdf_content[-1*len(pdf_content)//chunks::]))
                st.session_state.past.append(references)
                st.session_state.generated.append(references)


            #with st.spinner(text="Let me get a postdoc to do this for me..."):
            #    tts.convert_text_to_mp3(
            #        message= title, voice_name=selected_voice, mp3_filename="title.mp3"
            #    )
            
            #with st.spinner(text =
            #        "Eureka... - Every scientist ever\n"
            #        "I've got it! - Archimedes, probably")
            
            #with st.spinner()):
            #    tts.convert_text_to_mp3(
            #        message = paper_content,
            #        voice_name=selected_voice,
            #        mp3_filename="paper_content.mp3",
            #    )

            # Format output
            st.success("Ok here's your summary")
            st.warning("Did you try turning on the volume? - Bill Nye, perhaps")
            st.write("                          ")   
            st.write(f"**Layperson's Title:**")
            display_extracted_text(f"{title}", height = 200)
            #st.audio("title.mp3")
            st.write("                          ")
            st.write(f"**Summary:**")    
            display_extracted_text(f"{paper_content}")
            #st.audio("paper_content.mp3")
            st.write("                          ")
            st.write(f"**Key References:**")
            display_extracted_text(f"{references}", height = 400)
            st.write("                          ")
            st.write("                          ")
            
            # Allow to download as well
            download_str = []
            # Display the conversation history using an expander, and allow the user to download it
            with st.expander("Conversation", expanded=True):
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    st.info(st.session_state["past"][i],icon="🧐")
                    st.success(st.session_state["generated"][i], icon="🤖")
                    download_str.append(st.session_state["past"][i])
                    download_str.append(st.session_state["generated"][i])
                
                # Can throw error - requires fix
                download_str = '\n'.join(download_str)
                if download_str:
                    st.download_button('Download',download_str)
    
    # Show a reset button
    st.button("🔄", on_click = new_chat, type='primary')

    # Display the video and music player
    display_video_and_music_player(charming_boy, musique_file)

if __name__ == "__main__":
    main()

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.