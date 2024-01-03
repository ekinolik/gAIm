import time
import gaim_config as gc
import gradio as gr
from pprint import pprint
from chatbot import Llama

cb = Llama(model_path=gc.MODEL_PATH)
cb.createTemplate()
cb.createLlm()

def chat_predict(message, history, session):
    prediction = cb.predict(message, session)
    history.append((message, prediction))
    return "", history

def on_load(LblSession, request:gr.Request):
    session = cb.createNewSession()
    cb.createConversation(session=session)
    response = cb.start(session=session)

    return [[None, response]], session

def startSession():
    response = cb.start(session=cb.session)
    #print(response) # DEBUG
    return [[None, response]]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()#startSession)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    sessionLabel = gr.Label(visible=False)

    msg.submit(chat_predict, [msg, chatbot, sessionLabel], [msg, chatbot])

    gr.Blocks.load(demo, fn=on_load, inputs=[sessionLabel], outputs=[chatbot, sessionLabel])

demo.launch(server_name='0.0.0.0')