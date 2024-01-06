import time
import gaim_config as gc
import gradio as gr
from pprint import pprint
from chatbot import Llama, TEMPLATE, IMAGE_PROMPT_TEMPLATE
from generate_image import GenImage

# Create LLM to use for multiple types of interactions
llm = Llama(model_path=gc.MODEL_PATH)
llm.createLlm()

# Create chatbot instance
cb = Llama(llm.model_path)
cb.llm = llm.llm
cb.createTemplate(template=TEMPLATE)

# Create instruction instance
instruction = Llama(llm.model_path)
instruction.llm = llm.llm
instruction.createTemplate(template=IMAGE_PROMPT_TEMPLATE)
instruction.createConversation()

imgSD = GenImage(model=gc.SD_MODEL_PATH)

def chat_predict(history, session):
    message = history[-1][0]
    prediction = cb.predict(message, session)
    history[-1][1] = prediction
    
    return history

def user_output(message, history):
    return "", history + [[message, None]]

def on_load(LblSession, request:gr.Request):
    session = cb.createNewSession()
    cb.createConversation(session=session)
    response = cb.start(session=session)

    return [[None, response]], session

def createImgPromptFromHistory(history):
    historyStr = ""
    for i in history[:-1]:
        historyStr = f'{historyStr}Human: {i[0]}\nAI: {i[1]}\n\n'

    recent = f'Human: {history[-1][0]}\nAI: {history[-1][1]}'
    response = instruction.predictImagePrompt(recent=recent, history=historyStr)
    history.append([None, response])

    return history

def generateImageFromChatbot(history):
    historyStr = ""
    for i in history:
        historyStr = f'{historyStr}Human: {i[0]}\nAI: {i[1]}\n\n'

    img = generateImage(prompt=historyStr)

    return img.image

def generateImage(prompt, neg_prompt=''):
    imgSD.loadPrompt(prompt)
    imgSD.loadNegPrompt(neg_prompt)
    imgSD.splitLargePrompt()
    imgSD.generateImage()

    return imgSD

def generateImageFromEndChat(history):
    prompt = history[-1][1]
    img = generateImage(prompt=prompt)

    return img.image

def image_mod(image):
    return image

def fillImageWithoutContent():
    img.set_image()

img = gr.Image()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.ClearButton([msg, chatbot])

            sessionLabel = gr.Label(visible=False)

            msg.submit(user_output, [msg, chatbot], [msg, chatbot]).then(
                chat_predict, [chatbot, sessionLabel], [chatbot]
            ).then(
                createImgPromptFromHistory, [chatbot], [chatbot]
            ).then(
                #generateImageFromChatbot, [chatbot], [img]
                generateImageFromEndChat, [chatbot], [img]
            )

        with gr.Column():
            iface = gr.Interface(fn=fillImageWithoutContent, inputs=[], outputs=img)

    gr.Blocks.load(demo, fn=on_load, inputs=[sessionLabel], outputs=[chatbot, sessionLabel]).then(
        generateImageFromChatbot, [chatbot], [img]
    )

demo.launch(server_name='0.0.0.0')