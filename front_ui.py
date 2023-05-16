# frontend.py
import gradio as gr
import requests

def make_request(message):
    url = "http://localhost:8000/chat"  # Update with your backend URL
    data = {"text": message}
    response = requests.post(url, json=data)
    return response.json()["message"]

with gr.Blocks(height=900) as iface:

    with gr.Tab("ChatBot"):
        with gr.Column():
            chatbot = gr.Chatbot(label="ChatBBB_dev", placeholder="ChatBBB_dev",show_label=True).style(container=False)
        with gr.Row():
          with gr.Column(scale=0.85):
            msg = gr.Textbox(placeholder="พิมพ์คำถามของคุณที่นี่... (กด enter หรือ submit หลังพิมพ์เสร็จ)",show_label=False)
          with gr.Column(scale=0.15, min_width=0):
            submit = gr.Button("Submit")
        clear = gr.Button("Clear")


        def user(user_message, history):
            # bot_message = chatgpt_chain.predict(human_input=user_message)
            bot_message = make_request(user_message)
            history.append((user_message, bot_message))
            return "", history,gr.update(visible=True)
        def reset():
        #   chatgpt_chain.memory.clear()
          print("clear!")
        # feedback_chatbot_submit.click(fn=save_up, inputs=[chatbot,chatbot_radio,feedback_chatbot], outputs=[feedback_chatbot_ok,feedback_chatbot_box,], queue=False)
        clear.click(reset, None, chatbot, queue=False)
        submit_event = msg.submit(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=True)
        submit_click_event = submit.click(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=True)
    
iface.queue()
iface.launch()