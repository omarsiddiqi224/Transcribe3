import subprocess
#import streamlit as st
import gradio as gr
from transformers import pipeline
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
#import whisper
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.float16, device="cuda")
#pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
 

def read_file_content(filename):
    with open(filename, 'r') as file:
        return file.read()
        

def summarize(transcribed_text):
    print("before prompt")
    prompt_template = """Based on the provided conversation, your task is to summarize the key findings and derive insights. The diarization may not be 100 percent accurate, so take into consideration the conversation. Please create a thorough summary note under the heading 'SUMMARY KEY NOTES' and include bullet points about the key items discussed.
    Ensure that your summary is clear and informative, conveying all necessary information (include how the caller was feeling, meaning sentiment analysis). Focus on the main points mentioned in the conversation, such as Claims, Benefits, Providers, and other relevant topics. Additionally, create an action items/to-do list based on the insights and findings from the conversation.
    The main points to look for in a conversation are: Claims, Correspondence and Documents, Eligibility & Benefits, Financials, Grievance & Appeal, Letters, Manage Language, Accumulators, CGHP & Spending Account Buy Up, Group Search, Member Enrollment & Billing, Manage ID Cards, Member Limited Liability, Member Maintenance, Other Health insurance (COB), Provider Lookup, Search/ Update UM Authorization, Prefix and Inter Plan Search, Promised Action Search Inventory.
    Please note that while you can look for other points, it is important to prioritize the main points mentioned above.

        """

    #clean_text1 = prompt_template.strip()
    #clean_text1 = prompt_template

    print("before message")
    messages = [
        {
            "role": "system",
            "content": prompt_template,
        },
        {"role": "user", "content": transcribed_text},
    ]

    print("before tokenizer")
    
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("before pipe output")
    
    outputs = pipe(prompt, max_new_tokens=2000, do_sample=True, temperature=0.5, top_k=50, top_p=0.95)
    
    print("before generated text (all)")
    
    my_string = outputs[0]["generated_text"]
    
    print("before answer (split)")
    
    summarizing = my_string.split("<|assistant|>",1)[1]
    
    print("before return")
    
    return summarizing                                           


def summarize2(transcribed_text):

    print("before prompt")
    prompt_template = """Based on the call conversation, your task is to summarize the key findings and derive insights. Please create a thorough summary note under the heading 'SUMMARY KEY NOTES' and include bullet points about the key items discussed.
    Ensure that your summary is clear and informative, conveying all necessary information (include how the caller was feeling, meaning sentiment analysis). Focus on the main points mentioned in the conversation, such as Claims, Benefits, Providers, and other relevant topics. Additionally, create an action items/to-do list based on the insights and findings from the conversation.
    The main points to look for in a conversation are: Claims, Correspondence and Documents, Eligibility & Benefits, Financials, Grievance & Appeal, Letters, Manage Language, Accumulators, CGHP & Spending Account Buy Up, Group Search, Member Enrollment & Billing, Manage ID Cards, Member Limited Liability, Member Maintenance, Other Health insurance (COB), Provider Lookup, Search/ Update UM Authorization, Prefix and Inter Plan Search, Promised Action Search Inventory.
    Please note that while you can look for other points, it is important to prioritize the main points mentioned above.

        """


    print("before message")
    messages = [
        {
            "role": "system",
            "content": prompt_template,
        },
        {"role": "user", "content": transcribed_text},
    ]

    print("before tokenizer")

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("before pipe output")

    outputs = pipe(prompt, max_new_tokens=2000, do_sample=True, temperature=0.5, top_k=50, top_p=0.95)

    print("before generated text (all)")

    my_string = outputs[0]["generated_text"]

    print("before answer (split)")

    summarizing = my_string.split("<|assistant|>",1)[1]

    print("before return")

    return summarizing


#state = ""
text = ""
total = ""
def transcribe(audio, state=""):
    #global state
    global text
    global total
    time.sleep(3)
    text = transcriber(audio)["text"]
    state += text + " "
    total = state
    return state, state

#real_aud_text = ""
#transcribed_text = ""
#def transcribe(stream, new_chunk):
#    global real_aud_text
#    global transcribed_text
#    sr, y = new_chunk
#    y = y.astype(np.float32)
#    y /= np.max(np.abs(y))

#    if stream is not None:
#        stream = np.concatenate([stream, y])
#    else:
#        stream = y

#    transcribed_text = transcriber({"sampling_rate": sr, "raw": stream})["text"]
#    real_aud_text += transcribed_text + " "

 #   return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]

def aud_summarize2():
    global total
    audio_summarized = summarize2(total)
    return audio_summarized


#def aud_summarize():
 #   audio_summarized = summarize2(state)
  #  return audio_summarized

def process_audio_file(audio_file):
    if audio_file:
        # Split the file path to get directory and file name
        file_dir, file_name = os.path.split(audio_file)

        # Construct path for the converted audio file
        converted_file_path = os.path.join(file_dir, "converted_audio.wav")

        # Convert the audio file to WAV format with 32-bit PCM
        subprocess.call(['ffmpeg', '-i', audio_file, '-c:a', 'pcm_s32le', converted_file_path, '-y'])

        return converted_file_path

# Replace 'audio_file' with your actual file path variable
#processed_audio_file = process_audio_file(audio_file)

transcripts = ""
def transcribe2(audio_file):
    if audio_file:
        head, tail = os.path.split(audio_file)
        path = head
        
        if tail[-3:] != 'wav':
            subprocess.call(['ffmpeg', '-i', audio_file, "audio.wav", '-y'])
            tail = "audio.wav"
  
        subprocess.call(['ffmpeg', '-i', audio_file, "audio.wav", '-y'])
        tail = "audio.wav"
        print("before diarize") 
        #os.system(f"insanely-fast-whisper --file-name {tail} --hf_token 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 --flash True")
        subprocess.run([f"insanely-fast-whisper --file-name {tail} --hf_token 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 --flash True"], shell=True, capture_output=True, text=True)
        print("after diarize")
        subprocess.run(["python cleanup.py"], shell=True, capture_output=True, text=True)
        #os.system("python cleanup.py")

       
        text = read_file_content('audio.txt')
        print("after reading")
        fixed = text.strip()
        summarized = summarize(fixed)
        #print("after summary")
        global transcripts
        transcripts = text
        torch.cuda.empty_cache()
        return(text, summarized)
 


with gr.Blocks() as demo:
 
    #gr.Interface(
     #   transcribe,
      #  ["state", gr.Audio(sources=["microphone"], type="filepath", streaming=True)],
       # ["state", "text"],
       # live=True,
       # title="Real-Time Transcription",
    #)

    gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(sources="microphone", type="filepath", streaming=True),
            'state'
        ],
        outputs=[
            "textbox",
            "state"
        ],
        title="Real-Time Transcription and Summarization",
        live=True
    )
    #gr.Interface(

        #input_text = real_aud_text
        #print("input text =======>",input_text)

   # output_text = gr.Textbox(label="Summary:")
    #greet_btn = gr.Button("Get Summary")
    #greet_btn.click(fn=aud_summarize, inputs=[], outputs=output_text)
        

    output_text = gr.Textbox(label="Summary 2:")
    greet_btn = gr.Button("Get Summary 2")
    greet_btn.click(fn=aud_summarize2, inputs=[], outputs=output_text)
        
        #aud_summarize,
        #inputs="text",
        #print("inputs=====>",inputs),
        #outputs="text",
        #description="Click 'Get Summary' to summarize the audio conversation",

    #)

 
    gr.Interface(transcribe2,
        inputs=[
            gr.Audio(sources ='upload', type='filepath', label='Audio File'),
           
            ],
        outputs=["text", "text"],
        title="Transcribe and Summarize Files" 
    )

    gr.Interface(summarize2,
            inputs="text", 
            outputs="text", 
            title="Summarize Transcription")
 
 

#demo.launch(share=True)
# demo.queue().launch(debug=True, share=True, inline=False)
demo.queue().launch(debug=True, share=True, inline=False,auth = ('admin', 'admin'))
