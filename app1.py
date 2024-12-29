import os
import json, os, signal
import torch
import whisper
import werkzeug
from flask import Flask, request, jsonify
from pydub import AudioSegment
from langchain_community.llms import Ollama


#audio_file_path = "/Users/shubha/Downloads/cv-corpus-12.0-delta-2022-12-07/en/clips/common_voice_en_34925870.mp3"
def transcribe_audio(audio_file_path):
    try:
        audio= whisper.load_audio(audio_file_path, sr=16000)
        audio_tensor = torch.from_numpy(audio).to(torch.float32)
        result = whisper.load_model("base").transcribe(audio_tensor, fp16=False)["text"]
        print(result)
        return result
    except Exception as e:
        return f"Transcription error: {str(e)}"

def extract_information(transcribed_text):
    try:
        with open(transcribed_text,'r') as f:
            few_shot = f.read()
        context = "You are a helpful assistant analyzing a text."
        icl = f"For example, {few_shot}"
        query = f"Extract key information from this text: {transcribed_text}"

        extracted_info = Ollama(model = "mistral").invoke(context + icl + query)
        print(extracted_info)
        return extracted_info
    except Exception as e:
        return f"Information extraction error: {str(e)}"

app = Flask(__name__)
@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "status": "400 Bad Request"
            }), 400
        
        # Get the file
        uploaded_file = request.files['file']
        
        # Check if filename is empty
        if uploaded_file.filename == '':
            return jsonify({
                "error": "No selected file",
                "status": "400 Bad Request"
            }), 400
        
        # Generate secure filename
        filename = werkzeug.utils.secure_filename(uploaded_file.filename)
        file_path = os.path.join("uploads", filename)
        print(file_path)
        
        # Save file
        uploaded_file.save(file_path)
        
        # Validate file type (optional)
        allowed_extensions = ['mp3', 'wav', 'ogg', 'flac']
        if not filename.lower().split('.')[-1] in allowed_extensions:
            os.remove(file_path)
            return jsonify({
                "error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
                "status": "400 Bad Request"
            }), 400
        
        try:
            # Transcribe audio
            transcription = transcribe_audio(file_path)
            
            # Remove uploaded file
            os.remove(file_path)
            
            # Return results
            return jsonify({
                "transcription": transcription,
                "status": "success"
            })
        
        except Exception as e:
            # Remove file in case of processing error
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({
                "error": str(e),
                "status": "500 Internal Server Error"
            }), 500
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "500 Internal Server Error"
        }), 500
@app.route('/process-text', methods=['POST'])
def process_text():
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "status": "400 Bad Request"
            }), 400
        
        # Get the file
        uploaded_file = request.files['file']
        
        # Check if filename is empty
        if uploaded_file.filename == '':
            return jsonify({
                "error": "No selected file",
                "status": "400 Bad Request"
            }), 400
        
        # Generate secure filename
        filename = werkzeug.utils.secure_filename(uploaded_file.filename)
        file_path = os.path.join("uploads", filename)
        
        # Save file
        uploaded_file.save(file_path)
        
        # Validate file type (optional)
        allowed_extensions = ["txt","pdf"]
        if not filename.lower().split('.')[-1] in allowed_extensions:
            os.remove(file_path)
            return jsonify({
                "error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
                "status": "400 Bad Request"
            }), 400
        
        try:
            # Extract information
            extracted_info = extract_information(file_path)
            
            # Remove uploaded file
            os.remove(file_path)
            
            # Return results
            return jsonify({
                "extracted_information": extracted_info,
                "status": "success"
            })
        
        except Exception as e:
            # Remove file in case of processing error
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({
                "error": str(e),
                "status": "500 Internal Server Error"
            }), 500
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "500 Internal Server Error"
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002)
