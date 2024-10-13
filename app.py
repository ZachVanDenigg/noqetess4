# from flask import Flask, render_template, request

# # Create a flask app
# app = Flask(
#   __name__,
#   template_folder='templates',
#   static_folder='static'
# )

# @app.get('/')
# def index():
#   return render_template('index.html')

# @app.get('/hello')
# def hello():
#   return render_template('hello.html', name=request.args.get('name'))

# @app.errorhandler(404)
# def handle_404(e):
#     return '<h1>404</h1><p>File not found!</p><img src="https://httpcats.com/404.jpg" alt="cat in box" width=400>', 404


# if __name__ == '__main__':
#   # Run the Flask app
#   app.run(host='0.0.0.0', debug=True, port=8080)

from flask import Flask, render_template, request, redirect, url_for, flash
import os
import PyPDF2
import docx
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from openai import AzureOpenAI
from werkzeug.utils import secure_filename

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}
app.secret_key = 'supersecretkey'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# OpenAI client setup
endpoint = os.getenv("ENDPOINT_URL", "https://noqeai1.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "e1b6fed821db493db7d0e783f23b7872")
client = AzureOpenAI(azure_endpoint=endpoint, api_key=subscription_key, api_version="2024-05-01-preview")

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file_path):
    doc = docx.Document(docx_file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def is_query_relevant(query, document_text, threshold=0.4):
    query_words = re.findall(r'\b\w+\b', query.lower())
    doc_words = re.findall(r'\b\w+\b', document_text.lower())
    query_counter = Counter(query_words)
    doc_counter = Counter(doc_words)
    common_words = set(query_counter) & set(doc_counter)
    return len(common_words) / len(query_words) >= threshold

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract text based on file type
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension == '.pdf':
                with open(filepath, 'rb') as pdf_file:
                    docpdf_text = extract_text_from_pdf(pdf_file)
            elif file_extension == '.docx':
                docpdf_text = extract_text_from_docx(filepath)
            else:
                flash("Unsupported file format. Please upload a PDF or DOCX file.")
                return redirect(request.url)

            # Process the user query
            user_query = request.form.get('query')
            if user_query and is_query_relevant(user_query, docpdf_text):
                completion = client.chat.completions.create(
                    model=deployment,
                    messages=[
                        {"role": "system", "content": "You are an AI assistant..."},
                        {"role": "user", "content": f"The document contains: {docpdf_text}. My question: {user_query}"}
                    ],
                    max_tokens=4000,
                    temperature=0.7
                )
                response = completion.choices[0].message['content']
                return render_template('result.html', query=user_query, response=response)
            else:
                flash("The query is not relevant to the document content.")
                return redirect(request.url)
    
    return render_template('index.html')

@app.get('/hello')
def hello():
    return render_template('hello.html', name=request.args.get('name'))

@app.errorhandler(404)
def handle_404(e):
    return '<h1>404</h1><p>File not found!</p><img src="https://httpcats.com/404.jpg" alt="cat in box" width=400>', 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
