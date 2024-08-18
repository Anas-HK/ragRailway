from flask import Flask, request, jsonify
import os
import scipy.spatial.distance
import pdfplumber
import openai

# Set your OpenAI API key
openai.api_key = "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"

print("Current Working Directory:", os.getcwd())


# Define default PDF paths
pdf_paths = [
    'Email consent form.pdf',
    'rainenotfunc.pdf'
]

app = Flask(__name__)

def extract_text_from_pdf(pdf_path):
    text_by_page = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_by_page.append(page.extract_text())
    return text_by_page

def preprocess_text(text):
    return ' '.join(text.split())

def get_embeddings(texts, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=texts, model=model)
    embeddings = [item['embedding'] for item in response['data']]
    return embeddings

def index_documents(pdf_paths):
    all_texts = []
    all_embeddings = []

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        text_by_page = extract_text_from_pdf(pdf_path)
        preprocessed_texts = [preprocess_text(text) for text in text_by_page]
        embeddings = get_embeddings(preprocessed_texts)

        all_texts.extend(preprocessed_texts)
        all_embeddings.extend(embeddings)

    return all_texts, all_embeddings

def retrieve_relevant_documents(query, texts, embeddings, model="text-embedding-ada-002", top_k=5):
    query_embedding = get_embeddings([query], model=model)[0]
    similarities = [1 - scipy.spatial.distance.cosine(query_embedding, embedding) for embedding in embeddings]
    top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    relevant_texts = [texts[i] for i in top_k_indices]
    return relevant_texts

def generate_response(query, relevant_texts, model="gpt-3.5-turbo"):
    prompt = f"Query: {query}\n\nRelevant Information:\n"
    for text in relevant_texts:
        prompt += f"- {text}\n"
    prompt += "\nResponse:"

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Please format the result in HTML for email. Keep it under 250 tokens."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=250
    )
    return response['choices'][0]['message']['content'].strip()

@app.route('/query', methods=['GET'])
def query():
    query = request.args.get('query')

    if not query:
        return "Please provide a query parameter.", 400

    # Index documents from predefined PDFs
    texts, embeddings = index_documents(pdf_paths)

    # Retrieve relevant documents
    relevant_texts = retrieve_relevant_documents(query, texts, embeddings)

    # Generate response
    response = generate_response(query, relevant_texts)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
