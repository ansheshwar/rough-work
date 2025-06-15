from flask import Flask, request, jsonify
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os


from dotenv import load_dotenv

load_dotenv()   

app = Flask(__name__)

google_llm = GoogleGenerativeAI(google_api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-pro")


prompt_generate_questions = PromptTemplate(
    input_variables=["question"],
    template="Given the question: '{question}', generate 10 related questions that explore different aspects of the topic."
)

generate_questions_sequence = prompt_generate_questions | google_llm

prompt_answer_questions = PromptTemplate(
    input_variables=["related_question"],
    template="Provide a detailed answer to the following question: '{related_question}'."
)


answer_question_sequence = prompt_answer_questions | google_llm

@app.route('/generate', methods=['POST'])
def generate():
    print("Generate endpoint hit")  
    data = request.json
    question = data.get('question')
    

    related_questions = generate_questions_sequence.invoke({"question": question})
    questions = related_questions.split("\n")
    
    
    answers = []
    for q in questions:
        if q.strip():
            answer = answer_question_sequence.invoke({"related_question": q.strip()})
            answers.append({"question": q.strip(), "answer": answer})

    return jsonify(answers)

if __name__ == '__main__':
    app.run(debug=True)
