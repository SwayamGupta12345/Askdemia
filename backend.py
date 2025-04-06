import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pymongo import MongoClient
from typing import List
from fastapi import status

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")

if not api_key:
    raise ValueError("Error: GEMINI_API_KEY is missing from the .env file.")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["chatbot"]
messages_collection = db["message"]

# Initialize LLM with Gemini
llm = LLM(model="gemini/gemini-1.5-pro", api_key=api_key, verbose=True)
# Define AI Agents with improved backstories

chatbot_agent = Agent(
    name="Study Assistant",
    role="Conversational AI",
    goal="Assist students with study planning, scheduling, and stress management.",
    backstory="Once a virtual assistant designed for productivity hacks, the Study Assistant evolved into a deeply empathetic and organized AI, tailored to support students in creating effective study routines. With experience in balancing workload, time management, and well-being, this assistant is your go-to guide for achieving academic success with a clear plan.",
    llm=llm,
    memory=True,
    verbose=True
)

task_manager = Agent(
    name="Task Manager",
    role="Task & Assignment Tracker",
    goal="Monitor assignments, deadlines, and tasks.",
    backstory="Built as a digital academic planner, the Task Manager is laser-focused on deadlines, due dates, and deliverables. With a sharp eye for detail and a structured mindset, it keeps students updated on assignments, ensures nothing is forgotten, and promotes timely submissions through gentle reminders and progress tracking.",
    llm=llm
)

stress_predictor = Agent(
    name="Stress Predictor",
    role="Workload & Stress Analyzer",
    goal="Analyze student workload and predict stress levels.",
    backstory="Developed from research into student mental health, the Stress Predictor is trained to identify signs of academic burnout, overload, and imbalance. It analyzes study patterns, workload, and emotional cues to offer actionable feedback and suggestions that promote mental wellness and productivity harmony.",
    llm=llm
)

adaptive_scheduler = Agent(
    name="Adaptive Scheduler",
    role="Smart Study Planner",
    goal="Modify study plans based on workload and stress levels.",
    backstory="Created as a dynamic planning engine, the Adaptive Scheduler learns from your performance and stress signals to update your study timetable in real time. It’s like having a personal assistant who not only plans but also adapts to your needs, ensuring you're always productive without being overwhelmed.",
    llm=llm
)

teacher_agent = Agent(
    name="Teacher Bot",
    role="Academic Concept Explainer",
    goal="Explain complex concepts in simple terms.",
    backstory="Inspired by the world's best educators, Teacher Bot is a patient and knowledgeable AI tutor designed to break down complex concepts into simple, digestible explanations. Whether it’s advanced math or abstract theory, it teaches in a way that makes learning approachable and enjoyable.",
    llm=llm,
    verbose=True
)

motivator_agent = Agent(
    name="Motivator",
    role="Positive Reinforcement Bot",
    goal="Encourage and motivate students with uplifting responses.",
    backstory="Cheerful, energetic, and always optimistic, the Motivator was created to lift spirits and restore focus during tough study days. Drawing from psychology and positive reinforcement techniques, it delivers encouragement, affirmations, and pep talks to keep students moving forward with confidence.",
    llm=llm,
    verbose=True
)

# Request & Response models
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatHistoryResponse(BaseModel):
    user_id: str
    messages: List[dict]

@app.get("/")
async def root():
    return {"status": "Backend is active"}

@app.delete("/chat/history/{user_id}")
async def delete_chat_history(user_id: str):
    try:
        result = messages_collection.delete_many({"user_id": user_id})
        return {
            "message": "Chat history deleted successfully.",
            "deleted_count": result.deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# API endpoint for chat requests
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message.lower()

    if any(keyword in user_input for keyword in ["schedule", "plan", "study plan"]):
        agent = chatbot_agent
        description = f"Create a study schedule based on this request: {request.message}"
        expected_output = "A well-structured study plan."

    elif any(keyword in user_input for keyword in ["assignment", "deadline", "due", "task"]):
        agent = task_manager
        description = f"Check for pending assignments based on this request: {request.message}"
        expected_output = "A list of pending assignments."

    elif any(keyword in user_input for keyword in ["stress", "overwork", "burnout", "overwhelm"]):
        agent = stress_predictor
        description = f"Analyze workload and predict stress levels based on this request: {request.message}"
        expected_output = "A stress level assessment."

    elif any(keyword in user_input for keyword in ["explain", "concept", "definition", "theory", "understand"]):
        agent = teacher_agent
        description = f"Explain this academic concept: {request.message}"
        expected_output = "A clear and concise explanation of the topic."

    elif any(keyword in user_input for keyword in ["motivate", "encourage", "feeling low", "positive", "inspire"]):
        agent = motivator_agent
        description = f"Give an uplifting and positive message for this request: {request.message}"
        expected_output = "A motivating and cheerful response."

    else:
        agent = chatbot_agent
        description = f"Respond to this student query: {request.message}"
        expected_output = "A helpful response."

    # Create a task for the agent
    current_task = Task(description=description, agent=agent, expected_output=expected_output)
    crew = Crew(agents=[agent], tasks=[current_task])

    try:
        response = await asyncio.create_task(crew.kickoff_async())
        bot_response = response.raw

        # Store messages in MongoDB
        messages_collection.insert_many([
            {"user_id": request.user_id, "role": "user", "text": request.message},
            {"user_id": request.user_id, "role": "bot", "text": bot_response}
        ])

        return {"response": bot_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Retrieve chat history
@app.get("/chat/history/{user_id}")
async def get_chat_history(user_id: str):
    try:
        messages = list(messages_collection.find({"user_id": user_id}, {"_id": 0}))
        return {"user_id": user_id, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
