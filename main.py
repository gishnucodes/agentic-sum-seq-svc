from crewai.cli.cli import crewai
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
import logging

from starlette.responses import JSONResponse

## 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
# Initialize FastAPI app

GROQ_API_KEY="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"  # On Windows use `set GROQ_API_KEY=your_api_key_here`

app = FastAPI()

# Enable CORS
origins = [
    "*"
]
CORSMiddleware(app, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# Initialize Groq LLM
llm = LLM(model="groq/llama-3.3-70b-versatile", api_key='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')


# Define request model
class DocumentRequest(BaseModel):
    text: str


# Define CrewAI Agents
segregator = Agent(
    role='Text Segregator',
    goal='Analyze and divide text {input} into meaningful sections with appropriate headings.',
    backstory='An expert in document structuring and organization.',
    llm=llm,
    verbose=True
)

summarizer = Agent(
    role='Summarizer',
    goal='Generate a detailed summary of the given document.',
    backstory='A technical writer skilled in summarization and knowledge extraction.',
    llm=llm,
    verbose=True
)

combiner = Agent(
    role='Final Combiner',
    goal='Merge the structured text and summary into a final comprehensive document.',
    backstory='A meticulous compiler ensuring clarity and coherence.',
    llm=llm,
    verbose=True
)

# Define CrewAI Tasks
segregation_task = Task(
    description='Analyze the given text, identify different sections, and provide appropriate headings.',
    expected_output='Structured text with section headings.',
    agent=segregator
)

summary_task = Task(
    description='Summarize the entire document while preserving key details.',
    expected_output='A detailed summary of the document.',
    agent=summarizer,
    dependencies=[segregation_task]
)

combine_task = Task(
    description='Combine the structured text and summary into one final document.',
    expected_output='A single document containing both the structured text and summary.',
    agent=combiner,
    dependencies=[segregation_task, summary_task]
)


# Handle preflight requests
# @app.options("/process_document")
# async def preflight_process_document():
#     return JSONResponse(content={"message": "OK"}, status_code=200)


# API Endpoint
# @app.post("/api/process", tags=["Document Processing"], summary="Process and summarize document",)
# def process_document(request):
#     """
#     Takes a raw text document, segments it into sections, summarizes it, and combines results.
#     """
#     print(request.text)
#     try:
#         if not request.text:
#             raise HTTPException(status_code=400, detail="Text input is empty")
#         if len(request.text) > 10000:
#             raise HTTPException(status_code=400, detail="Text input is too long")
#
#         # Create Crew and execute workflow
#         crew = Crew(
#             agents=[segregator, summarizer, combiner],
#             tasks=[segregation_task, summary_task, combine_task],
#             verbose=True
#         )
#         print(request)
#         result = crew.kickoff(inputs={f"input": f"{request.text}"})
#         print(result.raw)
#         return result.raw
#     except ValueError as e:
#         logging.error(f"ValueError: {e}")
#         raise HTTPException(status_code=400, detail="Invalid input data")
#     except Exception as e:
#         logging.exception(f"Unexpected error: {e}")
#         raise HTTPException(status_code=500, detail="An unexpected error occurred")

from fastapi import FastAPI, HTTPException, Body
import logging

app = FastAPI()

@app.post("/api/process", summary="Process and summarize document")
async def process_document(text: str = Body(..., embed=True)):
    """
    Takes a raw text document, segments it into sections, summarizes it, and combines results.
    """
    logging.info(f"Received text: {text}")
    try:
        if not text:
            raise HTTPException(status_code=400, detail="Text input is empty")
        if len(text) > 10000:
            raise HTTPException(status_code=400, detail="Text input is too long")

        # Create Crew and execute workflow
        crew = Crew(
            agents=[segregator, summarizer, combiner],
            tasks=[segregation_task, summary_task, combine_task],
            verbose=True
        )
        result = crew.kickoff(inputs={"input": text})
        return result.raw
    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail="Invalid input data")
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

@app.post("/api/map", tags=["Document Processing"], summary="Process and summarize document",)
def process_document(request: DocumentRequest):
    """
    Takes a raw text document, segments it into sections, summarizes it, and combines results.
    """
    # print(request)
    if not request.text:
        raise HTTPException(status_code=400, detail="Text input is empty")

# Swagger UI Configuration
@app.get("/openapi.json", include_in_schema=False)
def get_open_api_endpoint():
    return get_openapi(title="Document Processing API", version="1.0.0",
                       description="API for processing and summarizing documents using CrewAI and Groq.",
                       routes=app.routes)


@app.get("/docs", include_in_schema=False)
def get_documentation():
    """
    Redirects to the Swagger UI documentation.
    """
    return app.openapi()

# if __name__ == "__main__":
#     crew = Crew(
#         agents=[segregator, summarizer, combiner],
#         tasks=[segregation_task, summary_task, combine_task],
#         verbose=True
#     )
#     text = input()
#     result = crew.kickoff(inputs=({f"input": f"{text}"}))
#     print(result.raw)

## python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
# Run the FastAPI server