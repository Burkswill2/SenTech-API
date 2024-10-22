from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from .dependencies import get_query_token, get_token_header
# from .internal import admin
from routers import markowitz_model

# app = FastAPI(dependencies=[Depends(get_query_token)])
app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:3000",  # Example: your frontend URL
    # "https://yourdomain.com",  # Add your production URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specify the allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
app.include_router(markowitz_model.router)
# app.include_router(
#     admin.router,
#     prefix="/admin",
#     tags=["admin"],
#     # dependencies=[Depends(get_token_header)],
#     responses={418: {"description": "I'm a teapot"}},
# )


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}