from fastapi import FastAPI

app = FastAPI()
@app.get("/Home")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/POST")
def post():
    return {"message": "This is a post request"}
