from fastapi import FastAPI

app = FastAPI()


@app.get("/root/{name}")
async def root(name):
    return f"welcome {name}"
