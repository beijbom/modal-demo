# During demo 
# curl -d '{"name": "Erik", "qty": 10}' -H "Content-Type: application/json" -X POST https://beijbom--web-api-py-f-dev.modal.run  
# Show auto update
# Show how to deploy

import modal
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "boto3")
app = modal.App("Echo-API")


class Item(BaseModel):
    name: str
    qty: int = 42


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def f(item: Item):
    import boto3
    # do things with boto3...
    return HTMLResponse(f"<html>Hello {item.name}!</html>")



