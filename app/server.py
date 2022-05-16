import asyncio
import sys
from io import BytesIO

import uvicorn
from fastai import *
from fastai.vision import *
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles
import pandas as pd


disease_info = pd.read_csv('app/models/disease_info.csv', encoding='cp1252')

export_file_name = "export_resnet34_model.pkl"
export_file_path = "app/models"

classes = {'Background_Without_Leaf': 0,
           'Tomato_Bacterial_spot': 1,
           'Tomato_Early_blight': 2,
           'Tomato_Late_blight': 3,
           'Tomato_Leaf_Mold': 4,
           'Tomato_Septoria_leaf_spot': 5,
           'Tomato_Spider_mites_Two_spotted_spider_mite': 6,
           'Tomato__Target_Spot': 7,
           'Tomato__Tomato_YellowLeaf__Curl_Virus': 8,
           'Tomato__Tomato_mosaic_virus': 9,
           'Tomato_healthy': 10}


app = Starlette()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["X-Requested-With", "Content-Type"],
)
app.mount("/static", StaticFiles(directory="app/static"))


async def setup_learner():
    try:
        learn = load_learner(export_file_path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and "CPU-only machine" in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()
index_path = Path(__file__).parent


@app.route("/")
async def homepage(request):
    index_file = index_path / "view" / "index.html"
    return HTMLResponse(index_file.open().read())


@app.route("/analyze", methods=["POST"])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data["file"].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    pred = classes[str(prediction)]
    title = disease_info['disease_name'][pred]
    description =disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    return JSONResponse({"result":title,"description":description,"prevention":prevent})

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
