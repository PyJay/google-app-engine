from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

model_file_url = 'https://www.googleapis.com/drive/v3/files/1P6JmhkN_PNFVpgJlmstL_yHlqlirPrnK?alt=media&key=AIzaSyBnxRpJjjqa_q7kCVB9o0im44yLzWqnrPc'
model_file_name = 'export.pkl'
classes = ['potato', 'tomato', 'gelato']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/rf'models/{model_file_name}')
    learn = load_learner(path/'models', model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)
    probability = float(prediction[2][prediction[1]]) * 100
    if  probability >= 50:
        response = f'{prediction[0]}! - with a probability of {probability}%'
    else:
        response = f'None of the above - the photo is closest to {prediction[0]} with a probability of {probability}%'
    return JSONResponse({'result': response})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

