# main.py
from flask import Flask, request, Response
import Net.net as Net
import json

import torch
import infer


app = Flask(__name__)

model = None
use_gpu = True
MODEL = 'E:/ai/models/ctpn-msra_ali-9-end.model'

def load_model():
    """Load the pre-trained model, you can use your model just as easily.
    """
    global model
    model = Net.CTPN()
    model.load_state_dict(torch.load(MODEL))
    model.eval()

def infer_one11(im_name, net):
    infer.infer_one(im_name, net)

@app.route('/predict', methods=["GET"])
def index():
    data = request.args.get('data')
    print("read data",data)
    infer_one11(data,model)
    return 'this server is running on port:6001, url is predict'
    # 在index函数就可以添加任何操作，如机器学习/深度学习模型操作等

if __name__=='__main__':
    app.debug = True
    load_model()
    app.run(host="0.0.0.0", port=6001)