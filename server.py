from flask import Flask, jsonify, make_response, request
import torch
import numpy as np
import json

app = Flask(__name__, static_folder='/', static_url_path='/')

def run_flask():
  app.run(host='0.0.0.0', port=7777, debug=True)

@app.route('/filter', methods=['POST'])
def filter():
  form = dict(request.form)
  # print(form)
  mat = torch.tensor(json.loads(form['conf_mat'])).long()
  thres = torch.tensor(int(form['threshold'])).long()
  name = json.loads(form['classname'])
  count = range(mat.shape[0])
  mat[count,count] = 0 # set diag to 0
  mat[mat<thres] = 0

  toDelete = []
  for i in range(mat.shape[0]):
    largeNum = (mat[i] > 0).sum() + (mat[:,i] > 0).sum()
    if largeNum.item() == 0:
      toDelete.append(i)

  while True:
    update = False
    for i in range(mat.shape[0]):
      largeNum = (mat[i] > 0).sum() + (mat[:,i] > 0).sum() # 统计与其相关的大于0的数
      if largeNum.item() == 0: # 没有大于0的数就将其删除
        # delete row and col
        ii = torch.arange(mat.shape[0]) != i
        mat = mat[ii][:,ii]
        name.remove(name[i])
        assert len(name) == mat.shape[0]
        update = True
        break
    if update:
      continue
    else:
      break
  print(mat)
  return make_response(jsonify({
    'conf_mat': json.dumps(mat.tolist()), 
    'classname': json.dumps(name),
    'toDelete': json.dumps(toDelete)
    }), 200)

if __name__ == '__main__':
  run_flask()