import time, json, requests

url = "http://127.0.0.1:9000/stream_chat"
data = {
    "query" : "Is 3M a capital-intensive business based on FY2022 data?",
    "system_msg": "You are a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown"
    }

headers = {"Content-type": "application/json"}

st = time.time()
with requests.post(url, data=json.dumps(data), headers=headers, stream=True) as r:
    if r.status_code == 200 : 
        for chunk in r.iter_content(2048, decode_unicode=True):
            # print(chunk)
            if(chunk != None):

                print(chunk, end="")
            else:
                print("internal ", r.status_code)
                print(chunk)
    else:
        print(r.json())


print(time.time()-st)
