import urllib.request
url = "https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/sft_config.py"
req = urllib.request.Request(url)
try:
    with urllib.request.urlopen(req) as response:
        content = response.read().decode('utf-8')
        print("chunked_lm_head" in content)
except Exception as e:
    print(e)
