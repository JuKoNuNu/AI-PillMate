import requests

NAVER_API_ID = "i4pQ74Kpc1G1WoFogoGk"
NAVER_API_SECRET = "J32tneBOhF"

def naver_shopping_search(query):
    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": NAVER_API_ID,
        "X-Naver-Client-Secret": NAVER_API_SECRET,
    }
    params = {
        "query": query,
        "display": 5,
        "start": 1,
        "sort": "sim"
    }

    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print("네이버 쇼핑 API 오류:", response.status_code)
        return None

# 사용 예
result = naver_shopping_search("감기약")
if result:
    for item in result.get("items", []):
        print(item["title"], item["link"])