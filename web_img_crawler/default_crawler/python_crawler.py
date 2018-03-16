# 1.필요한 모듈 import
from selenium import webdriver
import urllib.request
from bs4 import BeautifulSoup
# 2. run the Chrome webdriver
driver = webdriver.Chrome('C:\python_workspace\web_img_crawler\chrome_driver\chromedriver.exe')
driver.get('https://www.google.co.kr/imghp?hl=ko')

# 3. img_search
keyword = input("검색할 이미지를 입력하세요 : ")

# 4. search
elem = driver.find_element_by_name("q")
elem.clear()
elem.send_keys(keyword)
elem.submit()

# 5. get the html source
html = driver.page_source
soup = BeautifulSoup(html,"lxml")
images = soup.find_all("img")

# 6.download images
for i,k in zip(images, range(len(images))):
    if i.get('data-src') is not None:
        if images[k].get('data-src')[0] is "h":
            print(i.get('data-src'),"\n")
            name = k
            full_name = str(name)+".jpg"
            urllib.request.urlretrieve(i.get('data-src'),full_name)
            if k==10: break;
