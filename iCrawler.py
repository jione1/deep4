# iCrawler documentation : http://icrawler.readthedocs.io/en/latest/
# 구글뿐 아니라 Baidu, Bing, Flickr도 지원함!

from icrawler.builtin import GoogleImageCrawler
import os

def googleCrawl(name, image_dir):
   if name not in image_dir:
       try:
           os.mkdir(image_dir+"\\"+name)
       except:
           pass
   google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                       storage={'root_dir': image_dir+"\\"+name})
   google_crawler.crawl(keyword=name, max_num=500,
                        date_min=None, date_max=None,
                        min_size=(200,200), max_size=None)


googleCrawl("seolhyun", 'pic')