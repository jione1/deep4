# iCrawler documentation : http://icrawler.readthedocs.io/en/latest/
# 구글뿐 아니라 Baidu, Bing, Flickr도 지원함!

from icrawler.builtin import GoogleImageCrawler, BaiduImageCrawler
import os


def googleCrawl(name, image_dir):
   if name not in image_dir:
       try:
           os.mkdir(image_dir+"\\"+name)
       except:
           pass
   google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                       storage={'root_dir': image_dir+"\\"+name})
   google_crawler.crawl(keyword=name, max_num=1200,
                        min_size=(64,64), max_size=None)

googleCrawl("carrot", 'food')

'''
def baiduCrawl(name, image_dir):
   if name not in image_dir:
       try:
           os.mkdir(image_dir+"\\"+name)
       except:
           pass
   baidu_crawler = BaiduImageCrawler(parser_threads=4, downloader_threads=8,
                                       storage={'root_dir': image_dir+"\\"+name})
   baidu_crawler.crawl(keyword=name, max_num=1200,
                        min_size=(64,64), max_size=None)

baiduCrawl("紅蘿蔔", 'food')
'''