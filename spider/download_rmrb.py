import argparse
import random
import shutil

import requests
import bs4
import os
import time


def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def fetchUrl(url):
    '''
    功能：访问 url 的网页，获取网页内容并返回
    参数：目标网页的 url
    返回：目标网页的 html 内容
    '''

    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return r.text


def getPageList(year, month, day):
    '''
    功能：获取当天报纸的各版面的链接列表
    参数：年，月，日
    '''
    url = 'http://paper.people.com.cn/rmrb/html/' + year + '-' + month + '/' + day + '/nbs.D110000renmrb_01.htm'
    html = fetchUrl(url)
    bsobj = bs4.BeautifulSoup(html, 'html.parser')
    pageList = bsobj.find('div', attrs={'id': 'pageList'}).ul.find_all('div', attrs={'class': 'right_title-name'})
    linkList = []

    for page in pageList:
        link = page.a["href"]
        url = 'http://paper.people.com.cn/rmrb/html/' + year + '-' + month + '/' + day + '/' + link
        linkList.append(url)

    return linkList


def getTitleList(year, month, day, pageUrl):
    '''
    功能：获取报纸某一版面的文章链接列表
    参数：年，月，日，该版面的链接
    '''
    html = fetchUrl(pageUrl)
    bsobj = bs4.BeautifulSoup(html, 'html.parser')
    titleList = bsobj.find('div', attrs={'id': 'titleList'}).ul.find_all('li')
    linkList = []

    for title in titleList:
        tempList = title.find_all('a')
        for temp in tempList:
            link = temp["href"]
            if 'nw.D110000renmrb' in link:
                url = 'http://paper.people.com.cn/rmrb/html/' + year + '-' + month + '/' + day + '/' + link
                linkList.append(url)

    return linkList


def getContent(html):
    '''
    功能：解析 HTML 网页，获取新闻的文章内容
    参数：html 网页内容
    '''
    bsobj = bs4.BeautifulSoup(html, 'html.parser')

    # 获取文章 标题
    title = bsobj.h3.text + '\n' + bsobj.h1.text + '\n' + bsobj.h2.text + '\n'
    # print(title)

    # 获取文章 内容
    pList = bsobj.find('div', attrs={'id': 'ozoom'}).find_all('p')
    content = ''
    for p in pList:
        content += p.text + '\n'
        # print(content)

    # 返回结果 标题+内容
    resp = title + content
    return resp


def saveFile(content, path, filename, keysStat, arg):
    '''
    功能：将文章内容 content 保存到本地文件中
    参数：要保存的内容，路径，文件名
    '''
    # 如果没有该文件夹，则自动生成
    if not os.path.exists(path):
        os.makedirs(path)

    #只提取中文
    content_ch=''
    for ch in content:
        if is_Chinese(ch):
            if arg.trans:
                # ch=trans_by_zhtools(ch)
                pass
            content_ch+=ch
            keysStat.add(ch)

    # 保存文件
    with open(path + filename, 'w', encoding='utf-8') as f:
        f.write(content_ch)


def download_rmrb(year, month, day, destdir,keysStat, arg):
    '''
    功能：爬取《人民日报》网站 某年 某月 某日 的新闻内容，并保存在 指定目录下
    参数：年，月，日，文件保存的根目录
    '''
    pageList = getPageList(year, month, day)
    for page in pageList:
        titleList = getTitleList(year, month, day, page)
        for url in titleList:
            # 获取新闻文章内容
            html = fetchUrl(url)
            content = getContent(html)

            # 生成保存的文件路径及文件名
            temp = url.split('_')[2].split('.')[0].split('-')
            pageNo = temp[1]
            titleNo = temp[0] if int(temp[0]) >= 10 else '0' + temp[0]
            path = destdir + '/' + year + month + day + '/'
            fileName = year + month + day + '-' + pageNo + '-' + titleNo + '.txt'

            # 保存文件
            saveFile(content, path, fileName, keysStat, arg)

        #     break
        # break


class KeysStat():
    def __init__(self):
        self.keys=set()
        pass

    def add(self,ch):
        self.keys.add(ch)

    def finish(self):
        keys=list(self.keys)
        keys.sort()
        print('number of keys:',len(keys))
        keys=''.join(keys)
        print(keys)
        with open('keys.txt', 'w', encoding='utf-8') as f:
            f.write(keys)

if __name__ == '__main__':
    '''
    主函数：程序入口
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans', type=bool, default=False, help='transform to traditinal chinese charactor')
    parser.add_argument('--num', type=int, default=100, help='numbers of date to download')
    parser.add_argument('--destdir', type=str, default='data_test', help='')
    arg = parser.parse_args()
    keysStat=KeysStat()

    destdir = arg.destdir
    # if os.path.exists(destdir):
    #     shutil.rmtree(destdir)
    download_list=[]
    year_range=(2019,2019)
    month_range=(1,12)
    day_range=(1,28)
    index=0
    while index<arg.num:
        year = random.randint(year_range[0],year_range[1])
        month = random.randint(month_range[0],month_range[1])
        day = random.randint(day_range[0],day_range[1])
        if (year,month,day) in download_list:
            continue
        else:
            index+=1
            print("[%5d][%5d]爬取开始："%(index,arg.num), str(year), '%02d' % (month), '%02d' % (day))
            download_rmrb(str(year), '%02d'%(month), '%02d'%(day), destdir, keysStat, arg)
            print("[%5d][%5d]爬取完成："%(index,arg.num),str(year), '%02d'%(month), '%02d'%(day))
            time.sleep(3)

    # download_rmrb(year, month, day, destdir, keysStat, arg)
    # print("爬取完成：" + year + month + day)
    keysStat.finish()

