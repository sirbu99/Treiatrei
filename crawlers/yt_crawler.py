import platform
import logging
import sys
import io
import re
import ssl
import pickle
import time
import random
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select

DEBUG_MODE = True

msg = ""
accountno = ""
browser = None
comment_index = -1
browser = None

def print_v(arg):
    # logging.info("[" + str(datetime.datetime.utcnow()) + "] " + accountno + " " + str(arg))
    if "--verbose" in sys.argv: 
        print(arg)


def startFirefox(url):
    options = webdriver.FirefoxOptions()
    # if not "--GUI" in sys.argv: options.add_argument("--headless")
    if platform.system() == "Windows":
        browser = webdriver.Firefox(options=options, executable_path='./crawlers/Crawler - youtube/drivers/win/geckodriver')
    if platform.system() == "Darwin":
        browser = webdriver.Firefox(options=options, executable_path='./crawlers/Crawler - youtube/drivers/mac/geckodriver')
    if platform.system() == "Linux":
        firefox_binary = FirefoxBinary('/opt/firefox/firefox')
        browser = webdriver.Firefox(firefox_binary=firefox_binary, options=options, executable_path='./crawlers/Crawler - youtube/drivers/linux/geckodriver')

    try:
        # open dashboard
        browser.get(url)
        # print_v("Page loaded: '{}' [".format(browser.title) + browser.current_url +']')

    except:
        browser.quit()
        print_v("An exception has occured while starting firefox")
        return None
    
    return browser

# add delay to bypass fb activity check
def simulateHumanShort():
    time.sleep(random.randint(1,2))
    
def simulateHumanMedium():
    time.sleep(random.randint(3,7))
    
def simulateHumanLong():
    time.sleep(random.randint(10,35))
    
def browseURL(url, user, pwd):
    """Browse to Page"""
    global data

    if "--GUI" in sys.argv:
        print_v("Starting Firefox w Selenium in GUI mode...")
    else:
        print_v("Starting Firefox w Selenium in headless mode...")

    global browser
    browser = startFirefox(url)
    if browser == None:
        return None

    # wait for page to load
    try:
        WebDriverWait(browser, 20).until(
            expected_conditions.element_to_be_clickable((By.XPATH,"//button/span[contains(text(), 'I agree')]"))
        )
    except Exception as e:
        print_v("An exception has occured during page load(1): " + str(type(e)) + ' ' + str(e) )

    # agree
    try:
        btnAgree = browser.find_element_by_xpath("//button/span[contains(text(), 'I agree')]")
        browser.execute_script("arguments[0].scrollIntoView(true);", btnAgree)
        browser.execute_script("arguments[0].click();", btnAgree)
    except Exception as e:
        print_v("An exception has occured during page load(1-1): " + str(type(e)) + ' ' + str(e) )

    # wait for page to load
    try:
        WebDriverWait(browser, 40).until(
            expected_conditions.visibility_of_element_located((By.XPATH,"//div[@id='contents']"))
        )
    except Exception as e:
        print_v("An exception has occured during page load(2): " + str(type(e)) + ' ' + str(e) )

    comment = browser.find_element_by_xpath("//div[@id='meta']")
    browser.execute_script("arguments[0].scrollIntoView(true);", comment)

    comments_found = 0
    global comments
    comments = browser.find_elements_by_xpath("//ytd-comment-thread-renderer")

    # while True:
    #     comments = browser.find_elements_by_xpath("//ytd-comment-thread-renderer")
    #     browser.execute_script("arguments[0].scrollIntoView(true);", comments[-1])
    #     simulateHumanShort()
    #     if comments_found == len(comments):
    #         break
    #     comments_found = len(comments)


    # for comment in comments:
        # replies = comment.find_element_by_xpath("//div[@id='replies']//div[@id='expander']")
        # simulateHumanShort()

    # df = pd.DataFrame(data, columns=["raw text", "text_length", "source"])
    # df.to_csv('data/list.csv', index=False)

    # dumpFile(title="after loading", content=browser.find_element_by_tag_name("html"))
    return browser

def highlight_comment(index, color):
    browser.execute_script("arguments[0].setAttribute('style', 'background-color: " + color + "')", comments[index])

def next_yt_comment():
    global comments
    global comment_index

    comment_index += 1
    if comment_index >= len(comments):
        comments = browser.find_elements_by_xpath("//ytd-comment-thread-renderer")
        browser.execute_script("arguments[0].scrollIntoView(true);", comments[-1])
        simulateHumanShort()

    if comment_index >= len(comments):
        return "", -1

    comment = comments[comment_index]

    body = comment.find_element_by_xpath(".//div[@id='main']//ytd-expander[@id='expander']")
    content = body.find_element_by_xpath(".//div[@id='content']")

    try:
        show_more = body.find_element_by_xpath(".//paper-button[@id='more']")
        if body.find_element_by_xpath(".//paper-button[@id='more']").get_attribute("hidden") == None:
            browser.execute_script("arguments[0].click();", show_more)
            simulateHumanMedium()
    except:
        None

    return content.text, comment_index

    # print_v(content.text)
    # data += [[content.text, len(content.text), url]]

def start_crawler(url):
    # url = ""
    # for arg in sys.argv:
    #     if "--url=" in arg:
    #         url = arg[len("--url="):]

    # if url == "":
    #     print_v("Insufficient arguments")
    #     return

    # print_v("\n======= Script started with arguments " + str(sys.argv) + " =========")
    global msg
    global comment_index
    comment_index = -1

    # # store facebook credentials in pickle file. Do this once, to create the pickle file, then comment the following 3 lines of code

    # fb_credentials = open("../..//crawlers/Crawler - youtube/fb.pickle","rb")
    user, pwd = None, None #pickle.load(fb_credentials)

    browser =  browseURL(url, user, pwd)
    if browser == None:
        return


    # # finito
    # browser.quit()
    # print_v("======= Script ended =========")

data = []

sPath = ""

logging.basicConfig(filename=sPath + 'debug.log', level=logging.INFO)

# try:
#     main()
# except Exception as e:
#     logging.critical(e, exc_info=True)
#     if not browser is None:
#         browser.quit()
#         print_v("======= Script ended with error =========")