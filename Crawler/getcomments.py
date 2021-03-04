import platform
import os.path
import logging
import datetime
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

# https://www.facebook.com/eusuntdorianpopa/photos/a.402768046489498/3257697124329895/
# https://www.facebook.com/klausiohannis/photos/pcb.3654493634637861/3654541041299787/

msg = ""
accountno = ""
browser = None

def print_v(arg):
    # logging.info("[" + str(datetime.datetime.utcnow()) + "] " + accountno + " " + str(arg))
    if "--verbose" in sys.argv: 
        print(arg)


def dumpFile(title, content):
    if (not DEBUG_MODE): return
    if (os.path.exists("./DEBUG") == False):
        os.mkdir("./DEBUG")

    filename = title + " " + str(datetime.datetime.utcnow()) + ".dump.html"
    filename = filename.replace(":",".").replace(" ","-")

    # https://regex101.com/r/VjbmPV/2
    contentClean = content.get_attribute("outerHTML")
    pattern = re.compile("<script[\s\S]+?\/script>")
    subst = u""
    contentClean = re.sub(pattern, subst, contentClean)

    if platform.system() == "Windows":
        dump = io.open("./DEBUG/" + filename , "w", encoding="utf-8")
    if platform.system() == "Darwin":
        dump = io.open("./DEBUG/" + filename , "w", encoding="utf-8")
    if platform.system() == "Linux":
        dump = io.open("./DEBUG/" + filename , "w", encoding="utf-8")
    dump.write(contentClean)
    dump.close()

def startFirefox(url):
    options = webdriver.FirefoxOptions()
    if not "--GUI" in sys.argv: options.add_argument("--headless")
    if platform.system() == "Windows":
        browser = webdriver.Firefox(options=options, executable_path='./drivers/win/geckodriver')
    if platform.system() == "Darwin":
        browser = webdriver.Firefox(options=options, executable_path='./drivers/mac/geckodriver')
    if platform.system() == "Linux":
        firefox_binary = FirefoxBinary('/opt/firefox/firefox')
        browser = webdriver.Firefox(firefox_binary=firefox_binary, options=options, executable_path='./drivers/linux/geckodriver')

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
    time.sleep(random.randint(2,8))
    
def simulateHumanMedium():
    time.sleep(random.randint(6,14))
    
def simulateHumanLong():
    time.sleep(random.randint(15,35))
    
def browseURL(url, user, pwd):
    """Browse to Page"""

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
            expected_conditions.element_to_be_clickable((By.CSS_SELECTOR,"div[role=complementary]"))
        )
    except Exception as e:
        print_v("An exception has occured during page load(1): " + str(type(e)) + ' ' + str(e) )

    # # hide login overlay
    # login_overlay = browser.find_element_by_xpath("//div[@data-pagelet='page']/div[2]/div")
    # browser.execute_script("arguments[0].style.display = 'none';", login_overlay)

    # # # hide cookie banner
    # # cookie_overlay = browser.find_element_by_xpath("//div[@data-cookiebanner='banner']")
    # # browser.execute_script("arguments[0].style.display = 'none';", cookie_overlay)
    # cookie_overlay_btn = browser.find_element_by_xpath("//div[@aria-label='Close cookie banner button']")
    # cookie_overlay_btn.click()
    simulateHumanMedium()
    browser.find_element_by_xpath("//input[@aria-label='Email or Phone']").send_keys(user)
    simulateHumanShort()
    browser.find_element_by_xpath("//input[@aria-label='Password']").send_keys(pwd)
    simulateHumanShort()
    browser.find_element_by_xpath("//div[@aria-label='Accessible login button']").click()

    # wait for page to load
    try:
        WebDriverWait(browser, 20).until(
            expected_conditions.element_to_be_clickable((By.CSS_SELECTOR,"div[role=complementary]"))
        )
    except Exception as e:
        print_v("An exception has occured during page load(2): " + str(type(e)) + ' ' + str(e) )

    comments_box = browser.find_element_by_css_selector("div[role=complementary]")

    # wait for page to load
    try:
        WebDriverWait(browser, 20).until(
            expected_conditions.element_to_be_clickable((By.XPATH,"//ul/preceding-sibling::div//div[@role='button']"))
        )
    except Exception as e:
        print_v("An exception has occured during page load(3): " + str(type(e)) + ' ' + str(e) )

    # select all comments
    simulateHumanMedium()
    relevant_filter = comments_box.find_element_by_xpath("//ul/preceding-sibling::div//div[@role='button']")
    relevant_filter.click()

    simulateHumanShort()
    relevant_filter.find_element_by_xpath("//div[@role='menuitem'][3]").click()

    # wait for page to load
    try:
        WebDriverWait(browser, 20).until(
            expected_conditions.element_to_be_clickable((By.XPATH,"//ul/following-sibling::div//div[@role='button']"))
        )
    except Exception as e:
        print_v("An exception has occured during page load(4): " + str(type(e)) + ' ' + str(e) )

    lastScrappedComment = 0
    # loop until all messages are shown
    while True:
        comments = comments_box.find_elements_by_xpath("./div/div/div/div/div[4]/ul/li")
        # comments = comments_box.find_elements_by_tag_name("li")
        extractContent(comments[lastScrappedComment:])
        lastScrappedComment = len(comments)

        # click "View more comments" button, if available
        more_comments_btn = comments_box.find_element_by_xpath("//ul/following-sibling::div//div[@role='button']")
        visible_messages = more_comments_btn.find_element_by_xpath("./../following-sibling::div").text

        if visible_messages == "":
            break

        if " of " in visible_messages:
            parts = visible_messages.split(" of ")
            if parts[0] == parts[1]:
                break
        
        browser.execute_script("arguments[0].scrollIntoView(true);", more_comments_btn)
       
        loaded_messages = len(comments_box.find_elements_by_tag_name("ul"))
        # more_comments_btn.click()
        # click element by executing javascript
        browser.execute_script("arguments[0].click();", more_comments_btn)

        # wait for messages to load
        while len(comments_box.find_elements_by_tag_name("ul")) == loaded_messages:
            None

    # dumpFile(title="after loading", content=browser.find_element_by_tag_name("html"))
    return browser

def extractContent(comments):
    global data

    for comment in comments:
        simulateHumanShort()
        print_v("============================= new message ================================")
        reactions = 0
        see_more_btns = comment.find_elements_by_xpath(".//div[@role='button']")
        for see_more_btn in see_more_btns:
            print_v(see_more_btn.text)
            if see_more_btn.text == "See More":
                # expand message body
                browser.execute_script("arguments[0].scrollIntoView(true);", see_more_btn)
                browser.execute_script("arguments[0].click();", see_more_btn)
                simulateHumanLong()

            if " Repl" in see_more_btn.text:
                # expand replies
                # browser.execute_script("arguments[0].scrollIntoView(true);", see_more_btn)
                # browser.execute_script("arguments[0].click();", see_more_btn)
                simulateHumanMedium()
            
            if see_more_btn.text.isdigit():
                reactions = see_more_btn.text
                print_v("Reactions:" + see_more_btn.text)

        print_v(comment.get_attribute("innerText"))

        # save to file every time, not to loose the progress
        data += [[comment.get_attribute("innerText"),reactions]]
        df = pd.DataFrame(data, columns=["raw text","reactii"])
        df.to_csv('list.csv', index=False)

def main():
    url = ""
    for arg in sys.argv:
        if "--url=" in arg:
            url = arg[len("--url="):]

    if url == "":
        print_v("Insufficient arguments")
        return

    print_v("\n======= Script started with arguments " + str(sys.argv) + " =========")
    global msg
    
    # # store facebook credentials in pickle file. Do this once, to create the pickle file, then comment the following 3 lines of code
    # fb_credentials = open("../../fb.pickle","wb")
    # creds = ["inti80quila@gmail.com","EcwRJZE2H72rcsuUmJWG"]
    # pickle.dump(creds, fb_credentials)

    fb_credentials = open("../../fb.pickle","rb")
    user, pwd = pickle.load(fb_credentials)

    browser =  browseURL(url, user, pwd)
    if browser == None:
        return


    # finito
    browser.quit()
    print_v("======= Script ended =========")

data = []

sPath = ""

logging.basicConfig(filename=sPath + 'debug.log', level=logging.INFO)

try:
    main()
except Exception as e:
    logging.critical(e, exc_info=True)
    if not browser is None:
        browser.quit()
        print_v("======= Script ended with error =========")