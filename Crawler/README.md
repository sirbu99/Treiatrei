# Facebook comments crawler
 The script scrapes the text contents from urls of fb images with comments

# Instalation
Preferably use Anaconda to set up Python 3.7 environment.  
Install Selenium with:  
`pip install -U selenium`  
or  
`pip3 install -U selenium`  

Must have Mozilla Firefox installed

Optional:  
Download the latest Firefox Driver for the repective platform from https://github.com/mozilla/geckodriver/releases and replace the drivers in the ./drivers folder

On linux based systems, give execution permission to _geckodriver_ in ./drivers folder.

## Program arguments:
**MANDATORY --url=** must be a facebook page with comments to a picture i.e. https://www.facebook.com/klausiohannis/photos/pcb.3654493634637861/3654541041299787/  
**--GUI** browse in GUI mode  
**--verbose** show info for each step  

