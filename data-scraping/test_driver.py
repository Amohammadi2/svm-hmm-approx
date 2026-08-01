"""
Use this script to test your selenium driver setup before running "scrapery.py"
"""


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager

# Setup
service = Service(GeckoDriverManager().install())
driver = webdriver.Firefox(service=service)

try:
    # Navigate
    driver.get("https://www.google.com")
    
    # Find search box and type
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys("Selenium Firefox")
    search_box.submit()
    
    # Wait for results (implicitly or explicitly)
    driver.implicitly_wait(3)
    
    # Print page title
    print("Page title:", driver.title)
finally:
    driver.quit()