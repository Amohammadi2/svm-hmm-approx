"""
Scrape historical price data for Iranian 18K gold from TGJU.

Overview
--------
This script uses Selenium WebDriver (Firefox) to automate a browser,
navigate through the paginated historical price table on TGJU, and
collect historical daily price records.

The website loads table data dynamically (AJAX), therefore a normal
HTTP request is insufficient. Selenium is used to interact with the
website exactly as a user would.

Source
------
https://www.tgju.org/profile/tgju_gold_irg18/history

Data Collected
--------------
Each row corresponds to one trading day and contains the following
columns (from left to right):

    1. Reopened          : Opening price
    2. Lowest            : Lowest price
    3. Highest           : Highest price
    4. Final             : Closing (final) price
    5. Change_Amount     : Absolute price change
    6. Change_Percent    : Percentage price change
    7. Date_Miladi       : Gregorian date
    8. Date_Jalali       : Jalali (Persian) date

Missing Values
--------------
Some table cells may be empty.

Rather than failing, empty strings are converted to None so the data
can later be cleaned or imputed during preprocessing.

Output
------
All scraped rows are collected into a pandas DataFrame and exported to

    gold18_history.csv

using UTF-8 with BOM encoding (`utf-8-sig`) to preserve Persian text
compatibility in spreadsheet software such as Microsoft Excel.

Dependencies
------------
Python packages:

    selenium
    webdriver-manager
    pandas

Install with:

    pip install selenium webdriver-manager pandas

Browser Requirements
--------------------
Firefox must be installed.

The GeckoDriver executable is automatically downloaded and managed by
webdriver-manager, so no manual driver installation is required.

Limitations
-----------
- The script relies on the current HTML structure of TGJU.
  Changes to CSS selectors or page layout may require updates.

- The script assumes the pagination contains at least the requested
  number of pages.

- This scraper intentionally performs browser automation rather than
  reverse-engineering TGJU's internal API.

Author Notes
------------
This script is intended for historical data collection and research.
It is designed to be simple, readable, and robust against asynchronous
page loading while avoiding unnecessary DOM traversals.
"""


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
import pandas as pd
import time


URL = "https://www.tgju.org/profile/tgju_gold_irg18/history"

# Setup
service = Service(GeckoDriverManager().install())
driver = webdriver.Firefox(service=service)
wait = WebDriverWait(driver, 20)

driver.get(URL)

# ----------------------------
# Wait until table is present
# ----------------------------
wait.until(
    EC.presence_of_element_located(
        (By.CSS_SELECTOR, "table.table")
    )
)

all_rows = []


def extract_current_page():
    """
    Extract one page of the history table.
    Missing cells are replaced with None.
    """


    table = wait.until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "table.table")
        )
    )

    tbody = table.find_element(By.CSS_SELECTOR, "tbody#table-list")
    rows = tbody.find_elements(By.TAG_NAME, "tr")

    page_data = []

    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")

        values = []

        for cell in cells:
            txt = cell.text.strip()
            values.append(txt if txt != "" else None)

        page_data.append(values)

    return page_data, rows[0]

for page in range(12):

    print(f"Reading page {page + 1}")

    data, first_row = extract_current_page()
    all_rows.extend(data)

    if page == 11:
        break

    next_button = wait.until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "a.paginate_button.next")
        )
    )

    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center'});",
        next_button,
    )

    next_button.click()

    wait.until(EC.staleness_of(first_row))
    time.sleep(1)

driver.quit()

# -----------------------------------------
# Make all rows same length
# -----------------------------------------
max_cols = max(len(r) for r in all_rows)

normalized = [
    r + [None] * (max_cols - len(r))
    for r in all_rows
]

columns = [f"column_{i+1}" for i in range(max_cols)]

df = pd.DataFrame(normalized, columns=columns)

print(df.head())

df.to_csv("gold18_history.csv", index=False, encoding="utf-8-sig")

print(f"\nCollected {len(df)} rows.")