from pathlib import Path
import logging
import time
from typing import Optional, List, Tuple
from bs4 import BeautifulSoup
import pandas as pd

from selenium import webdriver
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
    NoSuchElementException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TGJUScraper:
    """
    An enterprise-grade, resilient Selenium scraper for TGJU historical price data.
    Features atomic HTML extraction, content-signature state verification, and retries.
    """

    DEFAULT_URL = "https://www.tgju.org/profile/tgju_gold_irg18/history"
    DEFAULT_COLUMNS = [
        "Reopened",
        "Lowest",
        "Highest",
        "Final",
        "Change_Amount",
        "Change_Percent",
        "Date_Miladi",
        "Date_Jalali",
    ]

    def __init__(
        self,
        url: str = DEFAULT_URL,
        headless: bool = True,
        max_pages: int = 12,
        timeout: int = 20,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize scraper with automatic cache targeting relative to scraper.py location.
        """
        base_dir = Path(__file__).resolve().parent
        cache_dir = base_dir / "cache"
        self.filepath = cache_dir / "gold18_history.csv"

        self.url = url
        self.headless = headless
        self.max_pages = max_pages
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _get_driver(self) -> Tuple[webdriver.Firefox, WebDriverWait]:
        """Creates browser driver instance with timeout configuration."""
        options = FirefoxOptions()
        if self.headless:
            options.add_argument("--headless")

        service = Service(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=options)
        driver.set_page_load_timeout(self.timeout * 2)
        wait = WebDriverWait(driver, self.timeout)
        return driver, wait

    def _parse_table_html(self, html_content: str) -> List[List[Optional[str]]]:
        """Parse raw HTML string using BeautifulSoup safely outside the DOM."""
        soup = BeautifulSoup(html_content, "html.parser")
        rows = soup.find_all("tr")
        parsed_data = []

        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue
            values = [cell.get_text(strip=True) or None for cell in cells]
            parsed_data.append(values)

        return parsed_data

    def _extract_current_table_atomic(
        self, driver: webdriver.Firefox, wait: WebDriverWait
    ) -> Tuple[List[List[Optional[str]]], str]:
        """
        Atomic extraction of table DOM into static HTML to prevent StaleElementReference errors.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                tbody_elem = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "tbody#table-list"))
                )
                # Fetch outerHTML in one synchronous JS wire execution
                raw_html = driver.execute_script("return arguments[0].outerHTML;", tbody_elem)
                parsed_rows = self._parse_table_html(raw_html)

                if parsed_rows:
                    first_row_signature = "|".join(str(c) for c in parsed_rows[0])
                    return parsed_rows, first_row_signature

                logger.warning(f"Table extraction empty on attempt {attempt}/{self.max_retries}.")
            except (StaleElementReferenceException, NoSuchElementException, TimeoutException) as e:
                logger.warning(f"DOM transient error on extraction ({type(e).__name__}). Retrying {attempt}/{self.max_retries}...")
                time.sleep(self.retry_delay * attempt)

        return [], ""

    def _turn_page_resilient(
        self, driver: webdriver.Firefox, wait: WebDriverWait, current_signature: str
    ) -> bool:
        """
        Resiliently handles pagination using JS click fallbacks and content-signature verification.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                next_btn = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a.paginate_button.next"))
                )

                # Stop if 'next' button is marked disabled
                btn_class = next_btn.get_attribute("class") or ""
                if "disabled" in btn_class:
                    logger.info("Pagination end reached ('next' button disabled).")
                    return False

                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_btn)
                time.sleep(0.3)

                # Dual click attempt (Selenium Click -> JavaScript Click Fallback)
                try:
                    next_btn.click()
                except (ElementClickInterceptedException, StaleElementReferenceException):
                    driver.execute_script("arguments[0].click();", next_btn)

                # Verify page transition via row content signature change
                start_time = time.time()
                while time.time() - start_time < self.timeout:
                    _, new_sig = self._extract_current_table_atomic(driver, wait)
                    if new_sig and new_sig != current_signature:
                        return True
                    time.sleep(0.5)

                logger.warning(f"Page transition signature verification timed out (attempt {attempt}).")
            except Exception as e:
                logger.warning(f"Pagination error ({type(e).__name__}) on attempt {attempt}/{self.max_retries}: {e}")
                time.sleep(self.retry_delay * attempt)

        return False

    def scrape(self, save: bool = True) -> pd.DataFrame:
        """
        Main public scrape routine with overall error isolation and partial data preservation.
        """
        driver = None
        all_rows: List[List[Optional[str]]] = []

        try:
            logger.info("Initializing resilient Firefox WebDriver...")
            driver, wait = self._get_driver()

            logger.info(f"Navigating to {self.url}")
            driver.get(self.url)

            # Wait for main table container
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.table")))

            for page_num in range(1, self.max_pages + 1):
                logger.info(f"Scraping page {page_num}/{self.max_pages}...")

                page_rows, current_sig = self._extract_current_table_atomic(driver, wait)
                if not page_rows:
                    logger.error(f"Failed to extract page {page_num} after retries. Aborting pagination.")
                    break

                all_rows.extend(page_rows)

                if page_num == self.max_pages:
                    break

                # Advance to next page
                success = self._turn_page_resilient(driver, wait, current_sig)
                if not success:
                    logger.info(f"Terminating crawl early at page {page_num}.")
                    break

        except Exception as e:
            logger.error(f"Scraping encountered critical issue: {e}. Preserving gathered records...", exc_info=True)
        finally:
            if driver:
                logger.info("Closing Web Browser...")
                try:
                    driver.quit()
                except Exception:
                    pass

        if not all_rows:
            logger.warning("No data rows captured during execution.")
            return pd.DataFrame()

        # Data normalization
        max_cols = max(len(r) for r in all_rows)
        normalized = [r + [None] * (max_cols - len(r)) for r in all_rows]

        if max_cols <= len(self.DEFAULT_COLUMNS):
            columns = self.DEFAULT_COLUMNS[:max_cols]
        else:
            columns = self.DEFAULT_COLUMNS + [
                f"column_{i+1}" for i in range(len(self.DEFAULT_COLUMNS), max_cols)
            ]

        df = pd.DataFrame(normalized, columns=columns)
        logger.info(f"Successfully scraped {len(df)} total records.")

        if save:
            self._save_to_disk(df)

        return df

    def _save_to_disk(self, df: pd.DataFrame) -> None:
        """Safely saves dataframe to the locked local cache path."""
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.filepath, index=False, encoding="utf-8-sig")
            logger.info(f"Data cached to: {self.filepath.resolve()}")
        except Exception as e:
            logger.error(f"Failed writing CSV to disk: {e}")
            raise IOError(f"Disk write error at {self.filepath}") from e

    def load_data(self) -> pd.DataFrame:
        """Loads dataset from the locked cache location."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Cache file missing: {self.filepath.resolve()}")

        try:
            logger.info(f"Reading cached data from {self.filepath.resolve()}")
            return pd.read_csv(self.filepath, encoding="utf-8-sig")
        except Exception as e:
            logger.error(f"Failed reading cache CSV: {e}")
            raise IOError(f"Could not read {self.filepath}") from e

    def delete_data(self) -> bool:
        """Deletes the cached CSV file if present."""
        if self.filepath.exists():
            try:
                self.filepath.unlink()
                logger.info(f"Cache deleted: {self.filepath.resolve()}")
                return True
            except Exception as e:
                logger.error(f"Failed removing cache file: {e}")
                raise IOError(f"Could not delete {self.filepath}") from e

        logger.info("Delete request skipped: No cache file exists.")
        return False

    def load_or_scrape(self, force_scrape: bool = False) -> pd.DataFrame:
        """Attempts loading local cache first; falls back to live scraping if unreadable or missing."""
        if not force_scrape and self.filepath.exists():
            try:
                return self.load_data()
            except Exception as e:
                logger.warning(f"Cache read failed ({e}). Proceeding to scrape live site.")

        logger.info("Cache unavailable or refresh requested. Initiating scrape...")
        return self.scrape(save=True)