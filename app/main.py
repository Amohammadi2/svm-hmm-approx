import streamlit as st
import pandas as pd
import streamlit as st
from numpy.random import default_rng as rng
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:
    sys.path.insert(0, parentdir) 

from datalink.scraper import TGJUScraper

st.set_page_config(
    page_title="VaR Dashboard",
    page_icon="👋",
)

st.title("VaR Analysis Dashboard")

st.markdown(
    "This is an expository web application built on top of our data processing core to demonstrate"
    " our real data pipeline from scraping to visual analysis")

# --- Controls ---
st.header("Data Scraper", divider="rainbow")

st.markdown(
    "This web app scrapes financial data from this website: https://www.tgju.org/profile/tgju_gold_irg18/history. "
    "Each page contains 30 data points (roughly one month) and you can configure how many pages to scrape."
)

pages = st.slider(
    "Number of pages to scrape",
    min_value=1,
    max_value=24,
    value=3,
)

if "raw_data" not in st.session_state:
    scraper = TGJUScraper(headless=True, max_pages=pages)
    st.session_state.raw_data = scraper.load_or_scrape()

if st.button("🔄 Scrape data", type="primary"):
    with st.spinner(f"Scraping {pages} pages..."):
        scraper = TGJUScraper(headless=True, max_pages=pages)
        st.session_state.raw_data = scraper.scrape()

    st.success("Data successfully rescraped!")

# --- Display ---
st.header("Raw Dataframe", divider="rainbow")
st.markdown("This part previews the cached data upon application startup, you can update to the latest data by rescraping.")
st.dataframe(
    st.session_state.raw_data,
    use_container_width=True,
)

st.header("Model Parameters", divider="rainbow")
st.markdown("You can either enter the values manually or fit the parameters against the data")