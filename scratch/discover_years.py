import cloudscraper
from bs4 import BeautifulSoup
import re

scraper = cloudscraper.create_scraper()
url = "https://www.bbtc.com.sg/resources/"
response = scraper.get(url)
soup = BeautifulSoup(response.text, "html.parser")

years = set()
for a in soup.find_all("a", href=True):
    href = a["href"]
    match = re.search(r"sermons-(\d{4})", href)
    if match:
        years.add(int(match.group(1)))

print(f"Found years: {sorted(list(years))}")
