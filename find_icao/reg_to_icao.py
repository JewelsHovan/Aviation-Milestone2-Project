import requests
from bs4 import BeautifulSoup

def get_icao24(registration_number):
    """
    Fetches the Mode S Code (Base 16 / Hex), which is the ICAO24 address, for the given registration number
    from the FAA registry site.

    Note:
    - The __RequestVerificationToken and Cookies are hard-coded here as taken from your cURL command.
      They might need to be updated if the session or token changes.
    """
    url = "https://registry.faa.gov/aircraftinquiry/Search/NNumberResult"

    # Form data as required by the FAA site.
    form_data = {
        "NNumbertxt": registration_number,
        "__RequestVerificationToken": "CfDJ8EJ3444mshFMjKw-azBuABfpPCer4s8nuQPNHNNJyMSGeNsTW8OFBoDzRoHidBRXJS00XuhkgoKhfTaKbSGTlDx9WbFgli3OxajOZcLaN7QxMhrQ2Glh36UWb1myxF01xyxyyKXTipLFO8e1JpddhRc"
    }

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "Content-Type": "application/x-www-form-urlencoded",
        "DNT": "1",
        "Origin": "https://registry.faa.gov",
        "Referer": "https://registry.faa.gov/aircraftinquiry/search/nnumberinquiry",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/132.0.0.0 Safari/537.36"),
        "sec-ch-ua": "\"Not A(Brand\";v=\"8\", \"Chromium\";v=\"132\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"macOS\""
    }

    response = requests.post(url, data=form_data, headers=headers, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    
    mode_s_code = None
    # Look for table rows containing the FAA data
    for row in soup.find_all("tr", class_="devkit-table-row"):
        cells = row.find_all("td")
        # Iterate over cells to locate the label "Mode S Code (Base 16 / Hex)"
        for i, cell in enumerate(cells):
            if "mode s code (base 16 / hex)" in cell.get_text(strip=True).lower():
                # Assume the next cell contains the value
                if i + 1 < len(cells):
                    mode_s_code = cells[i + 1].get_text(strip=True)
                    break
        if mode_s_code:
            break

    if not mode_s_code:
        raise ValueError("Mode S Code (ICAO24) not found in the response.")

    return mode_s_code

if __name__ == "__main__":
    # Example usage:
    registration = "N2UZ"
    try:
        result = get_icao24(registration)
        print(f"Mode S Code (ICAO24) for {registration}:", result)
    except Exception as e:
        print("Error:", e)
