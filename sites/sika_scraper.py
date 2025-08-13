import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import re
import sys
import pdfplumber

# Configure system output encoding
sys.stdout.reconfigure(encoding='utf-8')

# Base URL for Sika GCC website
BASE_URL = "https://gcc.sika.com"

def extract_technical_specifications(soup):
    """
    Extract technical specifications from product page HTML
    Returns a dictionary with four main sections:
    - Technical_Specifications
    - Application_Data
    - Certifications
    - Packaging_Info
    """
    technical_specs = {
        "Technical_Specifications": {},
        "Application_Data": {},
        "Certifications": {},
        "Packaging_Info": {}
    }

    # Extract main Technical Information section
    tech_section = soup.find("a", class_="cmp-accordion__title", string="Technical Information")
    if tech_section:
        content_div = tech_section.find_next("div", class_="cmp-accordion__content")
        if content_div:
            product_blocks = content_div.find_all("div", class_="cmp-product")
            
            for block in product_blocks:
                title_tag = block.find("h3")
                if not title_tag:
                    continue
                    
                title = title_tag.get_text(strip=True)
                content = extract_content(block)
                if content:
                    technical_specs["Technical_Specifications"][title] = content

    # Extract Application Information section
    app_section = soup.find("a", class_="cmp-accordion__title", string="Application Information")
    if app_section:
        content_div = app_section.find_next("div", class_="cmp-accordion__content")
        if content_div:
            app_blocks = content_div.find_all("div", class_="cmp-product")
            
            for block in app_blocks:
                title_tag = block.find("h3")
                if not title_tag:
                    continue
                    
                title = title_tag.get_text(strip=True)
                content = extract_content(block)
                if content:
                    technical_specs["Application_Data"][title] = content

    # Extract Certifications section
    cert_section = soup.find("a", class_="cmp-accordion__title", string="Sustainability / Certifications / Approvals")
    if cert_section:
        content_div = cert_section.find_next("div", class_="cmp-accordion__content")
        if content_div:
            cert_blocks = content_div.find_all("div", class_="cmp-product")
            
            for block in cert_blocks:
                title_tag = block.find("h3")
                if not title_tag:
                    continue
                    
                title = title_tag.get_text(strip=True)
                content = extract_content(block)
                if content:
                    technical_specs["Certifications"][title] = content

    # Extract Packaging Information
    packaging_section = soup.find("h3", string="Packaging")
    if packaging_section:
        packaging_block = packaging_section.find_parent("div", class_="cmp-product")
        if packaging_block:
            technical_specs["Packaging_Info"] = extract_content(packaging_block)

    return technical_specs

def extract_content(block):
    """
    Helper function to extract content from HTML blocks
    Supports tables, lists, paragraphs, and single span elements
    """
    # Case 1: Extract table data
    table = block.find("table")
    if table:
        return extract_table_data(table)
    
    # Case 2: Extract list items
    ul = block.find("ul")
    if ul:
        return [li.get_text(strip=True) for li in ul.find_all("li") if li.get_text(strip=True)]
    
    # Case 3: Extract paragraphs
    paragraphs = block.find_all("p")
    if paragraphs:
        return "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    
    # Case 4: Extract single span element
    span = block.find("span")
    if span and span.get_text(strip=True):
        return span.get_text(strip=True)
    
    return None

def extract_table_data(table):
    """
    Extract data from HTML tables
    Returns either a dictionary (for header-based tables) or list of rows
    """
    table_data = []
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    
    for row in table.find_all("tr"):
        cols = [col.get_text(strip=True).replace('\xa0', ' ') for col in row.find_all("td")]
        if cols:
            if headers and len(headers) == len(cols):
                table_data.append(dict(zip(headers, cols)))
            else:
                table_data.append(cols)
    
    return table_data[0] if len(table_data) == 1 else table_data

def extract_article_number_from_pdf(pdf_url):
    """
    Extract article number from product datasheet PDF
    Looks for long numeric strings (15-20 digits) in first page
    """
    try:
        # Download PDF temporarily
        response = requests.get(pdf_url)
        with open("temp_datasheet.pdf", "wb") as f:
            f.write(response.content)

        # Extract text from first page
        with pdfplumber.open("temp_datasheet.pdf") as pdf:
            first_page_text = pdf.pages[0].extract_text()

        # Find long numeric strings (likely article numbers)
        matches = re.findall(r"\b\d{15,20}\b", first_page_text)
        if matches:
            return matches[-1]  # Typically appears at bottom
        return None
    except Exception as e:
        print(f"Failed to extract from PDF: {e}")
        return None

def get_datasheet_url(soup, base_url):
    """
    Find product datasheet PDF URL using multiple strategies:
    1. Elements with data-document-type attribute
    2. PDF links in downloads section
    3. Any PDF link on page
    """
    # Strategy 1: Data-document-type elements
    datasheet_elem = soup.find(attrs={"data-document-type": True})
    if datasheet_elem:
        if datasheet_elem.name == "a" and datasheet_elem.get("href"):
            return urljoin(base_url, datasheet_elem["href"])
        if datasheet_elem.get("value"):
            return urljoin(base_url, datasheet_elem["value"])
        if datasheet_elem.get("data-document-path"):
            return urljoin(base_url, datasheet_elem["data-document-path"])

    # Strategy 2: Downloads section PDF
    downloads_section = soup.find(class_="downloads")
    if downloads_section:
        pdf_link = downloads_section.find("a", href=lambda h: h and h.lower().endswith(".pdf"))
        if pdf_link and pdf_link.get("href"):
            return urljoin(base_url, pdf_link["href"])

    # Strategy 3: Any PDF link on page
    pdf_link = soup.find("a", href=lambda h: h and h.lower().endswith(".pdf"))
    if pdf_link and pdf_link.get("href"):
        return urljoin(base_url, pdf_link["href"])

    return None

def extract_product_data(url):
    """
    Extract all product data from a product page URL
    Returns dictionary with complete product information
    """
    print(f"[+] Extracting: {url}")
    response = requests.get(url, timeout=60)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Initialize product dictionary with default values
    product = {
        "URL": url,
        "Brand": "Sika", 
        "Product Name": "",
        "Model / Article Number": "",
        "Category": "Construction technology",
        "Subcategory": "",
        "Technical Specifications": {},
        "Short Description": "",
        "Long Description": "",
        "Product Image URL": "",
        "Datasheet URL": "",
    }

    # Extract Product Name from H1 tag
    title = soup.find("h1")
    if title:
        product["Product Name"] = title.get_text(strip=True)

    # Extract Short Description
    short_desc = soup.find("div", class_="cmp-product__description--short")
    if short_desc:
        product["Short Description"] = short_desc.get_text(strip=True)

    # Extract Long Description
    long_desc = soup.select_one("p.cmp-text__paragraph")
    if long_desc:
        product["Long Description"] = long_desc.get_text(strip=True)

    # Find Datasheet PDF URL
    product["Datasheet URL"] = get_datasheet_url(soup, BASE_URL)
    
    # Extract Technical Specifications
    product["Technical Specifications"] = extract_technical_specifications(soup)

    # Extract Category/Subcategory from URL path
    path_parts = urlparse(url).path.strip("/").split("/")
    if len(path_parts) >= 4:
        subcategory_raw = " > ".join(path_parts[2:-1])
        product["Subcategory"] = subcategory_raw.replace("-", " ").title()
        
    # Extract Article Number (from PDF or page)
    product["Model / Article Number"] = extract_article_number_from_pdf(product["Datasheet URL"])
        
    return product

def extract_all_products(product_links_file):
    """
    Process all product links from a text file
    Returns list of product dictionaries
    """
    with open(product_links_file, encoding="utf-8") as f:
        links = [line.strip() for line in f if line.strip()]

    all_products = []
    for url in links:
        try:
            product_data = extract_product_data(url)
            all_products.append(product_data)
        except Exception as e:
            print(f"   ✗ Failed to extract {url}: {e}")
    
    return all_products

def save_to_json(data, filename):
    """
    Save extracted data to JSON file with proper encoding
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to {filename}")

if __name__ == "__main__":
    # Main execution: extract products and save to JSON
    products = extract_all_products("product_links.txt")
    save_to_json(products, "sika_product_data.json")