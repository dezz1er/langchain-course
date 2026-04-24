import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from langchain_tavily import TavilyExtract, TavilyMap
from rich.console import Console
from rich.panel import Panel

# Configure SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Initialize rich console for pretty printing
console = Console()

print("✅ All imports successful!")

api_key = os.environ["TAVILY_API_KEY"]

tavily_map = TavilyMap(
    max_depth=3,        # Crawl up to 3 levels deep
    max_breadth=15,     # Follow up to 15 links per page
    max_pages=50        # Limit to 50 total pages for demo
)

print("✅ TavilyMap initialized successfully!")

# Example website to map
demo_url = "https://python.langchain.com/docs/introduction/"

console.print(f"🔍 Mapping website structure for: {demo_url}", style="bold blue")
console.print("This may take a moment...")

# Map the website structure
site_map = tavily_map.invoke(demo_url)

# Display results
urls = site_map.get('results', [])
console.print(f"\n✅ Successfully mapped {len(urls)} URLs!", style="bold green")

# Show first 50 URLs as examples
console.print("\n📋 First 50 discovered URLs:", style="bold yellow")
for i, url in enumerate(urls[:50], 1):
    console.print(f"  {i:2d}. {url}")

if len(urls) > 10:
    console.print(f"  ... and {len(urls) - 50} more URLs")

# Initialize TavilyExtract
tavily_extract = TavilyExtract()

print("✅ TavilyExtract initialized successfully!")


def chunk_urls(urls: List[str], chunk_size: int = 3) -> List[List[str]]:
    """Split URLs into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs."""
    try:
        console.print(
            f"🔄 Processing batch {batch_num} with {len(urls)} URLs",
            style="blue"
        )
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        results = docs.get('results', [])
        console.print(
            f"✅ Batch {batch_num} completed - extracted {len(results)} documents",
            style="green"
        )
        return results
    except Exception as e:
        console.print(f"❌ Batch {batch_num} failed: {e}", style="red")
        return []


async def main():
    """Main async function to orchestrate the extraction workflow."""
    # Select a few interesting URLs for extraction
    sample_urls = [urls[15]]  # Take first URL
    console.print(
        f"📚 Extracting content from {len(sample_urls)} URLs...",
        style="bold blue"
    )

    # Extract content
    extraction_result = await tavily_extract.ainvoke(input={"urls": sample_urls})

    # Display results
    extracted_docs = extraction_result.get('results', [])
    console.print(
        f"\n✅ Successfully extracted {len(extracted_docs)} documents!",
        style="bold green"
    )

    # Show summary of each extracted document
    for i, doc in enumerate(extracted_docs, 1):
        url = doc.get('url', 'Unknown')
        content = doc.get('raw_content', '')

        # Create a panel for each document
        panel_content = f"""URL: {url}
Content Length: {len(content):,} characters
Preview: {content}..."""

        console.print(
            Panel(panel_content, title=f"Document {i}", border_style="blue")
        )
        print()  # Add spacing

    # Process a larger set of URLs in batches
    url_batches = chunk_urls(urls[:9], chunk_size=3)
    console.print(
        f"📦 Processing 9 URLs in {len(url_batches)} batches",
        style="bold yellow"
    )

    # Process batches concurrently
    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]
    batch_results = await asyncio.gather(*tasks)

    # Flatten results
    all_extracted = []
    for batch_result in batch_results:
        all_extracted.extend(batch_result)

    console.print(
        f"\n🎉 Batch processing complete! Total documents extracted: {len(all_extracted)}",
        style="bold green"
    )


if __name__ == "__main__":
    asyncio.run(main())
