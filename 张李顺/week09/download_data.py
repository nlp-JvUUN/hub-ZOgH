"""Download one public annual report: BYD 2023 (not part of the teaching dataset)."""
import requests
from config import DATA_DIR, PDF_PATH, PDF_URL


def main():
    DATA_DIR.mkdir(exist_ok=True)
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 1_000_000:
        print(f"Already present: {PDF_PATH}")
        return
    response = requests.get(PDF_URL, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    PDF_PATH.write_bytes(response.content)
    print(f"Saved {PDF_PATH.name} ({PDF_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
