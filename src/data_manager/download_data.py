from pathlib import Path
import urllib.request

def download_data(url, path:Path):
    print("Downloading data....")
    try:
        urllib.request.urlretrieve(url, path)
        print("Successfully retrieved data")
    except:
        print("Couldn't Fetch Data.")

if __name__ == "__main__":
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    path = Path(__file__).resolve().parent.parent.parent / "data" / "data.txt"
    download_data(url, path)
    print(path)