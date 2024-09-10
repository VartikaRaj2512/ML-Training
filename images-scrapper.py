from icrawler.builtin import GoogleImageCrawler

def download_images(query, limit=200, output_dir='downloads'):
    google_crawler = GoogleImageCrawler(storage={'root_dir': output_dir})
    google_crawler.crawl(keyword=query, max_num=limit)

if __name__ == "__main__":
    search_query = "brown eggs"
    download_images(search_query, limit=200, output_dir='downloads')
