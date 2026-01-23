import argparse

from chud.scraper import ScraperChan
from chud.preprocess import save_posts
from chud.utils import project_config

def main():
    parser = argparse.ArgumentParser(
        description='Scrape 4chan threads',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', '-c',
        default='config/scrape.yaml',
        help='Path to YAML config file'
    )

    args = parser.parse_args()
    config = project_config(args.config,'scrape')

    print("=== SCRAPING ===")
    scraper = ScraperChan(rate_limit=config["rate_limit"])
    
    all_posts = []

    for board in config['boards']:
        print(f"\n{'='*50}")
        print(f'Scraping /{board}/...')
        print(f"{'='*50}")
        
        posts = scraper.scrape_board(
            board,
            max_threads=config['max_threads'],
            min_replies=config['min_replies'],
            verbose=True
        )
        
        all_posts.extend(posts)
        print(f'\nScraped {len(posts)} posts from /{board}/')

    save_posts(all_posts, config["output_file"])

    print(f"\n{'='*50}")
    print(f'TOTAL POSTS SCRAPED: {len(all_posts)}')
    print(f"Saved to: {config['output_file']}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()