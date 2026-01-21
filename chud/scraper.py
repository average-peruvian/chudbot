"""
API documentation: https://github.com/4chan/4chan-API
"""

import re, time, requests
from dataclasses import dataclass, asdict

BASE_URL = 'https://a.4cdn.org'
RATE_LIMIT_DELAY = 1.0

def clean_comment(txt):
    if not txt:
        return ''
    
    txt = re.sub(r'<br\s*/?>','\n',txt)
    txt = re.sub(r'<a[^>]*class="quotelink"[^>]*>>>(\d+)</a>', r'>>\1',txt)
    txt = re.sub(r'<[^>]+>', '',txt)
    txt = re.sub(r'\n{3,}','\n\n',txt)
    txt = txt.strip()
    return txt

def extract_refs(comment):
    matches = re.findall(r'>>(\d+)',comment)
    return [int(m) for m in matches]

@dataclass
class Post:
    post_id: int
    thread_id: int
    board: str
    comment: str
    timestamp: int
    is_op: bool
    replies_to: list[int]

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class ScraperChan:
    def __init__(self, rate_limit = RATE_LIMIT_DELAY):
        self.session = requests.Session()
        self.rate_limit = rate_limit

    def _request(self, endpoint):
        url = f'{BASE_URL}/{endpoint}'
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            time.sleep(self.rate_limit)
            return response.json()
        except requests.RequestException as e:
            print(f'Request failed for {url}: {e}')
            return None
        
    def get_boards(self):
        data = self._request('boards.json')
        return data.get('boards', []) if data else []
    
    def get_catalog(self, board):
        data = self._request(f'{board}/catalog.json')
        if not data:
            return []
        
        threads = []
        for page in data:
            threads.extend(page.get('threads', []))
        return threads
    
    def get_thread(self, board, thread_id):
        data = self._request(f'{board}/thread/{thread_id}.json')
        return data.get('posts', []) if data else []
    
    def scrape_thread(self, board, thread_id):
        posts = []
        thread_posts = self.get_thread(board,thread_id)

        for j, post_data in enumerate(thread_posts):
            comment = clean_comment(post_data.get('com',''))
            if not comment:
                continue

            post = Post(
                post_id = post_data['no'],
                thread_id = thread_id,
                board = board,
                comment = comment,
                timestamp = post_data.get('time',0),
                is_op = (j == 0),
                replies_to = extract_refs(comment)
            )
            posts.append(post)

        return posts
    
    def scrape_board(self,
        board,
        max_threads = 50,
        min_replies = 5,
        verbose = True
        ):
        posts = []
        catalog = self.get_catalog(board)

        threads = [t for t in catalog if t.get('replies', 0) >= min_replies]
        threads = threads[:max_threads]

        if verbose:
            print(f'Scraping {len(threads)} threads from /{board}/...')

        for i, thread_meta in enumerate(threads):
            thread_id = thread_meta['no']
            thread_posts = self.scrape_thread(board, thread_id)
            posts.extend(thread_posts)

            if verbose and (i + 1) % 10 == 0:
                print(f'  Progress: {i + 1}/{len(threads)} threads')

        return posts
    
if __name__ == "__main__":
    scraper = ScraperChan()
    boards = scraper.get_boards()
    print(f'Available boards: {len(boards)}')
    print('Example boards:',[b['board'] for b in boards[:10]])
