import json, re, os, ijson
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm
import gc

from .scraper import Post, extract_refs, clean_comment

LLAMA_TEMPLATE = """<|start_header_id|>{role}<|end_header_id|>

{content}<|eot_id|>"""

def format_prompt(
        user_input,
        system_prompt = None,
        assist_output = None
    ):
    formatted = ""

    # System part
    if system_prompt:
        formatted += LLAMA_TEMPLATE.format(
            role='system',
            content=system_prompt
        )

    # User part
    formatted += LLAMA_TEMPLATE.format(
        role='user',
        content=user_input
    )

    # Assistant part
    if assist_output:
        formatted += LLAMA_TEMPLATE.format(
            role='assistant',
            content=assist_output
        )

    return formatted

def has_url(txt):
    url_patterns = [
            r'https?://',
            r'www\.',
            r'\.[a-z]{2,}/\S',
            r'bit\.ly',
            r'goo\.gl',
            r't\.co',
        ]
    
    for pattern in url_patterns:
        if re.search(pattern, txt, re.IGNORECASE):
            return True
    return False

class DataProcessor:
    def __init__(self,
        min_length = 20,
        max_length = 2000,
        max_quote_ratio = 0.7,
        deduplicate = True
        ):
        self.min_length = min_length
        self.max_length = max_length
        self.max_quote_ratio = max_quote_ratio
        self.deduplicate = deduplicate

    def filter_post(self, post):
        comment = post.comment

        if len(comment) < self.min_length or len(comment) > self.max_length:
            return False
        
        lines = comment.split('\n')
        if lines:
            quote_lines = sum(1 for l in lines if l.strip().startswith('>'))
            if quote_lines > len(lines) * self.max_quote_ratio:
                return False
        
        if has_url(comment):
            return False

        return True

    def build_conversation_pairs(self, posts):
        """
        Creates (context, response) pairs from reply chains.
        """
        post_map = {p.post_id: p for p in posts}
        
        pairs = []
        seen_inputs, seen_outputs = set(), set()

        for post in tqdm(posts):
            if not post.replies_to:
                continue

            # Post it replies to
            for ref_id in post.replies_to:
                if ref_id in post_map:
                    parent = post_map[ref_id]
                    if self.filter_post(parent) and self.filter_post(post):
                        if self.deduplicate:
                            norm_input = parent.comment
                            norm_output = post.comment
                            if norm_input in seen_inputs:
                                continue
                            if norm_output in seen_outputs:
                                continue
                            seen_inputs.add(norm_input)
                            seen_outputs.add(norm_output)
                        
                        pairs.append({
                            'input': parent.comment,
                            'output': post.comment,
                            'board': post.board
                        })

        gc.collect()
        return pairs
    
    def build_completion_data(self, posts):
        """
        Builds simple completion data (so just the posts lol).
        """
        return [
            {'text': post.comment, 'board': post.board}
            for post in posts
            if self.filter_post(post)
        ]
    
    def format_for_training(self,
        data,
        mode = 'completion',
        model_name = None,
        system_prompt = None
        ):
        formatted = []

        for item in data:
            if mode == 'completion':
                formatted.append({'text': item['text']})

            elif mode == 'chat':
                text = format_prompt(
                    user_input = item['input'],
                    system_prompt = system_prompt,
                    assist_output = item['output']
                )
                formatted.append({"text": text})
        
        return formatted
    
    def create_dataset(self,
        posts,
        mode = 'completion',
        model_name = None,
        system_prompt = None
        ):
        """
        Creates a HuggingFace-compatible Dataset from posts.
        """
        if mode == 'chat':
            data = self.build_conversation_pairs(posts)
        else:
            data = self.build_completion_data(posts)

        formatted = self.format_for_training(
            data, 
            mode=mode,
            model_name=model_name,
            system_prompt=system_prompt
        )

        return Dataset.from_list(formatted)
    
def save_posts(posts, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True,exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump([p.to_dict() for p in posts], f, indent=2)

    print(f'Saved {len(posts)} posts to {filepath}.')

def load_posts(filepath, max_posts=150000):
    posts = []

    with open(filepath, 'rb') as f:
        for i, post in enumerate(tqdm(ijson.items(f, 'item'),total=max_posts)):
            if i >= max_posts:
                break
            posts.append(
                Post.from_dict(post)
            )
    print(f'Loaded {len(posts)} posts from {filepath}')
    return posts

def dump_big_log(input_file, output_file):
    filesize = os.path.getsize(input_file)

    with open(input_file,'r',encoding='utf8') as inp, open(output_file, 'w', encoding='utf-8') as out:
        with tqdm(total=filesize, unit='B', unit_scale=True) as pbar:
            for line_no, line in enumerate(inp, 1):
                pbar.update(len(line))
                if not line.strip():
                    continue

                try:
                    thread_meta = json.loads(line)
                    thread_posts = thread_meta.get('posts', [])
                    thread_id = thread_meta.get('no',-1)
                except json.JSONDecodeError:
                    continue

                for j, post_data in enumerate(thread_posts):
                        comment = clean_comment(post_data.get('com',''))
                        if not comment:
                            continue

                        refs = extract_refs(comment)
                        comment = re.sub(r'>>\d+\s*','',comment)

                        post = Post(
                            post_id = post_data['no'],
                            thread_id = thread_id,
                            board = 'pol',
                            comment = comment,
                            timestamp = post_data.get('time',0),
                            is_op = (j == 0),
                            replies_to = refs
                        )

                        out.write(json.dumps(post.to_dict(), ensure_ascii=False, indent=2) + ",\n")