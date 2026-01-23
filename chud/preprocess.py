import json, re
from pathlib import Path
from datasets import Dataset

from .scraper import Post

TEMPLATE_LLAMA = """<|start_header_id|>user<|end_header_id|>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

TEMPLATE_LLAMA_SYSTEM = """<|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

def get_template(model_name, use_system_prompt = False):
    if 'instruct' in model_name.lower() and use_system_prompt:
        return TEMPLATE_LLAMA_SYSTEM
    
    return TEMPLATE_LLAMA

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

        for post in posts:
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
                use_system = system_prompt is not None
                template = get_template(model_name or "", use_system)
                
                for item in data:
                    if use_system:
                        text = template.format(
                            system=system_prompt,
                            input=item["input"],
                            output=item["output"]
                        )
                    else:
                        text = template.format(
                            input=item["input"],
                            output=item["output"]
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

def load_posts(filepath):
    with open(filepath) as f:
        data = json.load(f)

    posts = [Post.from_dict(p) for p in data]
    print(f'Loaded {len(posts)} posts from {filepath}')
    return posts