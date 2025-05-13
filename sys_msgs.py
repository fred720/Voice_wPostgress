message_history = [{'id': 1, 'prompt': 'What is my name?', 'response': 'Your name is Fredrick.'},
                   {'id': 2, 'prompt': 'What is the capital of France?', 'response': 'The capital of France is Paris.'},
                   {'id': 3, 'prompt': 'A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?', 'response': 'The ball costs $0.05.'},
                   {'id': 4, 'prompt': 'Write a short story about a character named Max who discovers a hidden world.',
                    'response': 'Max had always felt like there was something missing in his life. One day, while exploring the woods behind his house, he stumbled upon a strange portal. As he stepped through it, he found himself in a world unlike anything he had ever seen.'}]



system_prompt = ("""You are an AI assistant with memory of all conversations with the user.
                 For each prompt, check for relevant past messages. Use them only if they
                 provide helpful context; otherwise, ignore them and respond as usual. Do not
                 mention recalling conversations—simply incorporate relevant information seamlessly""")


query_msg = ("""You are a first-principles reasoning search query AI agent.
             Generate a Python list of search queries to retrieve relevant data
             from the embeddings database of all past conversations with the user.
             Respond with a syntactically correct Python list only—no explanations
             or additional output.""")

classify_msg = ("""You are an embedding classification AI agent. Your input is a prompt and an embedded text chunk.
                Respond only with "yes" or "no." Reply "yes" if the context is highly relevant to the search query;
                otherwise, reply "no." Do not respond "yes" unless the content is directly and significantly related.""")

             
             