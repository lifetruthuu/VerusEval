import openai
import os
from dotenv import load_dotenv
load_dotenv()

import uuid
import time
import pickle
os.makedirs('dumps', exist_ok=True)



client_local = openai.Client(
    base_url=os.getenv("ALPHAVERUS_API_BASE", "https://api.openai.com/v1"),
    api_key=os.getenv("ALPHAVERUS_API_KEY", "EMPTY"),
)



def run_llm(history, model, max_tokens = 1024, temperature = 0.3, n = 8, port = 30000):
    try:
        client = client_local
        request_model = os.getenv("ALPHAVERUS_MODEL", "llama-3.3-70b")
        print(f'Doing {n} generations at depth {(len(history)-1)//2}')
        for attempt in range(20):
            try:
                response = client.chat.completions.create(
                    model=request_model,
                    messages = history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    response_format={
                        "type": "text"
                    },
                    n = n,
                )
                break
            except Exception as e:
                print(f'Error in attempt {attempt}: {e}')
                time.sleep(3 * (attempt))

        filename = 'dumps/' + str(uuid.uuid4()) + f'_{model}.json'
        with open(filename, 'w') as f:
            f.write(str(response))

        select_uuid = str(uuid.uuid4())
        dumped_filename = 'dumped_generations_gpt4/' + select_uuid + f'_{model.split("/")[-1]}.pkl'
        os.makedirs(os.path.dirname(dumped_filename), exist_ok=True)
        with open(dumped_filename, 'wb') as f:
            pickle.dump(response, f)

        return [x.message.content for x in response.choices]
        
    except Exception as e:
        print('Error in calling api:', e)
        return []
