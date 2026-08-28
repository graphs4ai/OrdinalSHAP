import asyncio
from openai import AsyncOpenAI, RateLimitError, APIError, APITimeoutError


class InstanceModel:

  def __init__(
    self,
    model_name: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
    max_concurrency: int = 8,
    requests_per_minute: int = 50,
    max_retries: int = 5,
  ):
    self.model_name = model_name
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.max_retries = max_retries

    self.client = AsyncOpenAI(
      api_key=api_key,
      base_url=base_url,
    )

    self.semaphore = asyncio.Semaphore(max_concurrency)

    self.min_interval = 60.0 / requests_per_minute
    self.rate_lock = asyncio.Lock()
    self.last_request_time = 0.0


  async def _wait_rate_limit(self):
    async with self.rate_lock:
      loop = asyncio.get_running_loop()
      now = loop.time()

      wait_time = self.min_interval - (now - self.last_request_time)

      if wait_time > 0:
        await asyncio.sleep(wait_time)

      self.last_request_time = loop.time()


  async def generate(
    self,
    user_prompt: str,
    system_prompt: str | None = None,
  ):

    messages = []
    if system_prompt is not None:
      messages.append({
        "role": "system",
        "content": system_prompt,
      })

    messages.append({
      "role": "user",
      "content": user_prompt,
    })

    last_exception = None
    for attempt in range(self.max_retries):
      try:
        async with self.semaphore:
          await self._wait_rate_limit()

          response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
          )

        content = response.choices[0].message.content

        if content is None:
          raise ValueError("Empty response content")

        return content.strip()

      except (RateLimitError, APITimeoutError) as e:
        last_exception = e
        wait = 2 ** attempt
        print(f"[Retry {attempt + 1}/{self.max_retries}] {type(e).__name__}: waiting {wait}s")
        await asyncio.sleep(wait)

      except APIError as e:
        last_exception = e
        print(f"[API error] {type(e).__name__}: {e}")
        await asyncio.sleep(2 ** attempt)

      except Exception as e:
        last_exception = e
        print(f"[Unexpected error] {type(e).__name__}: {e}")
        await asyncio.sleep(2 ** attempt)

    raise last_exception

  async def batch_generate(self, prompts, system_prompt=None):

    tasks = [self.generate(user_prompt=prompt, system_prompt=system_prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)