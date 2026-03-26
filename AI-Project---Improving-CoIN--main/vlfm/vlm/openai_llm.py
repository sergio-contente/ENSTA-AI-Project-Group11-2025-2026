import os
from colorama import Fore
from colorama import init as init_colorama
from openai import OpenAI

init_colorama(autoreset=True)
from retrying import retry
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


class OpenAILLMClient:
    def __init__(self, llm_client_params) -> None:
        print(Fore.YELLOW + f"[INFO] Initializing OpenAI LLM")

        all_env_vars = os.environ

        # Use API key from params if provided, otherwise from env var
        if "api_key" not in llm_client_params or llm_client_params.get("api_key") is None:
            self.api_keys = [value for key, value in all_env_vars.items() if key.startswith("COIN_LLM_CLIENT_KEY")]
            llm_client_params["api_key"] = self.api_keys[0] if self.api_keys else None
        
        self.model = llm_client_params.get("model", "gpt-4o")
        del llm_client_params["model"]

        # Keep generation length conservative for local models.
        self.max_model_len = int(os.environ.get("LOCAL_LLM_MAX_MODEL_LEN", "4096"))
        self.max_output_tokens_cap = int(os.environ.get("LOCAL_LLM_MAX_OUTPUT_TOKENS", "1536"))

        self.client = OpenAI(**llm_client_params)
        print(Fore.YELLOW + f"[INFO] LLM configured with model: {self.model}, base_url: {llm_client_params.get('base_url', 'default')}")

    def _safe_max_tokens(self, prompt: str) -> int:
        # Rough token estimate: ~4 chars per token for English-like text.
        estimated_prompt_tokens = max(1, len(prompt) // 4)
        reserved_tokens = 64
        available = self.max_model_len - estimated_prompt_tokens - reserved_tokens
        if available <= 0:
            return 64
        return max(1, min(self.max_output_tokens_cap, available))

    @retry(
        retry_on_exception=(
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
            Exception,
        ),
        stop_max_attempt_number=1,
        wait_fixed=6000,
    )
    def ask(self, prompt: str) -> str:
        try:
            max_tokens = self._safe_max_tokens(prompt)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                top_p=1,
                max_tokens=max_tokens,
                seed=42,
            )
            return completion.choices[0].message.content

        except RateLimitError as e:
            print(Fore.RED + "[ERROR] Rate Limit Error")
            print(Fore.RED + f"[ERROR] {e}")
            raise Exception("retry")


if __name__ == "__main__":
    ## Test with python vlfm/vlm/openai_llm.py

    # Configuration options: "local", "groq", or "openai"
    # Set to "local" to use local vLLM server (requires launch_vlm_servers.sh with USE_LOCAL_LLM=true)
    llm_provider = os.environ.get("LLM_PROVIDER", "local")

    if llm_provider == "local":
        # Local vLLM server
        local_port = os.environ.get("LOCAL_LLM_PORT", "8000")
        llm_client_params = {
            "model": os.environ.get("LOCAL_LLM_MODEL_NAME", "Qwen2.5-Coder-32B-Instruct"),
            "base_url": f"http://localhost:{local_port}/v1",
            "api_key": "not-needed",
        }
    elif llm_provider == "groq":
        # Groq API (free key from https://groq.com/)
        llm_client_params = {
            "model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1",
        }
    else:
        # OpenAI
        llm_client_params = {
            "model": "gpt-4o",
        }

    llm_client = OpenAILLMClient(llm_client_params)

    prompt = "What is the capital of France?"
    response = llm_client.ask(prompt)
    print(Fore.GREEN + f"Response: {response}")
