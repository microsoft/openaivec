import os

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI


def provide_openai_client() -> OpenAI:
    """Provide OpenAI client based on environment variables. Prioritizes OpenAI over Azure."""
    if os.getenv("OPENAI_API_KEY"):
        return OpenAI()

    if all(
        os.getenv(name) for name in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_ENDPOINT", "AZURE_OPENAI_API_VERSION"]
    ):
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

    raise ValueError(
        "No valid OpenAI or Azure OpenAI environment variables found. "
        "Please set either OPENAI_API_KEY or AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_API_ENDPOINT, and AZURE_OPENAI_API_VERSION."
    )


def provide_async_openai_client() -> AsyncOpenAI:
    """Provide async OpenAI client based on environment variables. Prioritizes OpenAI over Azure."""
    if os.getenv("OPENAI_API_KEY"):
        return AsyncOpenAI()

    if all(
        os.getenv(name) for name in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_ENDPOINT", "AZURE_OPENAI_API_VERSION"]
    ):
        return AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

    raise ValueError(
        "No valid OpenAI or Azure OpenAI environment variables found. "
        "Please set either OPENAI_API_KEY or AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_API_ENDPOINT, and AZURE_OPENAI_API_VERSION."
    )
