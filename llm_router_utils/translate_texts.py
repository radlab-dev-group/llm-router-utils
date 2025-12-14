import argparse
import json

from llm_router_lib import LLMRouterClient


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    return p


def main(argv=None):
    args = prepare_parser(argv).parse_args(argv)

    client = LLMRouterClient(api="http://192.168.100.65:8080")

    response = client.translate(
        model="speakleash/Bielik-11B-v2.3-Instruct", texts=["Hello, how are you?"]
    )
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
