def main():
    from reflective_agent.utils.helpers import load_yaml

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent

    PROMPT_FILE = BASE_DIR / "config" / "prompts.yaml"
    prompts = load_yaml(str(PROMPT_FILE))


if __name__ == "__main__":
    main()
