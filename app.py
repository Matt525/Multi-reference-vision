import argparse

from refvision.web import launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the phone-friendly Multi-Reference Vision test UI")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary HTTPS Gradio link that can be opened on an iPhone.",
    )
    args = parser.parse_args()
    launch(share=args.share)
