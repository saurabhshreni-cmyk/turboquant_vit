"""Run once to create assets/banner.png."""
from utils import make_banner

if __name__ == "__main__":
    make_banner().save("assets/banner.png")
    print("Wrote assets/banner.png")
