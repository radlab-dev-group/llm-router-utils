from pathlib import Path
from setuptools import setup, find_packages

BASE_DIR = Path(__file__).parent

# ------------------------------------------------------------------
# Core version & long description
# ------------------------------------------------------------------
version = (BASE_DIR / ".version").read_text().strip()
long_description = (BASE_DIR / "README.md").read_text(encoding="utf-8")

# ------------------------------------------------------------------
# Runtime dependencies (empty by default – you can list them here)
# ------------------------------------------------------------------
install_requires = []

# ------------------------------------------------------------------
# Extras (optional groups of dependencies)
# ------------------------------------------------------------------
extras_require = {
    "llm-router": [
        "llm-router @ git+https://github.com/radlab-dev-group/llm-router",
        "llm-router-services @ git+https://github.com/radlab-dev-group/llm-router-services",
    ],
}

# ------------------------------------------------------------------
# Setup configuration
# ------------------------------------------------------------------
setup(
    name="llm-router-utils",
    version=version,
    description="Utility package for LLM‑Router – ready‑made tools and examples",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RadLab.dev Ream",
    url="https://github.com/radlab-dev-group/llm-router-utils",
    license="Apache-2.0",
    packages=find_packages(
        where=".", include=["llm_router_utils*"], exclude=("tests", "docs")
    ),
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": {
            "translate-texts=llm_router_utils.cli.translate_texts:main"
        }
    },
)
