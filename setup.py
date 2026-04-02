"""Setup configuration."""

from setuptools import setup, find_packages

setup(
    name="rl_self_reflection",
    version="0.1.0",
    description="RL-Guided Self-Reflection for LLM Reasoning",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "datasets>=2.16.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "torch>=2.1.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tqdm>=4.66.0",
        "loguru>=0.7.0",
        "rich>=13.7.0",
        "ratelimit>=2.2.3",
        "tenacity>=8.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rl-reasoning=main:main",
        ],
    },
)
