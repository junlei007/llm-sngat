"""
Setup script for LLM-SNGAT package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-sngat",
    version="1.0.0",
    author="LLM-SNGAT Research Team",
    author_email="research@example.com",
    description="LLM-Simulated Nonequivalent Groups with Anchor Test for educational test equating",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-sngat",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Education :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.8",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
        ],
        "api": [
            "fastapi>=0.68",
            "uvicorn>=0.15",
            "pydantic>=1.8",
        ]
    },
    entry_points={
        "console_scripts": [
            "llm-sngat=run_experiment:main",
            "llm-sngat-demo=llm_sngat:main",
        ],
    },
    include_package_data=True,
    package_data={
        "llm_sngat": ["data/*.json", "configs/*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-sngat/issues",
        "Source": "https://github.com/yourusername/llm-sngat",
        "Documentation": "https://llm-sngat.readthedocs.io",
    },
)