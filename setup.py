from setuptools import find_packages, setup


setup(
    name="pmi-mmr-reranker",
    version="0.1.0",
    description="A lightweight PMI-MMR reranker built on NumPy and FAISS.",
    long_description=open("package_README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Indrajit Kar",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy"],
)
