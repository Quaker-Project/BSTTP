from setuptools import setup, find_packages

# Leer descripción larga desde README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Dependencias
modules = [
    "dill>=0.3.5.1",
    "geopandas>=0.14.0",
    "importlib-resources",
    "jax>=0.4.27",
    "jaxlib>=0.4.27",
    "matplotlib>=3.5.2",
    "multipledispatch>=0.6.0",
    "numpy>=1.24.1",
    "numpyro>=0.10.0",
    "opt-einsum>=3.3.0",
    "packaging>=21.3",
    "pandas>=1.4.3",
    "pyparsing>=3.0.9",
    "scipy>=1.9.0",
    "six>=1.16.0",
    "tqdm>=4.64.0",
    "streamlit>=1.24.0"  # para ejecutar la app
]

setup(
    name="BSTPP",
    version="0.1.3",
    description="Bayesian Spatiotemporal Point Process",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imanring/BSTPP.git",
    author="Isaac Manring",
    author_email="isaacamanring@gmail.com",
    license="MIT",
    packages=find_packages(),  # detecta automáticamente 'bstpp'
    package_data={"bstpp": ["decoders/*", "data/*"]},  # incluir datos
    install_requires=modules,
    python_requires=">=3.10,<3.12",  # compatible con JAX y Streamlit Cloud
)
