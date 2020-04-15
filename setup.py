import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="face-feature-recognizer",  # Replace with your own username
    version="0.0.1",
    author="Karl Gylleus",
    author_email="karl.gylleus@gmail.com",
    description="Extracts facial features from images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    install_requires=["face_recognition"],
    python_requires=">=3.6",
)
