from setuptools import setup, find_packages


HYPE_E_DOT = '-e .'


def get_requirements(file_path):
        """
    Reads a file named 'requirements.txt' and returns a list of requirements.

    Parameters:
        None.

    Returns:
        A list of strings representing the requirements read from the file.

    Raises:
        FileNotFoundError: If the file 'requirements.txt' does not exist.
    """
        try:
            with open(file_path) as f:
                requirements = f.read().splitlines()
                requirements = [req for req in requirements if not req.startswith('#')]
            if HYPE_E_DOT in requirements:
                requirements.remove(HYPE_E_DOT)
            return requirements
        except FileNotFoundError:
            raise FileNotFoundError("The file 'requirements.txt' does not exist.")
        

setup(
     name='house_price_e2e',
     version='0.0.1',
     author='Abhishek Pathak',
     author_email='pathak0801.abhishek@gmail.com',
     packages=find_packages(),
     install_requires=get_requirements('requirements.txt'),
)
    

