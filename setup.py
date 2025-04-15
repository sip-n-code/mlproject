from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT='-e .'
# Some metadata about the project
def get_requirements(file_path:str)->List[str]:
    '''
    This functions returns list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", " ") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements



setup(
    name='mlproject',
    version='0.0.1',
    author='Sips',
    author_email='sl@example.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)