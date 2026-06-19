# this will work as a package and can deploy our model in a pipeline 
from setuptools import find_packages , setup
from typing import List

HYPEN_E_DOT = '-e.'

def get_requirements(file_path:str)->List[str]:
    '''
        This function will return the list of requirements 
    '''

    requirements = []
    with open(file_path) as file_obj :
        requirements = file_obj.readlines()
        requirements = [ req.strip() for req in file_obj.readlines() ]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name = 'EEG-Project',
    version='0.0.1',
    author='Yukti',
    author_email='gargyukti112@gmail.com',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)