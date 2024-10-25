from importlib import metadata 
import platform

def get_version():
    python_version = platform.python_version()
    exosims_version = metadata.version('EXOSIMS')

    reqs = metadata.distribution('EXOSIMS').requires
    required_packages = [str(req) for req in reqs]

    # Get installed versions of required packages
    installed_packages = {dist.metadata['Name']: dist.version for dist in metadata.distributions()}

    # Filter installed packages to those listed in requirements
    relevant_packages = {pkg: installed_packages.get(pkg.split('>=')[0], "Not installed") for pkg in required_packages}

    return {
        'Python': python_version,
        'EXOSIMS': exosims_version,
        'Packages': relevant_packages
    }


version_info = get_version() 
for key, value in version_info.items(): 
        if isinstance(value, dict): 
            print(f'{key}:') 
            for sub_key, sub_value in value.items():
                print(f'  {sub_key}:'.ljust(25)+f'{sub_value}') 
        else: 
            print(f'{key}:'.ljust(25)+f'{value}')