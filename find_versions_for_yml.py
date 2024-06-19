#%%
import yaml
import subprocess

# Path to your environment.yml file
env_file_path = 'environment.yml'

def get_package_versions(packages):
    """Get the version of packages using conda list."""
    versions = {}
    for package in packages:
        # Execute `conda list` for the package and decode the output
        result = subprocess.run(['conda', 'list', package], capture_output=True, text=True)
        output = result.stdout
        # Parse the output to find the version
        for line in output.split('\n'):
            if package in line:
                parts = line.split()
                # Assuming the package name is exactly matched and version is in the second column
                versions[package] = parts[1] if len(parts) > 1 else 'Unknown'
                break
    return versions

def main():
    with open(env_file_path, 'r') as file:
        env_data = yaml.safe_load(file)
    
    # Extract dependencies, focusing on those without specified versions
    dependencies = env_data.get('dependencies', [])
    packages_without_versions = [dep for dep in dependencies if isinstance(dep, str) and '==' not in dep]
    
    # Get versions of these packages
    package_versions = get_package_versions(packages_without_versions)
    
    for package, version in package_versions.items():
        print(f"{package}: {version}")

if __name__ == "__main__":
    main()