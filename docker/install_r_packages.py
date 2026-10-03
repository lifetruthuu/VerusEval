"""Install the paper's R plotting packages from checksum-verified source archives."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    packages = json.loads(Path('/opt/r-packages.lock.json').read_text())['packages']
    with tempfile.TemporaryDirectory() as directory:
        def download(package):
            path = Path(directory) / f"{package['package']}_{package['version']}.tar.gz"
            urls = [package['url'], 'https://cran.r-project.org/src/contrib/Archive/'
                    f"{package['package']}/{path.name}"]
            for url in dict.fromkeys(urls):
                result = subprocess.run(['curl', '-fLsS', '--retry', '3', '--max-time', '120',
                                         url, '-o', str(path)])
                if result.returncode == 0:
                    break
            else:
                raise RuntimeError(f"Cannot download {package['package']}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != package['sha256']:
                raise RuntimeError(f"Hash mismatch: {package['package']}")
            return path
        with ThreadPoolExecutor(4) as pool:
            sources = list(pool.map(download, packages))
        for package, source in zip(packages, sources):
            print(f"Installing {package['package']} {package['version']}", flush=True)
            subprocess.run(['R', 'CMD', 'INSTALL', '--no-docs', '--no-help', '--no-demo', str(source)], check=True)
        expected = ','.join(f"{p['package']}='{p['version']}'" for p in packages)
        subprocess.run(['Rscript', '-e', f'want <- c({expected}); '
                        'stopifnot(all(vapply(names(want), function(p) packageDescription(p, fields="Version"), "") == want))'], check=True)


if __name__ == '__main__':
    main()
