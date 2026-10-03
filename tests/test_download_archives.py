import functools
import hashlib
import http.server
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest

SPEC = importlib.util.spec_from_file_location(
    'archive_download', Path(__file__).resolve().parents[1] / 'downloads/download.py')
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


def entry(name, data):
    return {'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


class ArchiveDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        (self.source / 'parts').mkdir(parents=True)
        self.output = self.root / 'output'
        self.content = b'first archive section\x00second archive section\xff'
        blocks = [self.content[:20], self.content[20:]]
        parts = []
        for index, block in enumerate(blocks):
            name = f'archive.part-{index:03d}'
            (self.source / 'parts' / name).write_bytes(block)
            parts.append(entry(name, block))
        self.manifest = {'format': 1, 'artifacts': [{**entry('archive.tar.gz', self.content), 'parts': parts}]}
        self.requests = []
        requests = self.requests

        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                super().do_GET()

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(
            ('127.0.0.1', 0), functools.partial(Handler, directory=str(self.source)))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.url = f'http://127.0.0.1:{self.server.server_port}/'

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_http_download_resumes_and_replaces_only_corrupt_cached_parts(self):
        download.restore(self.manifest, self.output, base_url=self.url)
        archive = self.output / 'archive.tar.gz'
        self.assertEqual(archive.read_bytes(), self.content)
        self.assertEqual(len(self.requests), 2)
        download.restore(self.manifest, self.output, base_url=self.url)
        self.assertEqual(len(self.requests), 2)
        archive.unlink()
        (self.output / '.parts/archive.part-000').write_bytes(b'corrupt cache')
        download.restore(self.manifest, self.output, base_url=self.url)
        self.assertEqual(archive.read_bytes(), self.content)
        self.assertEqual(self.requests[-1], '/parts/archive.part-000')
        self.assertEqual(len(self.requests), 3)

    def test_corrupt_remote_part_does_not_publish_archive(self):
        part = self.source / 'parts/archive.part-000'
        part.write_bytes(b'x' * part.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            download.restore(self.manifest, self.output, base_url=self.url)
        self.assertFalse((self.output / 'archive.tar.gz').exists())

    def test_wrong_final_hash_does_not_publish_archive(self):
        self.manifest['artifacts'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Assembled archive SHA-256 mismatch'):
            download.restore(self.manifest, self.output, source_dir=self.source)
        self.assertFalse((self.output / 'archive.tar.gz').exists())

    def test_unsafe_paths_are_rejected_before_writing(self):
        self.manifest['artifacts'][0]['name'] = '../outside.tar.gz'
        with self.assertRaisesRegex(ValueError, 'Unsafe filename'):
            download.restore(self.manifest, self.output, source_dir=self.source)
        self.assertFalse(self.output.exists())

    def test_existing_nonmatching_archive_is_preserved(self):
        self.output.mkdir()
        path = self.output / 'archive.tar.gz'
        path.write_bytes(b'existing user data')
        with self.assertRaisesRegex(ValueError, 'move it aside'):
            download.restore(self.manifest, self.output, source_dir=self.source)
        self.assertEqual(path.read_bytes(), b'existing user data')


if __name__ == '__main__':
    unittest.main()
