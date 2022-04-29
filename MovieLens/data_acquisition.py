from os.path import join
from sys import stdout
from requests import get
from zipfile import ZipFile
import gzip
from shutil import copyfileobj


def data_acquisition(data_dir, url):

    """
    Download data with request library and
    extract dataframes with ZipFile and gzip
    """

    # get zip file name from url
    filename = url.split('/')[-1]
    file_path = join(data_dir, filename)

    # get zip file using requests library
    r = get(url, stream=True)

    if r.ok:
        # show download progress
        with open(file_path, "wb") as f:
            print("Downloading %s" % url)
            response = get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:
                # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                if file_path.endswith('.zip'):
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                        stdout.flush()
                else:
                    for chunk in r.raw.stream(1024, decode_content=False):
                        if chunk:
                            dl += len(chunk)
                            f.write(chunk)
                            done = int(50 * dl / total_length)
                            stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                            stdout.flush()

        # extract data
        if file_path.endswith('.zip'):
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        else:
            with gzip.open(file_path, 'rb') as f_in:
                with open(file_path[:-3], 'wb') as f_out:
                    copyfileobj(f_in, f_out)

    else:
        print(f'Download failed: status code {r.status_code}\n{r.text}')
