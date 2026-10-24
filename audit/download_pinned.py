"""Download only the frozen public inputs; no API key is bundled or requested."""
import argparse,hashlib,json
from pathlib import Path
from huggingface_hub import hf_hub_download

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=Path(__file__).with_name('download_manifest.json'))
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text());args.data.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(manifest['files'],1):
        path=args.data/row['path']
        if not path.exists() or digest(path)!=row['sha256']:
            path=Path(hf_hub_download(manifest['repo_id'],row['path'],repo_type='dataset',
                      revision=manifest['revision'],local_dir=args.data))
        if digest(path)!=row['sha256']:raise RuntimeError('Content hash mismatch: '+row['path'])
        if i%25==0:print(f'checked {i}/{len(manifest["files"])}',flush=True)
    print('All pinned files match their SHA-256 manifest.')

if __name__=='__main__':main()
