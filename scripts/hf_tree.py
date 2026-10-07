"""List a Hugging Face dataset subtree (all pages) and sum file sizes by path prefix, without downloading.

Usage: python scripts/hf_tree.py REPO PATH OUT.json [DEPTH]
"""
import json,sys,urllib.request,re,collections
repo,path,out=sys.argv[1],sys.argv[2],sys.argv[3]
url=f"https://huggingface.co/api/datasets/{repo}/tree/main/{path}?recursive=true&expand=false"
files=[]
while url:
    r=urllib.request.urlopen(url); files+=json.load(r)
    m=re.search(r'<([^>]+)>;\s*rel="next"',r.headers.get('Link') or ''); url=m.group(1) if m else None
json.dump(files,open(out,'w'))
tot=collections.Counter(); cnt=collections.Counter()
for f in files:
    if f['type']!='file': continue
    p=f['path'].split('/'); key='/'.join(p[:min(len(p)-1,int(sys.argv[4]) if len(sys.argv)>4 else 3)])
    sz=(f.get('lfs') or {}).get('size',f.get('size',0)); tot[key]+=sz; cnt[key]+=1
for k in sorted(tot): print(f"{k:70s} {cnt[k]:6d} files {tot[k]/1e9:9.3f} GB")
print('TOTAL',len(files),'entries',sum(tot.values())/1e9,'GB')
