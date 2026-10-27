#!/usr/bin/env python3
"""Non-destructive split validation and reproducible manifest creation for Warwick QU."""
from pathlib import Path
import argparse, csv, random, shutil
EXTS={".png",".bmp",".bm",".jpg",".jpeg",".tif",".tiff"}
def files_in(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTS)
def canonical_mask_stem(path):
    stem=Path(path).stem
    return stem[:-5] if stem.lower().endswith("_anno") else stem
def get_pairs(root, split):
    imdir=root/split/"img"; mdir=root/split/"mask"
    if not imdir.is_dir() or not mdir.is_dir(): return []
    images=files_in(imdir); masks=files_in(mdir); mask_map={}
    for p in masks:
        k=canonical_mask_stem(p)
        if k in mask_map: raise ValueError(f"Duplicate masks for {k}: {mask_map[k]} and {p}")
        mask_map[k]=p
    pairs=[]; missing=[]; used=set()
    for image in images:
        mask=mask_map.get(image.stem)
        if mask is None: missing.append(image.name)
        else: pairs.append((image,mask)); used.add(mask)
    extra=[p.name for p in masks if p not in used]
    if missing or extra: raise ValueError(f"{split}: images without masks={missing[:10]}, masks without images={extra[:10]}")
    return pairs
def write_manifest(root, split_pairs, path):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["split","image","mask","seed"]); w.writeheader()
        for split,pairs,seed in split_pairs:
            for image,mask in pairs:
                w.writerow({"split":split,"image":image.relative_to(root).as_posix(),"mask":mask.relative_to(root).as_posix(),"seed":seed})
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True,help="Warwick_QU_Dataset_Binary_FIXED")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--existing-val-policy",choices=["preserve","resplit"],default="preserve")
    ap.add_argument("--write",action="store_true",help="Write split_manifest.csv")
    ap.add_argument("--manifest",default="split_manifest.csv")
    ap.add_argument("--copy-mode",choices=["copy","hardlink"],default="copy")
    args=ap.parse_args(); root=Path(args.root).expanduser().resolve()
    train=get_pairs(root,"train"); val=get_pairs(root,"val"); test=get_pairs(root,"test")
    if not train or not test: raise ValueError("Expected non-empty train and test image/mask pairs")
    print(f"Current pairs: train={len(train)}, val={len(val)}, test={len(test)}, total={len(train)+len(val)+len(test)}")
    if args.existing_val_policy=="preserve" or not val:
        print("Preserving current train/val/test splits; test remains untouched.")
        if args.write:
            write_manifest(root,[("train",train,args.seed),("val",val,args.seed),("test",test,args.seed)],root/args.manifest)
            print("Manifest:",root/args.manifest)
        return
    # Repartition only current train+validation; never touch test.
    pool=train+val; rng=random.Random(args.seed); rng.shuffle(pool); n_val=max(1,round(len(pool)/9))
    new_val=pool[:n_val]; new_train=pool[n_val:]
    print(f"Proposed non-test repartition: train={len(new_train)}, val={len(new_val)}, test={len(test)}")
    if not args.write: print("Dry run. Pass --write to rebuild train/val and save a manifest."); return
    stage=root/".split_staging"
    if stage.exists(): shutil.rmtree(stage)
    for split,pairs in (("train",new_train),("val",new_val)):
        for sub in ("img","mask"): (stage/split/sub).mkdir(parents=True,exist_ok=True)
        for image,mask in pairs:
            for src,sub in ((image,"img"),(mask,"mask")):
                dst=stage/split/sub/src.name
                if args.copy_mode=="hardlink":
                    try: dst.hardlink_to(src)
                    except OSError: shutil.copy2(src,dst)
                else: shutil.copy2(src,dst)
    for split in ("train","val"):
        dst=root/split
        if dst.exists(): shutil.rmtree(dst)
        shutil.move(str(stage/split),str(dst))
    shutil.rmtree(stage,ignore_errors=True)
    train=get_pairs(root,"train"); val=get_pairs(root,"val")
    write_manifest(root,[("train",train,args.seed),("val",val,args.seed),("test",test,args.seed)],root/args.manifest)
    print(f"Written: train={len(train)}, val={len(val)}, test={len(test)}")
    print("Manifest:",root/args.manifest)
if __name__=="__main__": main()
