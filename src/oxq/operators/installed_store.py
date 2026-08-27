"""Safe installed-release publication and byte snapshots."""
from __future__ import annotations
import hashlib, json, os, shutil, stat, tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast
from jsonschema import Draft202012Validator
from oxq.operators.formats import canonical_json_bytes, safe_relative_path, sha256_bytes, strict_json_object
from oxq.operators.resources import materialize_operator_install_profile
from oxq.operators.safe_files import fsync_directory, read_regular_file, replace_directory, write_file

MARKER = "installed-release.json"
@dataclass(frozen=True)
class InstalledRelease:
    provider: str; release: str; target: str; path: Path; trust_state: str; certification_state: str; operators: tuple[tuple[str,str],...]
@dataclass(frozen=True)
class InstalledOperator:
    release: InstalledRelease; binding: Mapping[str,object]; manifest: Mapping[str,object]; certified_cases: tuple[Mapping[str,object],...]
@dataclass(frozen=True)
class InstalledReleaseSnapshot:
    release: InstalledRelease; marker: Mapping[str,object]; release_index: bytes; bundle: bytes; publication_files: Mapping[str,bytes]; manifest_files: Mapping[str,bytes]; baseline_files: Mapping[str,bytes]; wheel_snapshots: Mapping[str,Path]

class InstalledReleaseStore:
 def __init__(self, home: str|Path|None=None): self.home=Path(home if home is not None else os.getenv("OPEN_XQUANT_OPERATOR_HOME", "~/.config/open-xquant/operator-releases")).expanduser().resolve()
 def publish(self, staging_dir: Path, marker: Mapping[str,object])->InstalledRelease:
  raw,ident,files=_marker(marker); src=Path(staging_dir).resolve(strict=True); vals=_check(src,files,(src/MARKER).is_file())
  dest=self.home/ident[0]/ident[1]/ident[2]; self.home.mkdir(parents=True,exist_ok=True)
  with _lock(self.home):
   if os.path.lexists(dest):
    old=self._read(dest)
    if _all(old.path)=={**vals,MARKER:raw}: return old
    raise ValueError("installed release conflict")
   dest.parent.mkdir(parents=True,exist_ok=True); tmp=Path(tempfile.mkdtemp(prefix=".staging-",dir=dest.parent))
   try:
    for n,v in vals.items():
     p=tmp.joinpath(*safe_relative_path(n).parts); p.parent.mkdir(parents=True,exist_ok=True); write_file(p,v)
    write_file(tmp/MARKER,raw); fsync_directory(tmp); replace_directory(tmp,dest); tmp=None
   finally:
    if tmp is not None: shutil.rmtree(tmp,ignore_errors=True)
  return self._read(dest)
 def list(self)->tuple[InstalledRelease,...]:
  if not self.home.is_dir(): return ()
  out=[]
  for p in self.home.glob("*/*/*"):
   try:
    if p.is_dir() and not p.is_symlink() and (p/MARKER).is_file(): out.append(self._read(p))
   except (OSError,ValueError): pass
  return tuple(sorted(out,key=lambda x:(x.provider,x.release,x.target)))
 def get(self,provider:str,release:str,target:str|None=None)->InstalledRelease:
  found=[x for x in self.list() if x.provider==provider and x.release==release and (target is None or x.target==target)]
  if len(found)!=1: raise ValueError("installed release is unavailable or ambiguous")
  return found[0]
 def resolve_operator(self,operator_id:str,operator_version:str,provider:str,provider_release:str)->InstalledOperator:
  r=self.get(provider,provider_release)
  with self.snapshot(r) as s:
   stem=f"{operator_id}@{operator_version}"; b=strict_json_object(s.publication_files[f"bindings/{stem}.binding.json"]); m=strict_json_object(s.manifest_files[f"{stem}.operator.json"])
   cases=[]
   for raw in s.baseline_files.values():
    for c in strict_json_object(raw).get("cases",[]):
     if isinstance(c,dict) and c.get("operator_id")==operator_id and c.get("operator_version")==operator_version: cases.append(MappingProxyType(dict(c)))
   return InstalledOperator(r,MappingProxyType(b),MappingProxyType(m),tuple(cases))
 @contextmanager
 def snapshot(self,release:InstalledRelease)->Iterator[InstalledReleaseSnapshot]:
  current=self._read(release.path); marker=_readmarker(release.path); _,_,files=_marker(marker); vals=_check(release.path,files,True); td=Path(tempfile.mkdtemp(prefix="oxq-installed-release-"))
  try:
   wheels={}
   for n,v in vals.items():
    if n.startswith("wheels/"):
     p=td/Path(n).name; write_file(p,v); wheels[n[7:]]=p
   yield InstalledReleaseSnapshot(current,MappingProxyType(marker),vals[marker["release_index"]["path"]],vals[marker["bundle"]["path"]],MappingProxyType({n[12:]:v for n,v in vals.items() if n.startswith("publication/")}),MappingProxyType({n[10:]:v for n,v in vals.items() if n.startswith("manifests/")}),MappingProxyType({n[10:]:v for n,v in vals.items() if n.startswith("baselines/")}),MappingProxyType(wheels))
  finally: shutil.rmtree(td,ignore_errors=True)
 def _read(self,path:Path)->InstalledRelease:
  m=_readmarker(path); _,i,f=_marker(m); _check(path,f,True); ops=[]
  for n in f:
   if n.startswith("manifests/"):
    v=strict_json_object(read_regular_file(path/n)); ops.append((cast(str,v["operator_id"]),cast(str,v["operator_version"])))
  return InstalledRelease(i[0],i[1],i[2],path.resolve(),cast(str,m["trust_state"]),cast(str,m["certification_state"]),tuple(ops))

def _marker(m):
 try:
  v=dict(m); _validator().validate(v); raw=canonical_json_bytes(v); t=v["target"]
  if not all(_path_component(value) for value in (v["provider"],v["release"],t["python_tag"],t["abi_tag"],t["platform_tag"])): raise ValueError
  ident=(v["provider"],v["release"],"-".join(t[x] for x in ("python_tag","abi_tag","platform_tag"))); items=v["files"]; f={x["path"]:x for x in items}
  if len(f)!=len(items) or MARKER in f or any(f.get(v[x]["path"])!=v[x] for x in ("release_index","bundle")): raise ValueError
  return raw,ident,f
 except Exception: raise ValueError("installed release marker is invalid") from None
def _path_component(value):
 return isinstance(value,str) and value not in {"",".",".."} and not any(char in value for char in ("/","\\","\x00","\r","\n")) and not any(ord(char)<32 for char in value)
def _validator():
 with materialize_operator_install_profile() as p: return Draft202012Validator(json.loads(p["installed_release"].read_text()))
def _all(root):
 out={}
 for p in root.rglob("*"):
  s=p.lstat()
  if stat.S_ISLNK(s.st_mode) or (not stat.S_ISDIR(s.st_mode) and not stat.S_ISREG(s.st_mode)): raise ValueError("unsafe path")
  if stat.S_ISREG(s.st_mode): out[p.relative_to(root).as_posix()]=read_regular_file(p)
 return out
def _check(root,files,marker):
 vals=_all(root); allowed=set(files)|({MARKER} if marker else set())
 if set(vals)!=allowed: raise ValueError("installed release file set is not exact")
 for n,d in files.items():
  if len(vals[n])!=d["size_bytes"] or sha256_bytes(vals[n])!=d["digest"]: raise ValueError("installed release file digest or size is invalid")
 for tree in _readmarker(root).get("trees",[]) if marker else []:
  pref=tree["path"]+"/"; selected={n[len(pref):]:v for n,v in vals.items() if n.startswith(pref)}; h=hashlib.sha256()
  for n,v in sorted(selected.items()): h.update(n.encode());h.update(b"\0");h.update(sha256_bytes(v).encode());h.update(b"\n")
  if sum(map(len,selected.values()))!=tree["size_bytes"] or "sha256:"+h.hexdigest()!=tree["digest"]: raise ValueError("tree digest is invalid")
 return {n:vals[n] for n in files}
def _readmarker(p): return strict_json_object(read_regular_file(p/MARKER))
@contextmanager
def _lock(home):
 fd=os.open(home/".installed-release.lock",os.O_CREAT|os.O_RDWR,0o600)
 try:
  if os.name!="nt":
   import fcntl; fcntl.flock(fd,fcntl.LOCK_EX)
  else:
   import msvcrt, time
   os.lseek(fd,0,os.SEEK_END)
   if os.lseek(fd,0,os.SEEK_CUR)==0: os.write(fd,b"\0")
   os.lseek(fd,0,os.SEEK_SET)
   while True:
    try: msvcrt.locking(fd,msvcrt.LK_NBLCK,1); break
    except OSError: time.sleep(0.05)
  yield
 finally:
  if os.name!="nt":
   import fcntl; fcntl.flock(fd,fcntl.LOCK_UN)
  else:
   import msvcrt
   os.lseek(fd,0,os.SEEK_SET); msvcrt.locking(fd,msvcrt.LK_UNLCK,1)
  os.close(fd)
