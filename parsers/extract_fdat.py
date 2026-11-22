"""Extract firmware.tar + updater.img from a decrypted FDAT (r5/plain.bin) or DAT."""
import os, sys, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'FwRev', 'tools', 'fwtool.py-a7iv'))
from fwtool.sony import fdat, dat as datmod

src = sys.argv[1]
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)

with open(src, 'rb') as f:
    magic = f.read(8)
    f.seek(0)
    if magic == b'UDTRFIRM':
        work = open(src, 'rb')
        close_work = True
    else:
        d = datmod.readDat(f)
        name, dec = fdat.decryptFdat(d.firmwareData)
        print('crypter:', name)
        # ChunkedFile cannot seek; spill decrypted stream to a real file first
        tmp = os.path.join(outdir, '_decrypted.fdat')
        with open(tmp, 'wb') as o:
            while True:
                chunk = dec.read(1 << 20)
                if not chunk:
                    break
                o.write(chunk)
        work = open(tmp, 'rb')
        close_work = True
    contents = fdat.readFdat(work)
    print('model=0x%X region=%d version=%s accessory=%s' % (contents.model, contents.region, contents.version, contents.isAccessory))
    with open(os.path.join(outdir, 'firmware.tar'), 'wb') as o:
        contents.firmware.seek(0); shutil.copyfileobj(contents.firmware, o, 1 << 20)
    with open(os.path.join(outdir, 'updater.img'), 'wb') as o:
        contents.fs.seek(0); shutil.copyfileobj(contents.fs, o, 1 << 20)
    if close_work:
        work.close()
print('done')
