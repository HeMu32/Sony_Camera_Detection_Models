"""One-shot extraction of the AF subject-detection models from the three BODYDATA.DAT files.

Pipeline: DAT (U-FU container) -> FDAT (AES-CBC, CXD90057_k8) -> firmware.tar
          -> nflasha15 (ext2 /usr) -> /usr/sabin/*.bin and /usr/data/ai_dict.

Usage:
    python out/models/extract_all.py
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
FWTOOL = os.path.join(ROOT, 'FwRev', 'tools', 'fwtool.py-a7iv')
SEVENZIP = r'C:\Program Files\7-Zip\7z.exe'
WORK = os.path.join(HERE, 'work')

TARGETS = {
    '1m2': os.path.join(ROOT, 'FwRev', 'CamFw', '1M2_v400_BODYDATA.DAT'),
    '7m4': os.path.join(ROOT, 'FwRev', 'CamFw', '7M4_v602_BODYDATA.DAT'),
    'r5':  os.path.join(ROOT, 'FwRev', 'CamFw', '7RM5_v400_BODYDATA.DAT'),
}


def run(cmd):
    print('+', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def sevenzip(*args):
    run([SEVENZIP, *args])


def extract_firmware_tar(tag, dat_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tar = os.path.join(out_dir, 'firmware.tar')
    if os.path.exists(tar):
        print('have', tar)
        return tar
    sys.path.insert(0, FWTOOL)
    from fwtool.sony import dat as datmod, fdat

    with open(dat_path, 'rb') as f:
        d = datmod.readDat(f)
    name, dec = fdat.decryptFdat(d.firmwareData)
    print(tag, 'crypter', name)
    tmp = os.path.join(out_dir, '_decrypted.fdat')
    with open(tmp, 'wb') as o:
        while True:
            chunk = dec.read(1 << 20)
            if not chunk:
                break
            o.write(chunk)
    with open(tmp, 'rb') as f:
        contents = fdat.readFdat(f)
    with open(tar, 'wb') as o:
        contents.firmware.seek(0)
        shutil.copyfileobj(contents.firmware, o, 1 << 20)
    os.remove(tmp)
    return tar


def main():
    for tag, dat in TARGETS.items():
        out_dir = os.path.join(WORK, tag)
        tar = extract_firmware_tar(tag, dat, out_dir)
        part = os.path.join(out_dir, 'part')
        os.makedirs(part, exist_ok=True)
        nf15 = os.path.join(part, 'nflasha15')
        if not os.path.exists(nf15):
            sevenzip('e', tar, '-o' + part, '0700_part_image\\dev\\nflasha15', '-y')
        sabin = os.path.join(out_dir, 'sabin')
        os.makedirs(sabin, exist_ok=True)
        sevenzip('e', nf15, '-o' + sabin, 'sabin\\*', '-y')
        ai = os.path.join(out_dir, 'ai_dict')
        os.makedirs(ai, exist_ok=True)
        sevenzip('e', nf15, '-o' + ai, 'data\\ai_dict\\*', '-r', '-y')
    print('done')


if __name__ == '__main__':
    main()
