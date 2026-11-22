"""Prepare a VMware CPU lab using a verified Ubuntu cloud-image VMDK."""

import argparse
import io
from pathlib import Path

import pycdlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("disk", type=Path, help="Bootable VMDK in the VM directory")
    parser.add_argument("public_key", type=Path)
    args = parser.parse_args()
    disk = args.disk.resolve(strict=True)
    key = args.public_key.read_text(encoding="ascii").strip()
    if not key.startswith("ssh-ed25519 ") or "\n" in key or "\r" in key:
        parser.error("expected a single OpenSSH Ed25519 public key")

    vmx = disk.parent / "sony-nbg-cpu.vmx"
    seed = disk.parent / "seed.iso"
    if vmx.exists() or seed.exists():
        parser.error("VMX or seed already exists")
    contents = {
        "meta-data": "instance-id: sony-nbg-cpu-01\nlocal-hostname: sony-nbg-cpu\n",
        "user-data": (
            "#cloud-config\n"
            "ssh_pwauth: false\n"
            "disable_root: true\n"
            "users:\n"
            "  - name: nbglab\n"
            "    shell: /bin/bash\n"
            "    sudo: ALL=(ALL) NOPASSWD:ALL\n"
            "    lock_passwd: true\n"
            "    ssh_authorized_keys:\n"
            f"      - {key.split()[0]} {key.split()[1]}\n"
        ),
    }
    iso = pycdlib.PyCdlib()
    iso.new(interchange_level=3, vol_ident="CIDATA", rock_ridge="1.09")
    for name, content in contents.items():
        data = content.encode("ascii")
        iso.add_fp(
            io.BytesIO(data), len(data), f"/{name.upper().replace('-', '_')};1",
            rr_name=name,
        )
    iso.write(str(seed))
    iso.close()

    vmx.write_text(
        f'''.encoding = "UTF-8"
config.version = "8"
virtualHW.version = "17"
displayName = "Sony NBG CPU Lab"
guestOS = "ubuntu-64"
numvcpus = "4"
memsize = "4096"
scsi0.present = "TRUE"
scsi0.virtualDev = "lsilogic"
scsi0:0.present = "TRUE"
scsi0:0.fileName = "{disk.name}"
sata0.present = "TRUE"
sata0:0.present = "TRUE"
sata0:0.deviceType = "cdrom-image"
sata0:0.fileName = "{seed.name}"
sata0:0.startConnected = "TRUE"
ethernet0.present = "TRUE"
ethernet0.connectionType = "nat"
ethernet0.virtualDev = "e1000"
ethernet0.addressType = "generated"
usb.present = "FALSE"
ehci.present = "FALSE"
usb_xhci.present = "FALSE"
sound.present = "FALSE"
floppy0.present = "FALSE"
isolation.tools.copy.disable = "TRUE"
isolation.tools.paste.disable = "TRUE"
isolation.tools.dnd.disable = "TRUE"
serial0.present = "TRUE"
serial0.fileType = "file"
serial0.fileName = "boot.log"
mks.enable3d = "FALSE"
''',
        encoding="ascii",
    )
    print(vmx)


if __name__ == "__main__":
    main()
