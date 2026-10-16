import tarfile
from pathlib import Path
from PIL import Image

archive = Path("data/raw/rvl_cdip/data/rvl-cdip.tar.gz")
relative_path = "imagesq/q/o/c/qoc54c00/80035521.tif"
member_name = f"images/{relative_path}"

with tarfile.open(archive, "r:gz") as tar:
    member = tar.getmember(member_name)
    if not member.isfile():
        raise ValueError("Invlid image member")
    source = tar.extractfile(member)
    if source is None or not member.isfile():
        raise ValueError("Invalid image member")
    image_bytes = source.read()

# print(f"Extracted {len(image_bytes)} bytes")

output = Path("data/raw/rv1_cdip/samples/80035521.tif")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(image_bytes)

with Image.open(output) as img:
    img.verify()

print(f"Valid TIFF saved: {output}") 

