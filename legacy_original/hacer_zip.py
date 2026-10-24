import os, zipfile
BASE = r"E:\AIMS-sprint\entrega\submission"
OUT  = r"E:\AIMS-sprint\entrega\submission.zip"
if os.path.exists(OUT): os.remove(OUT)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(BASE):
        for f in sorted(fs):
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, BASE).replace(os.sep, "/"))
print("zip:", OUT, os.path.getsize(OUT), "bytes")
with zipfile.ZipFile(OUT) as z: print("contenido:", z.namelist())
