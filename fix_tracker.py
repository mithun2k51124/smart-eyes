with open(r"c:\Users\mithun\Desktop\smart eyes\fish_monitor.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace('TRACKER_TYPE    = "bytetrack"', 'TRACKER_TYPE    = "bytetrack.yaml"')
with open(r"c:\Users\mithun\Desktop\smart eyes\fish_monitor.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Patched: tracker -> bytetrack.yaml")
