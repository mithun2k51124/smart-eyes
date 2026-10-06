with open(r"c:\Users\mithun\Desktop\smart eyes\fish_monitor.py", "r", encoding="cp1252") as f:
    content = f.read()
content = content.replace("\x97", "-").replace("\x96", "-").replace("\x92", "'")
if "# -*- coding:" not in content[:60]:
    content = "# -*- coding: utf-8 -*-\n" + content
with open(r"c:\Users\mithun\Desktop\smart eyes\fish_monitor.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done: fish_monitor.py re-saved as UTF-8.")
